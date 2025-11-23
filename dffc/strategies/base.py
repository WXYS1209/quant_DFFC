"""
Improved vectorbt dual asset rebalancing strategy
Based on MarketCalls tutorial best practices

Features:
- Multi-asset rebalancing framework
- Configurable rebalancing frequency (D/W/M/Q/Y)
- Gradual weight adjustment with tolerance
- Simple trade execution delay support via weight matrix shifting
- Comprehensive performance analysis and visualization

Trade Delay Implementation:
- Simple and efficient: directly shifts weight matrix by N days
- T+0 (trade_delay=0): Immediate execution, suitable for stocks
- T+1 (trade_delay=1): Next-day execution, suitable for funds
- T+2+ (trade_delay=2+): Multi-day delay, suitable for special instruments

Usage:
    strategy = DualReallocationStrategy(prices=data, ...)
    res = strategy.run_backtest(
        initial_cash=100000,
        fees=0.001,
        trade_delay=1  # T+1 for funds
    )
    portfolio = res.portfolio
    rebalances = res.actual_rebalances
    weights = res.actual_weights
"""
import numpy as np
import pandas as pd
import vectorbt as vbt
from abc import ABC, abstractmethod
from types import SimpleNamespace
from vectorbt.utils.datetime_ import to_tzaware_datetime
from vectorbt import _typing as tp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set vectorbt configuration
vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True

pd.set_option('future.no_silent_downcasting', True)

class Strategy(ABC):
    """
    vectorbt-based strategy base class
    
    Abstract base class for all strategies, defines basic interface and common functionality
    """
    
    def __init__(self, 
                 prices: tp.ArrayLike, 
                 **kwargs):
        """
        Initialize strategy base class
        
        Args:
            prices: DataFrame, price data
            **kwargs: other strategy-specific parameters
        """
        self.prices = prices
        self.backtest_prices = self.prices.copy()

    @abstractmethod
    def run_backtest(self, 
                     **kwargs):
        """
        Run backtest - must be implemented by subclasses
        
        Args:
            initial_cash: float, initial capital
            fees: float, trading fee rate
            
        Returns:
            portfolio: vectorbt Portfolio object
        """
        pass
    
class ReallocationStrategy(Strategy):
    """
    General rebalancing strategy base class
    
    Multi-asset weight rebalancing framework that provides common rebalancing logic
    """
    
    def __init__(self, 
                 prices: tp.ArrayLike,
                 # rebalance_freq: str = 'D',
                 **kwargs) -> None:
        """
        Initialize rebalancing strategy
        
        Args:
            prices: DataFrame, price data (supports multi-asset)
            rebalance_freq: str, rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            adjust_factor: float, weight adjustment factor (0-1), 1 means immediately adjust to target weight
            **kwargs: other strategy parameters
        """
        super().__init__(prices, **kwargs)
        # self.rebalance_freq = rebalance_freq

        # Weight-related attributes, set by subclasses
        self.target_weights = None
        
    # def _create_rebalance_schedule(self):
    #     """Create rebalancing schedule"""
    #     # return pd.Series(True, index=self.backtest_prices.index)
    #     if self.rebalance_freq == 'D':
    #         # Daily rebalancing
    #         return pd.Series(True, index=self.backtest_prices.index)
    #     elif self.rebalance_freq == 'W':
    #         # Weekly rebalancing
    #         week_starts = self.backtest_prices.groupby(pd.Grouper(freq='W-MON')).first()
    #         return self.backtest_prices.index.isin(week_starts.index)
    #     elif self.rebalance_freq == 'M':
    #         # Monthly rebalancing
    #         month_starts = self.backtest_prices.groupby(pd.Grouper(freq='M')).first()
    #         return self.backtest_prices.index.isin(month_starts.index)
    #     elif self.rebalance_freq == 'Q':
    #         # Quarterly rebalancing
    #         quarter_starts = self.backtest_prices.groupby(pd.Grouper(freq='Q')).first()
    #         return self.backtest_prices.index.isin(quarter_starts.index)
    #     elif self.rebalance_freq == 'Y':
    #         # Annual rebalancing
    #         year_starts = self.backtest_prices.groupby(pd.Grouper(freq='Y')).first()
    #         return self.backtest_prices.index.isin(year_starts.index)
    #     else:
    #         raise ValueError(f"Unsupported rebalancing frequency: {self.rebalance_freq}")

    @staticmethod
    def _parse_numeric_param(
        value: tp.ArrayLike,
        name: str
    ) -> tp.ArrayLike:

        if value is not None:
            if isinstance(value, tp.Number):
                arr = np.asarray([value], dtype=float)
            else:
                arr = np.asarray(value, dtype=float).flatten()
        else:
            raise ValueError(f"{name} cannot be None")
        
        return arr
    
    @staticmethod
    def _parse_array_param(
        value: tp.AnyArray,
        name: str
    ) -> tp.ArrayLike:

        if value is None:
            raise ValueError(f"{name} cannot be None")

        arr = np.asarray(value, dtype=float).flatten()

        if arr.ndim == 1:
            return arr[np.newaxis, :]

        if arr.ndim == 2:
            return arr

        raise ValueError(f"{name} must be 1D or 2D array-like")
    
    @staticmethod
    def _apply_gradual_adjustment(weights: tp.ArrayLike,
                                  rb_mask: tp.ArrayLike,
                                  adjust_factor: float = 1.0,
                                  tolerance: float = 0.01):
        """
        Apply gradual adjustment logic (supports multi-asset)
        
        Args:
            target_weights: DataFrame, desired target weights over time
            rb_mask: Series or array, optional rebalancing schedule hints
            tolerance: float, weight difference tolerance
            
        Returns:
            actual_weights: DataFrame, actual weight series
            actual_rebalances: Series, actual rebalancing schedule
        """
        if weights is None:
            raise ValueError("weights must be provided for gradual adjustment")

        if weights.empty:
            raise ValueError("weights cannot be empty")

        weights = weights.copy()

        if isinstance(rb_mask, np.ndarray):
            rb_mask = pd.Series(rb_mask, index=weights.index)

        actual_weights = pd.DataFrame(np.nan,
                                      index=weights.index,
                                      columns=weights.columns)
        actual_weights.iloc[0] = weights.iloc[0].copy()

        actual_rb_mask = pd.Series(False, index=weights.index)
        actual_rb_mask.iloc[0] = True

        for i in range(1, len(weights)):
            current = actual_weights.iloc[i - 1]
            desired = weights.iloc[i]

            if not rb_mask.iloc[i]:
                actual_weights.iloc[i] = current
                continue

            weight_diff = desired - current
            min_diff = np.abs(weight_diff).min()
            if min_diff >= tolerance:
                adjusted = current + weight_diff * adjust_factor
                actual_weights.iloc[i] = adjusted
                actual_rb_mask.iloc[i] = True
            else:
                actual_weights.iloc[i] = current

        return actual_weights, actual_rb_mask

    @staticmethod
    def _apply_trade_delay(weights: pd.DataFrame,
                           rb_mask: pd.Series,
                           trade_delay: int) -> tuple[pd.DataFrame, pd.Series]:
        """Shift weights and rebalance mask to model trade execution delay."""
        if trade_delay <= 0:
            return weights, rb_mask

        delayed_weights = weights.shift(trade_delay) # .ffill()
        delayed_weights.iloc[:trade_delay] = weights.iloc[:trade_delay].values

        delayed_rb_mask = rb_mask.shift(trade_delay, fill_value=False)
        delayed_rb_mask.iloc[0] = rb_mask.iloc[0]
        delayed_rb_mask.iloc[trade_delay] = False

        return delayed_weights, delayed_rb_mask
    
    @staticmethod
    def _weights_to_orders(weights: pd.DataFrame,
                           rb_mask: pd.Series) -> pd.DataFrame:
        """Convert actual weights and rebalance mask into TargetPercent order matrix."""
        orders = pd.DataFrame(np.nan,
                              index=weights.index,
                              columns=weights.columns)
        orders.loc[rb_mask, :] = weights.loc[rb_mask, :]
        return orders

    @staticmethod
    def _prepare_orders(target_weights: tp.ArrayLike,
                        rb_mask: tp.ArrayLike,
                        adjust_factor: float,
                        tolerance: float,
                        trade_delay: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Helper to build orders, actual weights, and rebalance mask from target weights."""
        ga_weights, ga_rb_mask = ReallocationStrategy._apply_gradual_adjustment(
            weights=target_weights,
            rb_mask=rb_mask,
            adjust_factor=adjust_factor,
            tolerance=tolerance
        )

        actual_weights, rebalance_mask = ReallocationStrategy._apply_trade_delay(
            ga_weights,
            ga_rb_mask,
            trade_delay
        )

        orders = ReallocationStrategy._weights_to_orders(actual_weights, rebalance_mask)
        return orders, actual_weights, rebalance_mask
    
    @abstractmethod
    def _generate_target_weights(self, param_comb, **kwargs):
        """
        生成目标权重序列 - 子类必须实现
        
        Returns:
            target_weights: DataFrame, 目标权重序列（每行权重之和应为1）
        """
        pass
    
    @abstractmethod
    def _get_param_combinations(self):
        """Return (label, context) pairs for each parameter combination.

        Default implementation runs a single combination with None context."""
        pass
    
    def _process_combo(self,
                       pid: int,
                       combo: dict,
                       rb_mask: pd.Series,
                       trade_delay: int,
                       group_by: str) -> tuple[int, dict, pd.DataFrame]:
        """Helper to generate orders for a single parameter combination."""
        targets_weights = self._generate_target_weights(combo)
        self.target_weights = targets_weights
        orders, _, _ = ReallocationStrategy._prepare_orders(
            target_weights=targets_weights,
            rb_mask=rb_mask,
            adjust_factor=combo.get('adjust_factor', None),
            tolerance=combo.get('tolerance', None),
            trade_delay=trade_delay
        )

        orders.columns = pd.MultiIndex.from_product([[pid], orders.columns], names=[group_by, 'symbol'])
        return pid, combo, orders

    @staticmethod
    def _normalise_param_value(value):
        """Convert parameter values into hashable / serialisable representations."""
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            flat = np.asarray(value, dtype=float).flatten().tolist()
            return f"[{', '.join(f'{v:.6g}' for v in flat)}]"
        if isinstance(value, (list, tuple)):
            if any(isinstance(v, np.ndarray) for v in value):
                flat = np.asarray(value, dtype=float).flatten().tolist()
                return f"[{', '.join(f'{v:.6g}' for v in flat)}]"
            if len(value) > 1:
                return f"[{', '.join(str(v) for v in value)}]"
        return value

    def run_backtest(self,
                     start_date=None,
                     end_date=None,
                     group_by=None,
                     initial_cash=10_000.,
                     fees=0.,
                     trade_delay=0,
                     max_workers: int | None = None):
        """
        运行再平衡策略回测（支持多资产）
        
        Args:
            initial_cash: float, 初始资金
            fees: float, 交易费用率
            trade_delay: int, 交易执行延迟天数 (0=T+0, 1=T+1, 2=T+2, etc.)
                        基金推荐使用 trade_delay=1 (T+1)
            max_workers: Optional[int], 使用进程池处理参数组合的最大进程数。
                         None 表示使用默认值；≤1 则按顺序执行。
            
        Returns:
            SimpleNamespace: 结果对象，可通过 res.portfolio 访问（也支持 dict 风格 via vars(res)['portfolio']）
        """
        
        if start_date is not None:
            start_date = to_tzaware_datetime(start_date)
        if end_date is not None:
            end_date = to_tzaware_datetime(end_date)
        self.backtest_prices = self.prices.loc[start_date:end_date]

        combos = self._get_param_combinations()
        self.combos = combos

        if group_by is None:
            group_by = 'param_group' # True if len(combos) == 1 else 'param_group'
        
        pids = pd.Index(range(len(combos)), name=group_by)

        rb_mask = pd.Series(True, index=self.backtest_prices.index)
        # rb_mask = self._create_rebalance_schedule()
        
        backtest_prices_tiled = self.backtest_prices.vbt.tile(len(combos), keys=pids)

        orders_blocks: list[pd.DataFrame] = []
        pid_to_combo: dict[int, dict] = {}

        combo_results: list[tuple[int, dict, pd.DataFrame]] = []
        parallel_enabled = (max_workers is None or max_workers > 1) and len(combos) > 1

        if parallel_enabled:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_combo,
                        pid,
                        combos[pid],
                        rb_mask,
                        trade_delay,
                        group_by
                    ): pid
                    for pid in range(len(combos))
                }

                for future in tqdm(as_completed(futures), total=len(futures)):
                    combo_results.append(future.result())
        else:
            for pid, combo in enumerate(tqdm(combos)):
                combo_results.append(self._process_combo(pid, combo, rb_mask, trade_delay, group_by))

        combo_results.sort(key=lambda item: item[0])
        self.debug_combo_results = combo_results
        for pid, combo, orders in combo_results:
            orders_blocks.append(orders)
            pid_to_combo[pid] = combo
            
        orders_size = pd.concat(orders_blocks, axis=1).sort_index(axis=1)

        self.param_group_map = pid_to_combo

        # Change column names
        backtest_prices_tiled = backtest_prices_tiled.reindex(columns=orders_size.columns)

        param_keys = list(combos[0].keys())
        column_meta: dict[str, list] = {key: [] for key in param_keys}

        for pid, symbol in orders_size.columns:
            combo = pid_to_combo.get(pid, {})
            for key in param_keys:
                normalised = self._normalise_param_value(combo.get(key))
                column_meta[key].append(normalised)

        for key in reversed(param_keys):
            idx = pd.Index(column_meta[key], name=key)
            orders_size = orders_size.vbt.stack_index(idx)
            backtest_prices_tiled = backtest_prices_tiled.vbt.stack_index(idx)

        desired_order = [group_by, *param_keys, 'symbol']

        self.orders_size = (orders_size
                            .reorder_levels(desired_order, axis=1)
                            .sort_index(axis=1))
        self.backtest_prices_tiled = (backtest_prices_tiled
                                      .reorder_levels(desired_order, axis=1)
                                      .sort_index(axis=1))

        metadata_records = []
        for pid, combo in pid_to_combo.items():
            record = {'param_group': pid}
            record.update({key: self._normalise_param_value(combo.get(key)) for key in param_keys})
            metadata_records.append(record)
        self.param_metadata = (pd.DataFrame(metadata_records)
                                .set_index('param_group')
                                .sort_index())
        
        # 使用vectorbt运行回测
        portfolio = vbt.Portfolio.from_orders(
            close=self.backtest_prices_tiled,
            size=self.orders_size,
            size_type='targetpercent',
            group_by=group_by,
            cash_sharing=True,
            call_seq='auto',
            fees=fees,
            init_cash=initial_cash,
            freq='D',
            min_size=0.01,
            size_granularity=0.01
        )
        
        return portfolio
    
        
