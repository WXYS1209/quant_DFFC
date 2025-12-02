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
