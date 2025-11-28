from dffc.strategies import ReallocationStrategy
from dffc.backtesting import ReallocationBackTest
from dffc.holt_winters import HWDP, HW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vectorbt import _typing as tp
from itertools import product

class DualReallocationStrategy(ReallocationStrategy):
    """
    Dual Asset Rebalancing Strategy based on Holt-Winters
    
    Inherits from ReallocationStrategy, focuses on HW signal generation and dual asset weight allocation logic
    """
    
    def __init__(
        self, 
        prices: tp.ArrayLike, 
        threshold: tp.ArrayLike,
        up_weights: tp.ArrayLike = 0.2,
        down_weights: tp.ArrayLike = 0.8, 
        default_weights: tp.ArrayLike = 0.5,
        adjust_factor: tp.ArrayLike = 0.2,
        tolerance: tp.ArrayLike = 0.01,
        hw_params_list = None,
        **kwargs
    ):
        """
        Initialize dual asset rebalancing strategy
        
        Args:
            prices: DataFrame, price data (must have 2 columns)
            default_weights: list, default weights
            up_weights: list, weights for uptrend  
            down_weights: list, weights for downtrend
            threshold: float, hysteresis threshold
            hw_params_list: list of dict, HW parameters
                Format: [{"code": "code", "params": {"alpha": 0.1, "beta": 0.1, "gamma": 0.1, "season_length": 8}}, ...]
            **kwargs: other base class parameters (like adjust_factor, rebalance_freq etc.)
        """
        # Validate input data
        if len(prices.columns) != 2:
            raise ValueError("DualReallocationStrategy only supports dual assets, please provide 2-column price data")
        
        # Call parent class initialization
        super().__init__(prices, **kwargs)
        
        # Strategy specific parameters
        self._params = {
            'threshold': self._parse_numeric_param(threshold, 'threshold'),
            'adjust_factor': self._parse_numeric_param(adjust_factor, 'adjust_factor'),
            'default_weights': self._parse_numeric_param(default_weights, 'default_weights'),
            'up_weights': self._parse_numeric_param(up_weights, 'up_weights'),
            'down_weights': self._parse_numeric_param(down_weights, 'down_weights'),
            'tolerance': self._parse_numeric_param(tolerance, 'tolerance')
        }

        # Validate parameters
        self.hw_params_list = hw_params_list
        self._validate_hw_params()

    def _validate_hw_params(self):
        """Validate HW parameters configuration"""
        if self.hw_params_list is not None:
            if not isinstance(self.hw_params_list, list):
                raise ValueError("hw_params_list must be a list of dictionaries")
            
            if len(self.hw_params_list) != len(self.prices.columns):
                raise ValueError(f"hw_params_list must contain {len(self.prices.columns)} elements to match price columns")
            
            for i, param_dict in enumerate(self.hw_params_list):
                if not isinstance(param_dict, dict):
                    raise ValueError(f"hw_params_list[{i}] must be a dictionary")
                
                if 'code' not in param_dict or 'params' not in param_dict:
                    raise ValueError(f"hw_params_list[{i}] must contain 'code' and 'params' keys")
                
                params = param_dict['params']
                required_keys = ['alpha', 'beta', 'gamma', 'season_length']
                for key in required_keys:
                    if key not in params:
                        raise ValueError(f"hw_params_list[{i}]['params'] must contain '{key}' key")
    
    def _calculate_hw_signals(self):
        """Calculate Holt-Winters signals"""
        hw_signals = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        hw = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        if self.hw_params_list is not None:
            hw_params = {}
            for i, param_dict in enumerate(self.hw_params_list):
                col = self.prices.columns[i]
                params = param_dict['params']
                hw_params[col] = {
                    'alpha': params['alpha'],
                    'beta': params['beta'], 
                    'gamma': params['gamma'],
                    'season_length': params['season_length']  # Note: using season_length from external params
                }
                # print(f"  {col} ({param_dict['code']}): alpha={params['alpha']:.4f}, beta={params['beta']:.4f}, gamma={params['gamma']:.4f}, season={params['season_length']}")
        else:
            # Use default parameters
            print("Using default Holt-Winters parameters...")
            hw_params = {col: {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.1, 'season_length': 8} 
                        for col in self.prices.columns}

        for col in self.prices.columns:
            params = hw_params[col]
            hwdp_result = HWDP.run(
                self.prices[col].dropna(), 
                alpha=params['alpha'],
                beta=params['beta'], 
                gamma=params['gamma'],
                season_length=params['season_length'],
                multiplicative=True
            )
            hw_result = HW.run(
                self.prices[col].dropna(),
                alpha=params['alpha'],
                beta=params['beta'], 
                gamma=params['gamma'],
                season_length=params['season_length'],
                multiplicative=True
            )
            hw[col] = hw_result.hw
            hw_signals[col] = hwdp_result.hwdp
        self.hw = hw
        # self.hw_params_list = hw_params
        return hw_signals.loc[self.backtest_prices.index]
    
    def _generate_target_weights(self, param_combo):
        """Generate target weight series (based on hysteresis logic of HW signal difference)"""

        self.hw_signals = self._calculate_hw_signals()

        threshold = param_combo.get('threshold')
        adjust_factor = param_combo.get('adjust_factor')
        default_weights = param_combo.get('default_weights')
        up_weights = param_combo.get('up_weights')
        down_weights = param_combo.get('down_weights')

        # Calculate HW signals
        delta_hdp = - self.hw_signals.iloc[:, 0] + self.hw_signals.iloc[:, 1]

        # Symmetric hysteresis (Schmitt trigger) with two thresholds ±threshold
        # state ∈ {-1, 0, 1}; only flips when crossing the opposite threshold
        signals = pd.Series(0, index=self.backtest_prices.index, dtype=int)
        # Optionally set initial state if the very first delta is already beyond thresholds
        first_delta = delta_hdp.iloc[0]
        if first_delta >= threshold:
            signals.iloc[0] = 1
        elif first_delta <= -threshold:
            signals.iloc[0] = -1
        # Iterate and apply hysteresis
        for i in range(1, len(delta_hdp)):
            prev_state = signals.iloc[i-1]
            x = delta_hdp.iloc[i]
            if prev_state == 0:
                if x >= threshold:
                    signals.iloc[i] = 1
                elif x <= -threshold:
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0
            elif prev_state == 1:
                # Stay in +1 until cross below -threshold
                signals.iloc[i] = -1 if x <= -threshold else 1
            else:  # prev_state == -1
                # Stay in -1 until cross above +threshold
                signals.iloc[i] = 1 if x >= threshold else -1

        # Generate target weights based on signals
        target_weights = pd.DataFrame(index=self.backtest_prices.index, columns=self.backtest_prices.columns, dtype=float)
        target_weights.iloc[0] = [default_weights, 1-default_weights]
        for i in range(1, len(signals)):
            state = signals.iloc[i]
            if state == 1:
                target_weights.iloc[i] = [down_weights, 1-down_weights]
            elif state == -1:
                target_weights.iloc[i] = [up_weights, 1-up_weights]
            else:
                target_weights.iloc[i] = target_weights.iloc[i-1]
        
        return target_weights
    
    def _get_param_combinations(self):
        keys = tuple(self._params.keys())
        return [dict(zip(keys, values)) for values in product(*(self._params[k] for k in keys))]