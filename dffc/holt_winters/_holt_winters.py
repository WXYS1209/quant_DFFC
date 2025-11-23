from numba import njit
import numpy as np
from vectorbt import _typing as tp
from vectorbt.indicators.factory import IndicatorFactory

@njit(cache=True)
def hw_nb(a: tp.Array2d, 
          alpha: float,
          beta: float,
          gamma: float,
          m: int,
          multiplicative: bool = True
          ) -> tp.Array2d:
    """Compute additive or multiplicative holt-winters smoothing values."""
    return holt_winters_ets_nb(a, alpha, beta, gamma, m, multiplicative)

@njit(cache=True)
def hw_cache_nb(close: tp.Array2d,
                alphas: tp.List[float],
                betas: tp.List[float],
                gammas: tp.List[float],
                ms: tp.List[int],
                multiplicatives: tp.List[bool]
                ) -> tp.Dict[int, tp.Array2d]:
    """Caching function for Holt-Winters indicators."""
    cache_dict = dict()
    for i in range(len(alphas)):
        h = hash((alphas[i], betas[i], gammas[i], ms[i], multiplicatives[i]))
        if h not in cache_dict:
            cache_dict[h] = hw_nb(close, alphas[i], betas[i], gammas[i], ms[i], multiplicatives[i])
    return cache_dict

@njit(cache=True)
def hw_apply_nb(close: tp.Array2d, alpha: float, beta: float, gamma: float, m: int, multiplicative: bool,
                cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Array2d:
    """Apply function for Holt-Winters indicators."""
    h = hash((alpha, beta, gamma, m, multiplicative))
    return cache_dict[h]

@njit(cache=True)
def holt_winters_ets_1d_nb(a: tp.Array1d,
                           alpha: float,
                           beta: float,
                           gamma: float,
                           m: int,
                           multiplicative: bool = True) -> tp.Array1d:
    """
    ETS(A, A, A/M) Holt–Winters (triple exponential smoothing), one-step-ahead fitted values.

    Parameters
    ----------
    a : 1d array
        输入时间序列（float）。要求已清洗为数值且无 NaN（Numba 下不处理缺失）。
    alpha, beta, gamma : float in (0, 1)
        水平/趋势/季节 平滑参数。
    m : int
        季节长度（如 5/20/252）。
    multiplicative : bool
        是否使用乘法季节模型。False为加法模型(A,A,A)，True为乘法模型(A,A,M)。

    Returns
    -------
    fitted : 1d array
        一步先验拟合值（长度与 a 相同）。
    """
    n = len(a)
    
    # Parameter validation
    if n < 2 * m:
        raise ValueError("Data length must be at least 2 * seasonal_periods")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if m < 1:
        raise ValueError("m must be >= 1")
    
    # 乘法模型需要正值
    if multiplicative and np.any(a <= 0):
        raise ValueError("Multiplicative model requires all values to be positive")
    
    # Initialize output array
    fitted = np.empty_like(a, dtype=np.float64)
    
    # Initialize level, trend, and seasonal components
    level = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.float64)
    seasonal = np.zeros(n, dtype=np.float64)
    
    # Initial values - improved method
    # Level: average of first m observations
    l0 = np.mean(a[:m])
    
    # Trend: average of differences between first and second seasons
    t0 = (np.mean(a[m:2*m]) - np.mean(a[:m])) / m
    
    # Seasonal: improved initial seasonal components
    s0 = np.zeros(m, dtype=np.float64)
    
    if multiplicative:
        # 乘法模型：计算季节性因子（相对于去趋势水平的比值）
        for i in range(m):
            seasonal_vals = []
            for j in range(i, min(n, 2*m), m):
                detrended_level = l0 + t0 * j
                if detrended_level != 0:
                    seasonal_factor = a[j] / detrended_level
                    seasonal_vals.append(seasonal_factor)
            
            if len(seasonal_vals) > 0:
                s0[i] = np.mean(np.array(seasonal_vals))
            else:
                s0[i] = 1.0
        
        # 标准化季节因子使其平均值为1（乘法模型约束）
        s0_mean = np.mean(s0)
        if s0_mean != 0:
            for i in range(m):
                s0[i] = s0[i] / s0_mean
        else:
            # 如果平均值为0，设置为1（无季节性）
            for i in range(m):
                s0[i] = 1.0
    else:
        # 加法模型：原有逻辑
        for i in range(m):
            seasonal_vals = []
            for j in range(i, min(n, 2*m), m):
                detrended = a[j] - (l0 + t0 * j)
                seasonal_vals.append(detrended)
            
            if len(seasonal_vals) > 0:
                s0[i] = np.mean(np.array(seasonal_vals))
            else:
                s0[i] = 0.0

        # 标准化季节分量使其和为0（加法模型约束）
        s0_mean = np.mean(s0)
        for i in range(m):
            s0[i] -= s0_mean
    
    # Recursive calculation
    for t in range(n):
        # Get previous values
        if t == 0:
            l_prev = l0
            t_prev = t0
        else:
            l_prev = level[t-1]
            t_prev = trend[t-1]

        # Get seasonal component
        if t < m:
            s_tm = s0[t]
        else:
            s_tm = seasonal[t-m]
            
        # One-step-ahead forecast for current period
        if multiplicative:
            fitted[t] = (l_prev + t_prev) * s_tm
        else:
            fitted[t] = l_prev + t_prev + s_tm
        
        # Update level
        if multiplicative:
            if s_tm != 0:
                level[t] = alpha * (a[t] / s_tm) + (1 - alpha) * (l_prev + t_prev)
            else:
                level[t] = l_prev + t_prev  # 避免除零
        else:
            level[t] = alpha * (a[t] - s_tm) + (1 - alpha) * (l_prev + t_prev)

        # Update trend  
        trend[t] = beta * (level[t] - l_prev) + (1 - beta) * t_prev
        
        # Update seasonal component
        if multiplicative:
            if l_prev != 0:
                seasonal[t] = gamma * (a[t] / level[t]) + (1 - gamma) * s_tm
                # seasonal[t] = gamma * (a[t] / (l_prev + t_prev)) + (1 - gamma) * s_tm
            else:
                seasonal[t] = s_tm  # 避免除零
        else:
            seasonal[t] = gamma * (a[t] - level[t]) + (1 - gamma) * s_tm
            # seasonal[t] = gamma * (a[t] - l_prev - t_prev) + (1 - gamma) * s_tm
    
    return fitted


@njit(cache=True)
def holt_winters_ets_nb(a: tp.Array2d,
               alpha: float,
               beta: float,
               gamma: float,
               m: int,
               multiplicative: bool = True) -> tp.Array2d:
    """
    2-dim version of `holt_winters_1d_nb`.
    
    Applies Holt-Winters additive or multiplicative model to each column of the input array.
    """
    out = np.empty_like(a, dtype=np.float64)
    for col in range(a.shape[1]):
        out[:, col] = holt_winters_ets_1d_nb(a[:, col], alpha, beta, gamma, m, multiplicative)
    return out

@njit(cache=True)
def hw_delta_nb(close: tp.Array2d, hw: tp.Array2d) -> tp.Array2d:
    """Calculate Holt-Winters Delta: difference between original price and HW fitted values."""
    return close - hw


@njit(cache=True)
def hw_delta_apply_nb(close: tp.Array2d, alpha: float, beta: float, gamma: float, m: int, multiplicative: bool,
                      cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Array2d:
    """Apply function for Holt-Winters Delta indicators."""
    hw = hw_apply_nb(close, alpha, beta, gamma, m, multiplicative, cache_dict)
    return hw_delta_nb(close, hw)


@njit(cache=True)  
def hw_delta_percentage_1d_nb(hwd: tp.Array1d) -> tp.Array1d:
    """
    Calculate Holt-Winters Delta Percentage: normalized position of HWD.
    
    Parameters
    ----------
    hwd : 1d array
        Holt-Winters Delta values
        
    Returns
    -------
    hwdp : 1d array
        HWD percentage normalized to [-1, 1] range
    """
    n = len(hwd)
    hwdp = np.empty_like(hwd, dtype=np.float64)
    
    for i in range(n):
        if i < 3:
            # Not enough data for full window
            hwdp[i] = 0.0
        else:
            # Get window data
            len_i = (i+1) // 2
            start_idx = i - len_i + 1
            window_data = hwd[start_idx:i+1]
            
            # Calculate min and max in window
            min_val = np.min(window_data)
            max_val = np.max(window_data)
            
            # Normalize to [-1, 1]
            if max_val == min_val:
                # No variation in window
                hwdp[i] = 0.0
            else:
                # Current value position in window, normalized to [-1, 1]
                count = 0
                for x in window_data:
                    if x < hwd[i]:
                        count += 1
                hwdp[i] = (float(count) / float(len(window_data))) * 2 - 1
                
    return hwdp


@njit(cache=True)
def hw_delta_percentage_nb(hwd: tp.Array2d) -> tp.Array2d:
    """2-dim version of hw_delta_percentage_1d_nb."""
    out = np.empty_like(hwd, dtype=np.float64)
    for col in range(hwd.shape[1]):
        out[:, col] = hw_delta_percentage_1d_nb(hwd[:, col])
    return out


@njit(cache=True)
def hw_delta_percentage_cache_nb(close: tp.Array2d,
                                alphas: tp.List[float],
                                betas: tp.List[float], 
                                gammas: tp.List[float],
                                ms: tp.List[int],
                                multiplicatives: tp.List[bool]
                                ) -> tp.Dict[int, tp.Array2d]:
    """Caching function for Holt-Winters Delta Percentage indicators."""
    cache_dict = dict()
    hw_cache = hw_cache_nb(close, alphas, betas, gammas, ms, multiplicatives)
    
    for i in range(len(alphas)):
        hw_h = hash((alphas[i], betas[i], gammas[i], ms[i], multiplicatives[i]))
        hw_val = hw_cache[hw_h]
        hwd = hw_delta_nb(close, hw_val)
        
        h = hash((alphas[i], betas[i], gammas[i], ms[i], multiplicatives[i]))
        if h not in cache_dict:
            cache_dict[h] = hw_delta_percentage_nb(hwd)
    return cache_dict


@njit(cache=True)
def hw_delta_percentage_apply_nb(close: tp.Array2d, alpha: float, beta: float, gamma: float, m: int, 
                                multiplicative: bool,
                                cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Array2d:
    """Apply function for Holt-Winters Delta Percentage indicators."""
    h = hash((alpha, beta, gamma, m, multiplicative))
    return cache_dict[h]


HW = IndicatorFactory(
    class_name='HW',
    module_name=__name__,
    short_name='hw',
    input_names=['close'],
    param_names=['alpha', 'beta', 'gamma', 'm', 'multiplicative'],
    output_names=['hw']
).from_apply_func(
    hw_apply_nb,
    cache_func=hw_cache_nb,
    param_product=True,
    # kwargs_to_args=['adjust'],
    multiplicative=True
)

HWD = IndicatorFactory(
    class_name='HWD',
    module_name=__name__,
    short_name='hwd',
    input_names=['close'],
    param_names=['alpha', 'beta', 'gamma', 'm', 'multiplicative'],
    output_names=['hwd']
).from_apply_func(
    hw_delta_apply_nb,
    cache_func=hw_cache_nb,
    multiplicative=True
)

HWDP = IndicatorFactory(
    class_name='HWDP',
    module_name=__name__,
    short_name='hwdp',
    input_names=['close'],
    param_names=['alpha', 'beta', 'gamma', 'm', 'multiplicative'],
    output_names=['hwdp']
).from_apply_func(
    hw_delta_percentage_apply_nb,
    cache_func=hw_delta_percentage_cache_nb,
    multiplicative=True
)