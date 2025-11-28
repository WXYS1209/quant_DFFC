from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as opt
from plotly.subplots import make_subplots
from tqdm import tqdm

from dffc.holt_winters._holt_winters import HW

# ============================================================================
# Constants
# ============================================================================
MOVING_AVERAGE_WINDOW = 30
"""滑动平均窗口大小，用于计算趋势基准线"""

END_DAY_DEFAULTS = list(range(-400, 0, 40)) + [None]
"""默认的 end_day 列表，用于回溯拟合参数稳定性分析"""


# ============================================================================
# Utility Functions
# ============================================================================

def sliding_average(
    arr: pd.Series | np.ndarray,
    window: int,
) -> pd.Series | np.ndarray:
    """计算中心滑动平均，边界使用较小窗口，输出长度与输入相同。"""
    if isinstance(arr, pd.Series):
        # 对于pandas Series，保持索引
        n = len(arr)
        half = window // 2
        result = pd.Series(index=arr.index, dtype=float)
        
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            result.iloc[i] = arr.iloc[start:end].mean()
        return result
    else:
        # 对于numpy数组，保持原有逻辑
        arr = np.asarray(arr, dtype=float)
        n = arr.shape[0]
        half = window // 2
        result = np.empty_like(arr, dtype=float)
        
        if arr.ndim == 1:
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                result[i] = arr[start:end].mean()
        else:
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                result[i] = arr[start:end].mean(axis=0)
        return result

def holtwinters_rolling(arr, alpha, beta, gamma, season_length, multiplicative=False):
    """执行 Holt-Winters 三参数滚动平滑。"""
    hw = HW.run(arr, alpha, beta, gamma, season_length, multiplicative=multiplicative)
    return hw.hw

def calc_scaling_factor(fluc_A, fluc_B):
    """最小二乘法计算最优缩放因子 a，使 a * fluc_B ≈ fluc_A。"""
    denom = np.dot(fluc_B, fluc_B)
    if denom == 0:
        return 0.0
    return np.dot(fluc_A, fluc_B) / denom

def calc_MSE(fluc_A, fluc_B, scaling_factor):
    """计算缩放后的均方误差 MSE(fluc_A, scaling_factor * fluc_B)。"""
    errors = fluc_A - scaling_factor * fluc_B
    mse = np.mean(errors ** 2)
    return mse

def optimize_single_season(season, original_data, fluc_data, holtwinters_begindate, holtwinters_enddate, options, bounds):
    """单个季节长度的参数优化（用于并行计算）。"""
    # 基于季节长度设置初始猜测
    if season <= 7:
        initial_guess = [0.3, 0.01, 0.1]
    elif season <= 15:
        initial_guess = [0.2, 0.05, 0.15]
    else:
        initial_guess = [0.1, 0.1, 0.2]

    def local_objective(params):
        alpha, beta, gamma = params
        try:
            smoothed = holtwinters_rolling(original_data, alpha, beta, gamma, season_length=season)
            holtwinters_fluc = original_data - smoothed
            fluc_sub = fluc_data[holtwinters_begindate:holtwinters_enddate]
            holtwinters_fluc_sub = holtwinters_fluc[holtwinters_begindate:holtwinters_enddate]
            a = calc_scaling_factor(fluc_sub, holtwinters_fluc_sub)
            mse = calc_MSE(fluc_sub, holtwinters_fluc_sub, a)
            return mse
        except Exception:
            return 1e10  # 计算失败返回大值

    res = opt.minimize(local_objective,
                       initial_guess,
                       bounds=bounds,
                       method='L-BFGS-B',
                       options=options)
    
    return {
        'season_length': season,
        'success': res.success,
        'fun': res.fun,
        'x': res.x
    }

def optimize_holtwinters_parameters(original_data,
                                    holtwinters_begindate,
                                    holtwinters_enddate,
                                    max_workers=8):
    """并行优化 HW 参数，返回 (最优参数, 最优季节长度, MSE)。"""
    mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
    fluc_data = original_data - mean_data

    options = {
        'ftol': 1e-9,
        'gtol': 1e-6,
        'maxiter': 10000,
        'maxfun': 10000,
        'disp': False,
    }

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    seasons = list(range(7, 50))

    with ProcessPoolExecutor(max_workers=min(len(seasons), max_workers)) as executor:
        futures = {
            executor.submit(
                optimize_single_season,
                season,
                original_data,
                fluc_data,
                holtwinters_begindate,
                holtwinters_enddate,
                options,
                bounds,
            ): season
            for season in seasons
        }

        results = []
        for future in as_completed(futures):
            result = future.result()
            if result['success']:
                results.append(result)
    
    if not results:
        return _optimize_holtwinters_parameters_serial(original_data, holtwinters_begindate, holtwinters_enddate)

    
    best_result = min(results, key=lambda x: x['fun'])
    return best_result['x'], best_result['season_length'], best_result['fun']

def _optimize_holtwinters_parameters_serial(original_data, holtwinters_begindate, holtwinters_enddate):
    """串行版本的参数优化（并行失败时的备选）。"""
    mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
    fluc_data = original_data - mean_data

    best_mse = np.inf
    best_params = None
    best_season = None

    options = {
        'ftol': 1e-9,
        'gtol': 1e-6,
        'maxiter': 10000,
        'maxfun': 10000,
        'disp': False,
    }

    for season in range(7, 50):
        if season <= 7:
            initial_guess = [0.3, 0.05, 0.1]
        elif season <= 15:
            initial_guess = [0.2, 0.1, 0.1]
        else:
            initial_guess = [0.1, 0.15, 0.1]
            
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        def local_objective(params, _season=season):
            alpha, beta, gamma = params
            smoothed = holtwinters_rolling(original_data, alpha, beta, gamma, season_length=_season)
            holtwinters_fluc = original_data - smoothed
            fluc_sub = fluc_data[holtwinters_begindate:holtwinters_enddate]
            holtwinters_fluc_sub = holtwinters_fluc[holtwinters_begindate:holtwinters_enddate]
            a = calc_scaling_factor(fluc_sub, holtwinters_fluc_sub)
            mse = calc_MSE(fluc_sub, holtwinters_fluc_sub, a)
            return mse

        res = opt.minimize(local_objective,
                           initial_guess,
                           bounds=bounds,
                           method='L-BFGS-B',
                           options=options)
        if res.fun < best_mse:
            best_mse = res.fun
            best_params = res.x
            best_season = season

    return best_params, best_season, best_mse

def compute_optimize_result(end_day, original_data, max_workers=8):
    """计算单个 end_day 的优化结果。"""
    best_params, best_season, best_mse = optimize_holtwinters_parameters(original_data, 0, end_day, max_workers)
    return {
        "end_day": end_day,
        "alpha": best_params[0],
        "beta": best_params[1],
        "gamma": best_params[2],
        "season_length": best_season,
        "mse": best_mse
    }


# ============================================================================
# HWOptimizer Class
# ============================================================================

class HWOptimizer:
    """HW 参数优化器，支持批量优化、参数分析和更新。"""
    
    def __init__(self, original_data: pd.DataFrame):
        """初始化优化器。
        
        Args:
            original_data: 包含多个资产价格数据的 DataFrame
        """
        if not isinstance(original_data, pd.DataFrame):
            raise ValueError("original_data must be a pandas DataFrame")
        
        self.original_data = original_data
        self.assets = {}  # {code: HWAssetResult}
        
    def optimize(self,
                 save=True,
                 output_base_dir="./hw_opt_results",
                 result_filename="hw_opt_results",
                 max_workers=8,
                 end_day_list: Optional[Sequence[Optional[int]]] = None,
                 progress_cb: Optional[Callable[[int, int, str, Optional[int]], None]] = None):
        """执行批量优化。"""
        code_list = self.original_data.columns.tolist()
        
        if save:
            os.makedirs(output_base_dir, exist_ok=True)
        
        if end_day_list is None:
            end_day_values = END_DAY_DEFAULTS
        else:
            end_day_values = list(end_day_list)
            if not end_day_values:
                end_day_values = END_DAY_DEFAULTS

        total_steps = max(1, len(code_list) * len(end_day_values))
        completed_steps = 0

        for code in code_list:
            try:
                value_data = self.original_data[code].dropna()
                results = []
                end_day_iter = (tqdm(end_day_values, desc=f"Optimizing for {code}", leave=False)
                                if progress_cb is None else end_day_values)
                for end_day in end_day_iter:
                    result = compute_optimize_result(end_day, value_data, max_workers)
                    results.append(result)
                    completed_steps += 1
                    if progress_cb is not None:
                        progress_cb(completed_steps, total_steps, code, end_day)

                results_df = pd.DataFrame(results)
                last_result = results_df.iloc[-1]
                
                asset_result = HWAssetResult(
                    code=code,
                    value_data=value_data,
                    results_df=results_df,
                    params={
                        'alpha': last_result['alpha'],
                        'beta': last_result['beta'],
                        'gamma': last_result['gamma'],
                        'season_length': last_result['season_length'],
                        'mse': last_result['mse']
                    }
                )
                self.assets[code] = asset_result

                if progress_cb is None:
                    print(f" {code} processed! Alpha={last_result['alpha']:.4f}, Beta={last_result['beta']:.4f}, Gamma={last_result['gamma']:.4f}, Season={last_result['season_length']}")
                
            except Exception as e:
                print(f" {code} processing failed: {str(e)}")
                self.assets[code] = HWAssetResult(code=code, error=str(e))
        
        if save:
            self.save_summary(output_base_dir, result_filename)
        
        return self.get_summary()
    
    def update_params(self, code: str, params: dict):
        """更新指定资产的推荐参数。"""
        if code not in self.assets:
            raise ValueError(f"Asset {code} not found")
        self.assets[code].update_params(params)
    
    def get_asset(self, code: str):
        """获取指定资产的优化结果。"""
        return self.assets.get(code)
    
    def get_summary(self):
        """获取所有资产的汇总信息。"""
        return [asset.to_dict() for asset in self.assets.values()]
    
    def save_summary(self, output_base_dir: str, filename: str):
        """保存汇总结果到 JSON 文件，同时保存每个资产的单独文件。"""
        summary = []
        for asset in self.assets.values():
            # 保存单个资产文件
            if asset.status == 'success':
                asset.save(output_base_dir)
            
            # 添加到汇总
            item = asset.to_dict()
            if 'results_df' in item and isinstance(item['results_df'], pd.DataFrame):
                item['results_df'] = item['results_df'].to_dict(orient='records')
            summary.append(item)
        
        # 保存汇总文件
        with open(os.path.join(output_base_dir, f"{filename}.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_summary(cls, original_data: pd.DataFrame, summary: list):
        """从汇总数据恢复优化器状态。"""
        optimizer = cls(original_data)
        for item in summary:
            code = item['code']
            value_data = original_data[code].dropna() if code in original_data.columns else None
            
            if item['status'] == 'success':
                results_df = pd.DataFrame(item['results_df']) if isinstance(item['results_df'], list) else item['results_df']
                asset_result = HWAssetResult(
                    code=code,
                    value_data=value_data,
                    results_df=results_df,
                    params=item['params']
                )
            else:
                asset_result = HWAssetResult(code=code, error=item.get('error'))
            
            optimizer.assets[code] = asset_result
        
        return optimizer


class HWAssetResult:
    """单个资产的 HW 优化结果。"""
    
    def __init__(self, code: str, value_data: Optional[pd.Series] = None, 
                 results_df: Optional[pd.DataFrame] = None, params: Optional[dict] = None,
                 error: Optional[str] = None):
        self.code = code
        self.value_data = value_data
        self.results_df = results_df
        self.params = params or {}
        self.error = error
        self.status = 'failed' if error else 'success'
    
    def update_params(self, params: dict):
        """更新推荐参数。"""
        self.params.update(params)
    
    def get_params(self):
        """获取当前推荐参数。"""
        return self.params.copy()
    
    def analyze_stability(self, recent_threshold: int = -100):
        """分析参数稳定性，返回推荐配置。"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        df = self.results_df.copy()
        
        # 最近时间段的数据
        recent_df = df[df['end_day'] >= recent_threshold]
        
        if recent_df.empty:
            return None
        
        # 最常出现的 season_length
        season_mode = recent_df['season_length'].mode().values[0]
        
        # 在该 season_length 下的最近数据
        stable_recent = df[
            (df['season_length'] == season_mode) & 
            (df['end_day'] >= recent_threshold)
        ]
        
        if stable_recent.empty:
            return None
        
        # 使用最新值作为推荐
        latest = stable_recent.iloc[-1]
        
        return {
            'alpha': float(latest['alpha']),
            'beta': float(latest['beta']),
            'gamma': float(latest['gamma']),
            'season_length': int(latest['season_length']),
            'mse': float(latest['mse']),
            'analysis': {
                'recent_threshold': recent_threshold,
                'season_mode': int(season_mode),
                'stable_count': len(stable_recent),
                'alpha_mean': float(stable_recent['alpha'].mean()),
                'beta_mean': float(stable_recent['beta'].mean()),
                'gamma_mean': float(stable_recent['gamma'].mean()),
            }
        }
    
    def to_dict(self):
        """转换为字典格式。"""
        result = {
            'code': self.code,
            'status': self.status,
        }
        
        if self.error:
            result['error'] = self.error
        else:
            result['data_points'] = len(self.value_data) if self.value_data is not None else 0
            result['params'] = self.params
            result['results_df'] = self.results_df
        
        return result
    
    def save(self, output_base_dir: str):
        """保存到 JSON 文件。"""
        output_dir = os.path.join(output_base_dir, str(self.code))
        os.makedirs(output_dir, exist_ok=True)
        
        save_data = self.to_dict()
        if 'results_df' in save_data and isinstance(save_data['results_df'], pd.DataFrame):
            save_data['results_df'] = save_data['results_df'].to_dict(orient='records')
        
        filepath = os.path.join(output_dir, f"hw_opt_results_{MOVING_AVERAGE_WINDOW}_{self.code}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    def compute_hw_curves(self) -> list[dict]:
        """计算每个 end_day 对应的 HW 平滑曲线。"""
        if self.value_data is None or self.results_df is None:
            raise ValueError("Missing value_data or results_df")
        
        curves = []
        for _, row in self.results_df.iterrows():
            alpha, beta, gamma = row['alpha'], row['beta'], row['gamma']
            season_length = int(row['season_length'])
            hw_smooth = holtwinters_rolling(
                self.value_data.values, alpha, beta, gamma, season_length
            )
            curves.append({
                'end_day': row['end_day'],
                'curve': hw_smooth,
            })
        return curves
    
    def plot(self,
             save: bool = False,
             output_dir: Optional[str] = None,
             filename_prefix: Optional[str] = None,
             display: bool = True) -> go.Figure:
        """绘制 HW 优化结果的交互式图表。"""
        if self.value_data is None or self.results_df is None:
            raise ValueError("Missing value_data or results_df")
        
        mean_data = sliding_average(self.value_data, MOVING_AVERAGE_WINDOW)
        hw_curves = self.compute_hw_curves()

        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
            row_heights=[0.65, 0.35],
            vertical_spacing=0.08,
            subplot_titles=("", ""),
        )

        # Row 1: 原始数据与滑动平均
        fig.add_trace(
            go.Scatter(
                x=self.value_data.index,
                y=self.value_data.values,
                name="Original",
                mode="lines",
                line=dict(color="#1f77b4"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=mean_data.index,
                y=mean_data.values,
                name="Moving Avg",
                mode="lines",
                line=dict(color="#ff7f0e", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Row 1: HW 曲线（默认显示最后一条）
        hw_trace_start_idx = len(fig.data)
        for i, curve in enumerate(hw_curves):
            end_day = curve['end_day']
            label = "all" if end_day is None else str(end_day)
            is_last = (i == len(hw_curves) - 1)
            fig.add_trace(
                go.Scatter(
                    x=self.value_data.index,
                    y=curve['curve'],
                    name=f"HW (end_day={label})",
                    mode="lines",
                    line=dict(width=2, color="#2ca02c"),
                    visible=is_last,
                    showlegend=is_last,
                ),
                row=1,
                col=1,
            )
        hw_trace_end_idx = len(fig.data)
        num_hw_traces = hw_trace_end_idx - hw_trace_start_idx

        # Row 2: 参数变化曲线
        end_day_display = self.results_df['end_day'].fillna(0)
        metrics_primary = ["alpha", "beta", "gamma"]
        for metric in metrics_primary:
            fig.add_trace(
                go.Scatter(
                    x=end_day_display,
                    y=self.results_df[metric],
                    name=metric,
                    mode="lines+markers",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(
                x=end_day_display,
                y=self.results_df["season_length"],
                name="season_length",
                mode="lines+markers",
                line=dict(dash="dot"),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        # 构建下拉菜单
        total_traces = len(fig.data)
        buttons = []
        for i, curve in enumerate(hw_curves):
            end_day = curve['end_day']
            label = "all" if end_day is None else str(end_day)
            visibility = [True, True]
            for j in range(num_hw_traces):
                visibility.append(j == i)
            visibility.extend([True] * (total_traces - hw_trace_end_idx))
            
            buttons.append(dict(
                label=f"end_day={label}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{self.code} - HW Optimization (end_day={label})"}
                ]
            ))

        fig.update_layout(
            updatemenus=[
                dict(
                    active=len(hw_curves) - 1,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="#ccc",
                )
            ],
            title=dict(
                text=f"{self.code} - HW Optimization",
                x=0.5,
            ),
            hovermode="x unified",
            height=900,
        )

        fig.update_xaxes(title="Date", row=1, col=1)
        fig.update_yaxes(title="Value", row=1, col=1)
        fig.update_xaxes(title="end_day", row=2, col=1)
        fig.update_yaxes(title="alpha / beta / gamma", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title="season_length", row=2, col=1, secondary_y=True)

        if save:
            if output_dir is None:
                raise ValueError("output_dir is required when save=True")
            os.makedirs(output_dir, exist_ok=True)
            prefix = filename_prefix or f"hw_plot_{MOVING_AVERAGE_WINDOW}_{self.code}"
            html_path = os.path.join(output_dir, f"{prefix}.html")
            fig.write_html(html_path)

        if display:
            fig.show()

        return fig


# ============================================================================
# Backward Compatibility
# ============================================================================

def process_hw_opt(original_data,
                   save=True,
                   output_base_dir="./hw_opt_results",
                   result_filename="hw_opt_results",
                   max_workers=8,
                   end_day_list: Optional[Sequence[Optional[int]]] = None,
                   progress_cb: Optional[Callable[[int, int, str, Optional[int]], None]] = None):
    """批量处理多个资产的 HW 参数优化（兼容旧接口）。"""
    optimizer = HWOptimizer(original_data)
    return optimizer.optimize(
        save=save,
        output_base_dir=output_base_dir,
        result_filename=result_filename,
        max_workers=max_workers,
        end_day_list=end_day_list,
        progress_cb=progress_cb
    )


def compute_hw_curves(
    value_data: pd.Series,
    results_df: pd.DataFrame,
) -> list[dict]:
    """根据优化结果计算每个 end_day 对应的 HW 平滑曲线（兼容旧接口）。"""
    curves = []
    for _, row in results_df.iterrows():
        alpha, beta, gamma = row['alpha'], row['beta'], row['gamma']
        season_length = int(row['season_length'])
        hw_smooth = holtwinters_rolling(
            value_data.values, alpha, beta, gamma, season_length
        )
        curves.append({
            'end_day': row['end_day'],
            'curve': hw_smooth,
        })
    return curves


def plot_hw_opt(
    code: str,
    value_data: pd.Series,
    results_df: pd.DataFrame,
    save: bool = False,
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    display: bool = True,
) -> go.Figure:
    """绘制 HW 优化结果的交互式图表（兼容旧接口）。"""
    asset = HWAssetResult(code=code, value_data=value_data, results_df=results_df)
    return asset.plot(save=save, output_dir=output_dir, filename_prefix=filename_prefix, display=display)

