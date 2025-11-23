"""Backtest orchestration helpers.

This module houses light-weight wrappers around strategy instances so that
callers can run a backtest, inspect statistics, and render plots without
rewriting boilerplate code.  The public surface is intentionally small: supply
an already-configured strategy, optionally pass default keyword arguments for
`run_backtest`, and use :class:`BackTest` or its subclasses to analyse and
visualise the results.

Notes
-----
BackTest never mutates the provided strategy beyond what
``strategy.run_backtest`` already performs internally.  The same strategy
instance can therefore be safely reused for multiple runs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as plotly_colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from abc import ABC, abstractmethod

class BackTest(ABC):
    """Thin orchestrator around a Strategy instance.

    Concrete subclasses are expected to encapsulate strategy-specific plotting
    or post-processing but they all adhere to the same three-step workflow:

    1. :meth:`run` to execute the underlying strategy and cache the resulting
       vectorbt portfolio.
    2. :meth:`stats` to expose the most important statistics in a friendly
       shape.
    3. :meth:`plot` to visualise the outcome (implemented by subclasses).

    The base class itself is intentionally state-less beyond storing the most
    recent run result in ``self.pf`` and a convenience flag ``self._multi``
    indicating whether the run produced multiple parameter groups.
    """

    def __init__(self, strategy: Any, **run_kwargs: Any) -> None:
        self.strategy = strategy
        # Default kwargs for strategy.run_backtest; can be overridden per call in run()
        self._run_defaults = dict(run_kwargs)
        self._multi = None

    def run(self, **kwargs: Any) -> SimpleNamespace:
        """Execute ``strategy.run_backtest`` with merged keyword arguments.

        Parameters
        ----------
        **kwargs
            Per-call overrides forwarded directly to
            :meth:`strategy.run_backtest`.  Any defaults supplied to the
            constructor are merged with these values (per-call arguments win).

        Returns
        -------
        SimpleNamespace
            Whatever object the concrete strategy returns.  For convenience the
            resulting portfolio is cached on ``self.pf`` for subsequent use.
        """
        # Merge init defaults with call-time kwargs (call-time takes precedence)
        merged = {**getattr(self, '_run_defaults', {}), **kwargs}
        self.pf = self.strategy.run_backtest(**merged)

        pf_size = self.pf.wrapper.shape[1]
        self._multi = (pf_size != self.strategy.backtest_prices.shape[1])
    
    def stats(self,
              selected: Any = None) -> Any:
        """Return summary statistics for the most recent run.

        Parameters
        ----------
        selected
            Optional selector used when the run produced multiple parameter
            groups.  When provided the method returns the statistics for that
            specific column (whatever format ``vectorbt`` exposes).

        Returns
        -------
        Any
            Either a ``pandas.DataFrame`` for multi-parameter runs or the raw
            ``vectorbt`` stats object for single runs.

        Raises
        ------
        ValueError
            If ``selected`` does not correspond to a valid portfolio column.
        """
        if not hasattr(self, 'pf'):
            raise RuntimeError('BackTest.run() must be called before stats().')

        pf = self.pf

        if not self._multi:
            if selected is not None:
                raise ValueError('Single-parameter backtests do not support the "selected" argument.')
            return pf.stats()

        if self._multi:
            if selected is not None:
                try:
                    return pf[selected].stats()
                except KeyError:
                    available = getattr(pf.wrapper, 'columns', [])
                    sample_cols = ', '.join(map(str, available[:5]))
                    extra_hint = f'. Available columns include: {sample_cols}' if sample_cols else ''
                    raise ValueError(f"Unknown portfolio key: {selected}{extra_hint}") from None
            bch_tr = pf.total_benchmark_return()
            strat_tr = pf.total_return()
            strat_sr = pf.sharpe_ratio()
            strat_mdd = pf.max_drawdown()

            metadata = getattr(self.strategy, 'param_metadata', None)
            if metadata is None or metadata.empty:
                return pf.stats()

            stats_df = metadata.copy()
            stats_df = stats_df.reset_index()
            index_cols = list(stats_df.columns)
            float_index_cols = [col for col in index_cols if pd.api.types.is_numeric_dtype(stats_df[col])]
            if float_index_cols:
                def _round_numeric_columns(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                    for col in cols:
                        frame[col] = frame[col].round(8)
                    return frame

                stats_df = _round_numeric_columns(stats_df, float_index_cols)

            stats_df['benchmark_return'] = bch_tr
            stats_df['total_return'] = strat_tr
            stats_df['sharpe_ratio'] = strat_sr
            stats_df['max_drawdown'] = strat_mdd

            stats_df = stats_df.set_index(index_cols).sort_index()

            new_levels = []
            for level in stats_df.index.levels:
                if pd.api.types.is_float_dtype(level.dtype):
                    new_levels.append(pd.Index(np.round(level.to_numpy(), 8), name=level.name))
                else:
                    new_levels.append(level)

            stats_df.index = stats_df.index.set_levels(new_levels)

            return stats_df
    
    @abstractmethod
    def plot(self, **kwargs):
        pass


class ReallocationBackTest(BackTest):

    def get_best_param(self, metric: str = 'total_return', top_n: int = 1) -> pd.DataFrame:
        """Convenience helper to fetch the top parameter sets.

        Parameters
        ----------
        metric
            Column name from the multi-run statistics table used for ranking.
        top_n
            How many parameter combinations to return (sorted descending).

        Returns
        -------
        pandas.DataFrame
            ``top_n`` parameter combinations for multi-run backtests, including
            the ranking metric as an additional column.  For single-run
            backtests a one-row dataframe describing the executed parameter set.
        """
        if self._multi:
            ranked = self.stats()[metric].sort_values(ascending=False).head(top_n)
            param_df = ranked.index.to_frame(index=False)
            param_df[metric] = ranked.to_numpy()
            return param_df.reset_index(drop=True)

        metadata = getattr(self.strategy, 'param_metadata', None)
        if metadata is not None and not metadata.empty:
            return metadata.reset_index().iloc[[0]].reset_index(drop=True)

        param_map = getattr(self.strategy, 'param_group_map', None)
        if isinstance(param_map, dict) and 0 in param_map:
            return pd.DataFrame([param_map[0]])

        raise RuntimeError('Unable to determine parameter set for single-run backtests.')

    def get_weighted_best_params(
        self,
        metric_weight: Mapping[str, float],
        top_n: int = 5,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Rank parameter sets by a weighted combination of statistics.

        Parameters
        ----------
        metric_weight
            Mapping from statistic column name to its weight. Positive values
            favour higher metric values, negative weights invert the preference.
        top_n
            Number of parameter sets to return.
        normalize
            When ``True`` (default) each metric is min-max normalised before
            weighting. Disable to work with raw metric values.

        Returns
        -------
        pandas.DataFrame
            Parameter combinations sorted by the computed weighted score. The
            resulting frame includes the weighted score alongside the raw
            metric values used.
        """
        if not metric_weight:
            raise ValueError("metric_weight must not be empty.")

        stats_df = self.stats()
        if not isinstance(stats_df, pd.DataFrame):
            raise RuntimeError(
                "Weighted parameter selection requires a multi-parameter backtest."
            )

        stats_df = stats_df.copy()
        metric_series = pd.Series(metric_weight, dtype=float)

        missing_metrics = [metric for metric in metric_series.index if metric not in stats_df.columns]
        if missing_metrics:
            raise KeyError(f"Metrics not found in stats table: {missing_metrics}")

        metric_values = (
            stats_df.loc[:, metric_series.index]
            .apply(pd.to_numeric, errors='coerce')
        )

        empty_metrics = [metric for metric in metric_series.index if metric_values[metric].isna().all()]
        if empty_metrics:
            raise ValueError(f"Metrics contain only NaN values: {empty_metrics}")

        if normalize:
            col_max = metric_values.max()
            col_min = metric_values.min()
            denom = (col_max - col_min).replace(0, 1)
            metric_values = (metric_values - col_min) / denom

        total_weight = metric_series.abs().sum()
        if total_weight == 0:
            raise ValueError("At least one metric weight must be non-zero.")

        weights = metric_series / total_weight
        weighted_score = metric_values.mul(weights, axis=1).sum(axis=1)

        ranked = weighted_score.sort_values(ascending=False).head(top_n)
        ranked_params = ranked.index.to_frame(index=False)
        ranked_params['weighted_score'] = ranked.to_numpy()

        for metric in metric_series.index:
            ranked_params[metric] = stats_df.loc[ranked.index, metric].to_numpy()

        return ranked_params.reset_index(drop=True)
    
    def plot(self, **kwargs):
        """Visualise run results via Plotly.

        Subclasses can pass through additional keyword arguments to tweak the
        plotting behaviour.  When the backtest produced a single parameter
        combination the detailed single plot is rendered, otherwise a compact
        heatmap dashboard is shown.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure describing the backtest outcome.
        """
        if not self._multi:
            # 单参数组合单资产
            return self._plot_single(**kwargs)

        index_levels = kwargs.pop('index_levels', None)
        column_levels = kwargs.pop('column_levels', None)

        if index_levels is None or column_levels is None:
            metadata = getattr(self.strategy, 'param_metadata', None)
            if (
                metadata is None
                or metadata.empty
                or not isinstance(metadata.index, pd.MultiIndex)
            ):
                missing = []
                if index_levels is None:
                    missing.append('index_levels')
                if column_levels is None:
                    missing.append('column_levels')
                missing_str = ' and '.join(missing)
                raise ValueError(
                    f'Unable to infer {missing_str} automatically; please provide them explicitly.'
                )

            level_names = [
                name if name is not None else f'level_{i}'
                for i, name in enumerate(metadata.index.names)
            ]
            split_point = max(1, len(level_names) // 2)

            if index_levels is None:
                index_levels = level_names[:split_point]
            if column_levels is None:
                column_levels = level_names[split_point:] or level_names[-1:]

        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise TypeError(f'plot() got unexpected keyword arguments: {unexpected}')

        return self._plot_multi(index_levels, column_levels)
    
    def _plot_single(self, column=None):
        """Render a detailed interactive dashboard for a single parameter set.

        Parameters
        ----------
        column
            Optional column selector passed to ``vectorbt``'s ``plot``.  When
            omitted a custom three-panel view (price & orders, weights, returns)
            is constructed to surface the most important diagnostics.
        """

        if column is not None:
            try:
                fig = self.pf.plot(group_by=False, column=column)
                fig.show()
                return fig
            except Exception as e:
                raise ValueError(f"Unknown portfolio column: {column}. Error: {e}")
        
        num_assets = len(self.strategy.backtest_prices.columns)
        base_colors = plotly_colors.Set1
        color_cycle = (base_colors * ((num_assets // len(base_colors)) + 1))[:num_assets]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Prices & Orders',
                'Weights',
                'Cumulative Returns'
            )
        )

        orders = self.pf.orders.records
        price_index = self.strategy.backtest_prices.index

        # Panel 1: price evolution augmented with buy/sell markers sourced from
        # vectorbt order records.
        for i, col in enumerate(self.strategy.backtest_prices.columns):
            fig.add_trace(
                go.Scatter(
                    x=price_index,
                    y=self.strategy.backtest_prices.iloc[:, i],
                    mode='lines',
                    line=dict(color=color_cycle[i], width=2),
                    name=str(col),
                    legendgroup=f'price-{col}',
                    hovertemplate=f'<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>Price: %{{y:.4f}}<extra></extra>'
                ),
                row=1,
                col=1
            )

        if hasattr(orders, 'side') and getattr(orders, 'idx', None) is not None and len(orders.idx) > 0:
            order_dates = price_index[orders.idx]
            buy_mask = orders.side == 0
            sell_mask = orders.side == 1

            if buy_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=order_dates[buy_mask],
                        y=orders.price[buy_mask],
                        mode='markers',
                        marker=dict(symbol='triangle-up', color='green', size=10, line=dict(width=1, color='darkgreen')),
                        name='Buy',
                        legendgroup='trades',
                        hovertemplate='<b>Buy</b><br>%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.4f}<extra></extra>'
                    ),
                    row=1,
                    col=1
                )
            if sell_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=order_dates[sell_mask],
                        y=orders.price[sell_mask],
                        mode='markers',
                        marker=dict(symbol='triangle-down', color='red', size=10, line=dict(width=1, color='darkred')),
                        name='Sell',
                        legendgroup='trades',
                        hovertemplate='<b>Sell</b><br>%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.4f}<extra></extra>'
                    ),
                    row=1,
                    col=1
                )

    # Panel 2: target allocation realised through time (stacked area chart).
        asset_values = self.pf.asset_value(group_by=False)
        if isinstance(asset_values.columns, pd.MultiIndex):
            asset_values = asset_values.copy()
            asset_values.columns = asset_values.columns.get_level_values(-1)

        total_value = self.pf.value()
        if isinstance(total_value, pd.DataFrame):
            total_value = total_value.iloc[:, 0]
        weights = asset_values.div(total_value, axis=0)

        if isinstance(weights.columns, pd.MultiIndex):
            weights = weights.copy()
            weights.columns = weights.columns.get_level_values(-1)

        weights = weights.reindex(columns=self.strategy.backtest_prices.columns)

        for i, col in enumerate(self.strategy.backtest_prices.columns):
            fig.add_trace(
                go.Scatter(
                    x=weights.index,
                    y=weights.iloc[:, i],
                    mode='lines',
                    line=dict(color=color_cycle[i]),
                    stackgroup='weights',
                    name=f'Weight · {col}',
                    legendgroup=f'weight-{col}',
                    hovertemplate=f'<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>Weight: %{{y:.2%}}<extra></extra>'
                ),
                row=2,
                col=1
            )

    # Panel 3: strategy vs equal-weight benchmark cumulative returns.
        portfolio_value = self.pf.value()
        if isinstance(portfolio_value, pd.DataFrame):
            portfolio_value = portfolio_value.iloc[:, 0]

        portfolio_returns = (portfolio_value / self.pf.init_cash - 1) * 100

        benchmark_returns = self.strategy.backtest_prices.pct_change().mean(axis=1).fillna(0)
        benchmark_cumret = ( (1 + benchmark_returns).cumprod() - 1 )* 100

        fig.add_trace(
            go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Rebalancing Strategy',
                legendgroup='performance',
                hovertemplate='Strategy<br>%{x|%Y-%m-%d}<br>Multiple: %{y:.2f}%<extra></extra>'
            ),
            row=3,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=benchmark_cumret.index,
                y=benchmark_cumret,
                mode='lines',
                line=dict(color='orange', width=2),
                name='Equal-Weight Benchmark',
                legendgroup='performance',
                hovertemplate='Benchmark<br>%{x|%Y-%m-%d}<br>Multiple: %{y:.2f}%<extra></extra>'
            ),
            row=3,
            col=1
        )

        total_returns = self.pf.total_return() * 100

        y_values = np.concatenate(
            [
                portfolio_returns.to_numpy(dtype=float),
                benchmark_cumret.to_numpy(dtype=float),
            ]
        )
        y_min = np.nanmin(y_values)
        y_max = np.nanmax(y_values)
        y_span = max(y_max - y_min, 1e-9)
        y_pos = y_max + 0.05 * y_span

        fig.add_annotation(
            x=0.02,
            y=y_pos,
            xref='paper',
            yref='y3',
            text=f'Total Returns: {total_returns:.2f}%',
            showarrow=False,
            bgcolor='rgba(173, 216, 230, 0.7)',
            bordercolor='rgba(30, 30, 120, 0.4)',
            font=dict(size=12, family='DejaVu Sans')
        )

        fig.update_layout(
            height=950,
            width=1200,
            template='plotly_white',
            hovermode='x unified',
            font=dict(family='DejaVu Sans', size=13),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='left', x=0)
        )

        fig.update_xaxes(matches='x1')
        fig.update_yaxes(range=[0, 1], row=2, col=1)

        # fig.show()
        return fig

    def _plot_multi(self, index_levels, column_levels):
        """Plot vertically stacked heatmaps for multi-parameter statistics.

        Parameters
        ----------
        index_levels, column_levels
            Level names used when unstacking the statistics DataFrame.  These
            originate from the strategy's metadata and define the parameter grid
            shown on the y/x axes respectively.
        """
        pf_stat = self.stats()

        metrics = [
            ('total_return', 'Total Return', 'Viridis', 0.89),
            ('sharpe_ratio', 'Sharpe Ratio', 'RdBu', 0.5),
            ('max_drawdown', 'Max Drawdown', 'Inferno', 0.11)
        ]

        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.16,
            row_heights=[0.28] * len(metrics),
            subplot_titles=[m[1] for m in metrics]
        )

        def _format_labels(idx: pd.Index) -> list[str]:
            if isinstance(idx, pd.MultiIndex):
                labels: list[str] = []
                for key in idx.to_list():
                    parts = []
                    for lvl_name, value in zip(idx.names, key):
                        if isinstance(value, float):
                            value_str = f"{value:.6f}".rstrip('0').rstrip('.')
                        else:
                            value_str = str(value)
                        parts.append(value_str)
                    labels.append(', '.join(parts))
                return labels
            return [str(v) for v in idx]

        for row, (metric_key, display_name, colorscale, colorbar_y) in enumerate(metrics, start=1):
            pf_mtx = pf_stat[metric_key].vbt.unstack_to_df(
                symmetric=False,
                index_levels=index_levels,
                column_levels=column_levels
            )

            z_values = pf_mtx.values
            y_labels = _format_labels(pf_mtx.index)
            x_labels = _format_labels(pf_mtx.columns)

            fig.add_trace(
                go.Heatmap(
                    z=z_values,
                    x=x_labels,
                    y=y_labels,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=display_name,
                        # titleside='right',
                        len=0.24,
                        y=colorbar_y,
                        thickness=16
                    ),
                    hovertemplate='<b>' + display_name + '</b><br>' +
                                  'Index: %{y}<br>' +
                                  'Column: %{x}<br>' +
                                  'Value: %{z:.4f}<extra></extra>'
                ),
                row=row,
                col=1
            )

            fig.update_xaxes(
                row=row,
                col=1,
                title_text=', '.join(column_levels)
                # tickangle=-45
            )
            fig.update_yaxes(
                row=row,
                col=1,
                title_text=', '.join(index_levels)
            )

        fig.update_layout(
            height=380 * len(metrics),
            width=900,
            template='plotly_white',
            title='Portfolio Statistics Heatmaps',
            margin=dict(l=80, r=140, t=80, b=60)
        )

        # fig.show()
        return fig
    
    
    def clear(self) -> None:
        """No-op (kept for API compatibility). BackTest is stateless."""
        return None

