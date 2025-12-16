"""Streamlit page for Holt-Winters parameter optimization."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dffc._utils import validate_fund_code
from dffc.data_provider.eastmoney_provider import EastMoneyFundProvider
from dffc.holt_winters import HWOptimizer
from dffc.holt_winters._holt_winters import HW


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
FUND_CACHE_DIR = DATA_DIR / "fund"
RESULT_DIR = DATA_DIR / "hw_opt_results"


def _ensure_cache_dirs() -> None:
	FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	RESULT_DIR.mkdir(parents=True, exist_ok=True)
	DATA_DIR.mkdir(parents=True, exist_ok=True)


def _parse_code(raw: str) -> Tuple[Optional[str], Optional[str]]:
	if not raw:
		return None, None
	code = raw.strip().replace("，", ",").split(",")[0]
	try:
		return validate_fund_code(code), None
	except Exception:
		return None, code


def _load_fund_meta() -> Dict[str, str]:
	path = APP_DIR / "_utils" / "fund_meta.csv"
	if not path.exists():
		return {}
	try:
		df = pd.read_csv(path, dtype=str).fillna("")
		names: Dict[str, str] = {}
		for _, row in df.iterrows():
			code = str(row.get("code", "")).strip()
			if not code:
				continue
			name = row.get("name") or row.get("abbr") or ""
			names[code] = str(name)
		return names
	except Exception:
		return {}


def _read_cached_frame(code: str) -> Optional[pd.DataFrame]:
	pkl_path = FUND_CACHE_DIR / f"{code}.pkl"
	csv_path = FUND_CACHE_DIR / f"{code}.csv"
	path = pkl_path if pkl_path.exists() else csv_path if csv_path.exists() else None
	if path is None:
		return None
	try:
		if path.suffix == ".pkl":
			df = pd.read_pickle(path)
		else:
			df = pd.read_csv(path, parse_dates=["date"])
		if "date" in df.columns:
			df = df.rename(columns={"date": "_date_index"})
		if "_date_index" in df.columns:
			df = df.set_index("_date_index")
		df.index = pd.to_datetime(df.index)
		df.sort_index(inplace=True)
		return df
	except Exception:
		return None


def _should_persist(cached: Optional[pd.DataFrame], updated: pd.DataFrame) -> bool:
	if cached is None or cached.empty:
		return True
	if updated.empty:
		return False
	cache_latest_ts = pd.to_datetime(cached.index[-1])
	cache_latest_row = cached.iloc[-1]
	updated_latest_ts = pd.to_datetime(updated.index[-1])
	updated_latest_row = updated.iloc[-1]
	if cache_latest_ts != updated_latest_ts:
		return True
	return not cache_latest_row.equals(updated_latest_row)


def _persist_cache(code: str, df: pd.DataFrame) -> None:
	try:
		payload = df.copy()
		payload.index = pd.to_datetime(payload.index)
		payload.to_pickle(FUND_CACHE_DIR / f"{code}.pkl")
	except Exception:
		pass


def _refresh_fund_history(code: str) -> Optional[pd.DataFrame]:
	_ensure_cache_dirs()
	cached = _read_cached_frame(code)
	now = datetime.now()
	if cached is not None and not cached.empty:
		start_dt = pd.to_datetime(cached.index.max()) - timedelta(days=10)
		start_dt = start_dt.to_pydatetime()
	else:
		start_dt = datetime(2000, 1, 1)
	provider = EastMoneyFundProvider()
	try:
		fresh = provider.get_data(code, start_date=start_dt, end_date=now)
	except Exception:
		fresh = pd.DataFrame()
	frames: List[pd.DataFrame] = []
	if cached is not None and not cached.empty:
		frames.append(cached)
	if fresh is not None and not fresh.empty:
		frames.append(fresh)
	if not frames:
		return None
	merged = pd.concat(frames)
	merged.index = pd.to_datetime(merged.index)
	merged = merged[~merged.index.duplicated(keep="last")]
	merged.sort_index(inplace=True)
	if _should_persist(cached, merged):
		_persist_cache(code, merged)
	return merged


def _select_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
	for field in ("unit_value", "cumulative_value", "close", "price"):
		if field in df.columns:
			series = pd.to_numeric(df[field], errors="coerce").dropna()
			if not series.empty:
				return series
	numeric_cols = df.select_dtypes(include=[float, int]).columns
	if not numeric_cols.empty:
		series = df[numeric_cols[0]].dropna()
		return series
	return None


def _load_fund_series(codes: List[str]) -> Dict[str, pd.Series]:
	series_map: Dict[str, pd.Series] = {}
	for code in codes:
		frame = _refresh_fund_history(code)
		if frame is None or frame.empty:
			continue
		series = _select_price_series(frame)
		if series is None or series.empty:
			continue
		series = series.copy()
		series.name = code
		series_map[code] = series
	return series_map


def _summarize_series(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
	rows = []
	for code, ser in series_map.items():
		ser = ser.dropna()
		if ser.empty:
			continue
		rows.append(
			{
				"code": code,
				"start": ser.index.min().strftime("%Y-%m-%d"),
				"end": ser.index.max().strftime("%Y-%m-%d"),
				"rows": len(ser),
			}
		)
	return pd.DataFrame(rows)


def _render_results(summary: List[dict]) -> None:
	if not summary:
		st.info("暂无结果")
		return

	def _to_jsonable(items: List[dict]) -> List[dict]:
		converted: List[dict] = []
		for item in items:
			entry = dict(item)
			if isinstance(entry.get("results_df"), pd.DataFrame):
				entry["results_df"] = entry["results_df"].to_dict(orient="records")
			converted.append(entry)
		return converted

	df = pd.DataFrame(summary)
	if "params" in df.columns:
		params_df = df["params"].apply(pd.Series)
		params_df.columns = [f"param_{c}" for c in params_df.columns]
		df = pd.concat([df.drop(columns=["params"]), params_df], axis=1)
	st.dataframe(df, use_container_width=True)
	json_payload = json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2)
	st.download_button(
		"下载结果 JSON",
		data=json_payload,
		file_name="hw_opt_summary.json",
		mime="application/json",
	)


def _get_manual_params(code: str) -> Dict[str, float]:
	params_store = st.session_state.setdefault("hw_opt_manual_params", {})
	return params_store.get(code, {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "season_length": 0})


def _set_manual_params(code: str, params: Dict[str, float]) -> None:
	params_store = st.session_state.setdefault("hw_opt_manual_params", {})
	params_store[code] = {
		"alpha": float(params.get("alpha", 0.0)),
		"beta": float(params.get("beta", 0.0)),
		"gamma": float(params.get("gamma", 0.0)),
		"season_length": int(params.get("season_length", 0)),
	}


def _get_preview_series(code: str) -> Optional[pd.Series]:
	cache = st.session_state.setdefault("hw_opt_preview_cache", {})
	if code in cache:
		return cache[code]
	series_map = _load_fund_series([code])
	if not series_map:
		return None
	series = next(iter(series_map.values()))
	cache[code] = series
	return series


def _get_preview_params() -> Dict[str, float]:
	params = st.session_state.setdefault(
		"hw_opt_preview_params",
		{"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "season_length": 0},
	)
	return {
		"alpha": float(params.get("alpha", 0.0)),
		"beta": float(params.get("beta", 0.0)),
		"gamma": float(params.get("gamma", 0.0)),
		"season_length": int(params.get("season_length", 0)),
	}


def _render_hw_curve(code: str, series: pd.Series, params: Dict[str, float]) -> None:
	series = pd.to_numeric(series, errors="coerce").dropna()
	if series.empty:
		st.warning("当前基金无可用价格数据，无法绘制。")
		return
	season = int(params.get("season_length", 0) or 0)
	alpha = float(params.get("alpha", 0.0) or 0.0)
	beta = float(params.get("beta", 0.0) or 0.0)
	gamma = float(params.get("gamma", 0.0) or 0.0)
	if season <= 0:
		st.info("season_length 需大于 0 才能计算平滑曲线。")
		return
	if len(series) < season * 2:
		st.info("样本长度不足以进行 Holt-Winters 平滑，请增加数据或减小季节长度。")
		return
	try:
		indicator = HW.run(series, alpha=alpha, beta=beta, gamma=gamma, season_length=season, multiplicative=True)
		smooth = pd.Series(indicator.hw, index=series.index)
	except Exception as exc:
		st.error(f"计算平滑曲线失败: {exc}")
		return
	ma30 = series.rolling(window=30, min_periods=1, center=True).mean()
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=series.index, y=series.values, name=f"{code} 净值", mode="lines", line=dict(color="#7f7f7f")))
	fig.add_trace(go.Scatter(x=smooth.index, y=smooth.values, name="HW 平滑", mode="lines", line=dict(color="#d62728")))
	fig.add_trace(go.Scatter(x=ma30.index, y=ma30.values, name="30日均线", mode="lines", line=dict(color="#1f77b4")))
	fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	st.plotly_chart(fig, use_container_width=True)


def _render_objective_curve(results_df: pd.DataFrame, title: str = "优化目标") -> None:
	metric_col = next((col for col in ["mse", "rmse", "objective"] if col in results_df.columns), None)
	if metric_col is None:
		st.info("结果中未找到可用的目标函数列。")
		return
	df = results_df.copy()
	df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
	df = df.dropna(subset=[metric_col])
	if df.empty:
		st.info("目标函数数据为空。")
		return
	if "end_day" in df.columns:
		x_axis = df["end_day"].apply(lambda v: "None" if pd.isna(v) else int(v))
		x_label = "end_day"
	else:
		x_axis = list(range(len(df)))
		x_label = "step"
	window = max(3, min(20, len(df) // 5 or 1))
	df["ma"] = df[metric_col].rolling(window=window, min_periods=1).mean()
	fig = go.Figure()
	fig.add_trace(
		go.Scatter(x=x_axis, y=df[metric_col], mode="lines+markers", name=metric_col, line=dict(color="#1f77b4"))
	)
	fig.add_trace(
		go.Scatter(x=x_axis, y=df["ma"], mode="lines", name=f"{metric_col} 滑动均值", line=dict(color="#d62728", dash="dash"))
	)
	fig.update_layout(height=360, xaxis_title=x_label, yaxis_title=metric_col, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	st.markdown(f"#### {title}")
	st.plotly_chart(fig, use_container_width=True)


def _render_param_curves(summary: List[dict]) -> None:
	"""Plot alpha/beta/gamma and season_length vs end_day for the first successful asset."""
	if not summary:
		return
	item = None
	for x in summary:
		if x.get("status") != "success":
			continue
		if "results_df" not in x:
			continue
		item = x
		break
	if item is None:
		return
	results = item.get("results_df")
	if isinstance(results, list):
		df = pd.DataFrame(results)
	elif isinstance(results, pd.DataFrame):
		df = results.copy()
	else:
		return
	if df.empty:
		return
	needed = {"end_day", "alpha", "beta", "gamma", "season_length"}
	if not needed.issubset(set(df.columns)):
		return
	df = df[df["end_day"].notna() | df["end_day"].isna()].copy()
	df["end_day_str"] = df["end_day"].apply(lambda v: "None" if pd.isna(v) else str(int(v)))
	df.sort_values(by="end_day", inplace=True, na_position="last")
	for col in ["alpha", "beta", "gamma"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df["season_length"] = pd.to_numeric(df["season_length"], errors="coerce")
	if df[["alpha", "beta", "gamma", "season_length"]].dropna(how="all").empty:
		return
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	fig.add_trace(go.Scatter(x=df["end_day_str"], y=df["alpha"], mode="lines+markers", name="alpha"), secondary_y=False)
	fig.add_trace(go.Scatter(x=df["end_day_str"], y=df["beta"], mode="lines+markers", name="beta"), secondary_y=False)
	fig.add_trace(go.Scatter(x=df["end_day_str"], y=df["gamma"], mode="lines+markers", name="gamma"), secondary_y=False)
	fig.add_trace(
		go.Scatter(x=df["end_day_str"], y=df["season_length"], mode="lines+markers", name="season_length", line=dict(dash="dash")),
		secondary_y=True,
	)
	# 标记最优参数点（使用最新 end_day 记录）
	if "mse" in df.columns:
		best_row = df.loc[df["mse"].idxmin()]
	else:
		best_row = df.iloc[-1]
	fig.add_trace(
		go.Scatter(
			x=[best_row["end_day_str"]],
			y=[best_row["alpha"]],
			mode="markers",
			marker=dict(color="#ff7f0e", size=12, symbol="star"),
			name="best alpha",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=[best_row["end_day_str"]],
			y=[best_row["beta"]],
			mode="markers",
			marker=dict(color="#2ca02c", size=12, symbol="star"),
			name="best beta",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=[best_row["end_day_str"]],
			y=[best_row["gamma"]],
			mode="markers",
			marker=dict(color="#d62728", size=12, symbol="star"),
			name="best gamma",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=[best_row["end_day_str"]],
			y=[best_row["season_length"]],
			mode="markers",
			marker=dict(color="#9467bd", size=12, symbol="star"),
			name="best season_length",
		),
		secondary_y=True,
	)
	fig.update_yaxes(title_text="alpha / beta / gamma", secondary_y=False)
	fig.update_yaxes(title_text="season_length", secondary_y=True)
	fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	st.markdown("#### 参数稳定性")
	st.plotly_chart(fig, use_container_width=True)


def main() -> None:
	st.title("Holt-Winters 参数优化")
	st.caption("选择单个基金，更新数据并优化 Holt-Winters 参数。")

	# 数据概览（置顶展示，覆盖已加载与预览缓存）
	overview_series: Dict[str, pd.Series] = {}
	overview_series.update({k: v for k, v in st.session_state.get("hw_opt_series_map", {}).items() if isinstance(v, pd.Series)})
	overview_series.update({k: v for k, v in st.session_state.get("hw_opt_preview_cache", {}).items() if isinstance(v, pd.Series)})
	if overview_series:
		st.subheader("数据概览")
		info_df = _summarize_series(overview_series)
		st.dataframe(info_df, use_container_width=True)

	meta_fund_names = _load_fund_meta()
	fund_options = sorted(meta_fund_names.keys()) or ["007467"]
	with st.form("hw-opt-form"):
		code_choice = st.selectbox(
			"添加基金代码",
			options=fund_options,
			format_func=lambda x: f"{x} - {meta_fund_names.get(x, '')}".strip(" -"),
		)
		max_workers = st.slider("并行进程数", min_value=1, max_value=12, value=4, step=1)
		with st.expander("优化选项", expanded=False):
			col_a, col_b, col_c = st.columns(3)
			with col_a:
				end_start = st.number_input("end_day 起始", value=-400, step=10, format="%d")
			with col_b:
				end_stop = st.number_input("end_day 终止(不含)", value=0, step=10, format="%d")
			with col_c:
				end_step = st.number_input("end_day 间隔", value=40, min_value=1, step=1, format="%d")
		load_btn = st.form_submit_button("更新并加载数据")
		run_btn = st.form_submit_button("运行优化")

	st.markdown("## Holt-Winters 平滑预览")
	default_preview = st.session_state.get("hw_opt_preview_code") or (fund_options[0] if fund_options else "")
	with st.form("hw-preview-form"):
		preview_code = st.selectbox(
			"选择基金",
			options=fund_options,
			format_func=lambda x: f"{x} - {meta_fund_names.get(x, '')}".strip(" -"),
			index=fund_options.index(default_preview) if default_preview in fund_options else 0,
		)
		params_default = _get_preview_params()
		col_a, col_b, col_c, col_d = st.columns(4)
		with col_a:
			alpha_input = st.number_input(
				"alpha",
				min_value=0.0,
				max_value=1.0,
				value=float(params_default.get("alpha", 0.0)),
				step=0.01,
				format="%f",
				key="hw_prev_alpha",
			)
		with col_b:
			beta_input = st.number_input(
				"beta",
				min_value=0.0,
				max_value=1.0,
				value=float(params_default.get("beta", 0.0)),
				step=0.01,
				format="%f",
				key="hw_prev_beta",
			)
		with col_c:
			gamma_input = st.number_input(
				"gamma",
				min_value=0.0,
				max_value=1.0,
				value=float(params_default.get("gamma", 0.0)),
				step=0.01,
				format="%f",
				key="hw_prev_gamma",
			)
		with col_d:
			season_input = st.number_input(
				"season_length",
				min_value=0,
				value=int(params_default.get("season_length", 0)),
				step=1,
				format="%d",
				key="hw_prev_season",
			)
		plot_btn = st.form_submit_button("绘制平滑曲线", use_container_width=True)

	if plot_btn:
		params_current = {"alpha": alpha_input, "beta": beta_input, "gamma": gamma_input, "season_length": season_input}
		st.session_state["hw_opt_preview_params"] = params_current
		series = _get_preview_series(preview_code)
		if series is None or series.empty:
			st.error("未获取到基金数据，请检查代码或网络。")
		else:
			st.session_state["hw_opt_preview_code"] = preview_code
			_render_hw_curve(preview_code, series, params_current)

	if load_btn:
		valid_code, invalid_code = _parse_code(code_choice)
		if invalid_code:
			st.warning(f"基金代码格式未通过校验，已跳过：{invalid_code}")
		if not valid_code:
			st.error("请输入有效基金代码。")
		else:
			with st.spinner("正在更新/加载基金数据..."):
				series_map = _load_fund_series([valid_code])
			if not series_map:
				st.error("未获取到基金数据，请检查代码或网络。")
			else:
				st.session_state["hw_opt_series_map"] = series_map
				price_df = pd.concat(series_map.values(), axis=1)
				price_df.sort_index(inplace=True)
				st.session_state["hw_opt_price_df"] = price_df
				_set_manual_params(valid_code, _get_manual_params(valid_code))
				st.success(f"已加载基金 {valid_code} 的数据。")

	series_map = st.session_state.get("hw_opt_series_map", {})
	if series_map:
		code = st.session_state.get("hw_opt_preview_code") or next(iter(series_map.keys()))
		st.session_state["hw_opt_preview_code"] = code

	if run_btn:
		price_df = st.session_state.get("hw_opt_price_df")
		if price_df is None or price_df.empty:
			st.error("请先加载基金数据。")
		else:
			end_list = list(range(int(end_start), int(end_stop), int(end_step)))
			end_list.append(None)
			if not end_list:
				st.error("end_day_list 为空，请检查优化选项。")
				return
			progress = st.progress(0.0, text="准备优化...")
			status = st.empty()
			optimizer = HWOptimizer(price_df)

			def _progress_cb(done: int, total: int, code: str, end_day: Optional[int]):
				total_safe = max(total, 1)
				fraction = min(1.0, done / total_safe)
				label = f"优化 {code} end_day={end_day}"
				progress.progress(fraction, text=label)
				status.write(label)

			with st.spinner("正在运行优化..."):
				summary = optimizer.optimize(
					save=True,
					output_base_dir=str(RESULT_DIR),
					result_filename="hw_opt_summary",
					max_workers=max_workers,
					end_day_list=end_list,
					progress_cb=_progress_cb,
				)
			st.session_state["hw_opt_summary"] = summary
			# 将首个成功资产的最优参数写回输入默认值
			first_success = next((x for x in summary if x.get("status") == "success" and "params" in x), None)
			if first_success:
				best_params = first_success.get("params", {})
				_set_manual_params(first_success.get("code", code_choice), best_params)
			st.success("优化完成。")

	summary = st.session_state.get("hw_opt_summary")
	if summary:
		st.subheader("优化结果")
		_render_results(summary)
		_render_param_curves(summary)
		code = st.session_state.get("hw_opt_preview_code")
		if code:
			st.markdown("#### 最佳参数")
			best_row = next((x.get("params", {}) for x in summary if x.get("status") == "success" and x.get("params") and x.get("code") == code), None)
			if best_row:
				best_df = pd.DataFrame([best_row])
				st.dataframe(best_df, use_container_width=True)
			result_item = next((x for x in summary if x.get("code") == code and x.get("results_df") is not None), None)
			if result_item:
				res_df = result_item.get("results_df")
				if isinstance(res_df, list):
					res_df = pd.DataFrame(res_df)
				elif isinstance(res_df, pd.DataFrame):
					res_df = res_df.copy()
				if isinstance(res_df, pd.DataFrame) and not res_df.empty:
					_render_objective_curve(res_df, title="优化目标（含滑动均值）")


if __name__ == "__main__":
	main()