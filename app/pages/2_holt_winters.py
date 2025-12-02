"""Streamlit page for Holt-Winters asset monitoring."""

from __future__ import annotations

import json
import math
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import dffc
from dffc.data_provider.fund_est_val_crawler import FundEstimateProvider
from dffc.holt_winters._holt_winters import HWDP

try:  # Streamlit >=1.18 ships this helper, otherwise users may install plugin.
	from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	st_autorefresh = None


APP_DIR = Path(__file__).resolve().parents[1]
META_DIR = APP_DIR / "_utils"
AUTO_REFRESH_MS = 30_000  # 半分钟刷新估值


st.set_page_config(
	page_title="Holt-Winters 资产面板",
	layout="wide",
	initial_sidebar_state="collapsed",
)


@dataclass
class FundPanelRow:
	name: str
	code: str
	nav_date: datetime
	nav_change_pct: Optional[float]
	est_change_pct: Optional[float]
	estimate_hdp: Optional[float]
	current_hdp: Optional[float]


def _auto_refresh() -> None:
	if st_autorefresh is None:
		st.caption("未检测到 st_autorefresh，已维持手动刷新。可安装 streamlit-autorefresh 获得自动刷新。")
		return
	st_autorefresh(interval=AUTO_REFRESH_MS, key="hw-fund-panel-refresh")


@st.cache_data(show_spinner=False)
def _load_fund_meta() -> Dict[str, str]:
	path = META_DIR / "fund_meta.csv"
	if not path.exists():
		return {}
	df = pd.read_csv(path, dtype=str).fillna("")
	names = {}
	for _, row in df.iterrows():
		code = str(row.get("code", "")).strip()
		if not code:
			continue
		name = row.get("name") or row.get("abbr") or ""
		names[code] = name
	return names


def _parse_hw_params(payload: Optional[bytes]) -> Dict[str, dict]:
	if not payload:
		return {}
	try:
		data = json.loads(payload.decode("utf-8"))
	except Exception as exc:  # pragma: no cover - Streamlit UI feedback path
		st.error(f"无法解析 Holt-Winters 参数文件: {exc}")
		return {}
	params: Dict[str, dict] = {}
	if isinstance(data, list):
		entries = data
	else:
		entries = [data]
	for item in entries:
		code = str(item.get("code", "")).strip()
		param_set = item.get("params", {})
		if not code or not param_set:
			continue
		params[code] = param_set
	return params


def _load_assets_from_pickle(payload: bytes) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
	with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
		tmp.write(payload)
		tmp.flush()
		try:
			obj = dffc.FundData.load(tmp.name)
		except Exception:
			tmp.seek(0)
			obj = pickle.load(tmp)

	fund_frames: Dict[str, pd.DataFrame] = {}
	fund_names: Dict[str, str] = {}
	if isinstance(obj, dffc.FundData):
		obj = obj.update()
		for symbol in obj.symbols:
			frame = obj.data[symbol].copy()
			frame.index = pd.to_datetime(frame.index)
			frame.sort_index(inplace=True)
			fund_frames[str(symbol)] = frame
		if getattr(obj, "names", None):
			fund_names.update({str(k): v for k, v in obj.names.items()})
	elif isinstance(obj, dict):
		for key, value in obj.items():
			if isinstance(value, pd.DataFrame):
				frame = value.copy()
				frame.index = pd.to_datetime(frame.index)
				frame.sort_index(inplace=True)
				fund_frames[str(key)] = frame
	elif isinstance(obj, pd.DataFrame):
		frame = obj.copy()
		frame.index = pd.to_datetime(frame.index)
		frame.sort_index(inplace=True)
		fund_frames["unknown"] = frame
	else:
		raise ValueError("上传的 PKL 文件不包含可识别的基金数据结构")
	return fund_frames, fund_names


@st.cache_data(show_spinner=False, ttl=55)
def _fetch_estimate_snapshot(code: str) -> Optional[Dict[str, object]]:
	try:
		provider = FundEstimateProvider()
		df = provider.get_data(code)
	except Exception:
		return None
	if df is None or df.empty:
		return None
	row = df.iloc[-1]
	return {
		"estimate_value": row.get("estimate_value"),
		"estimate_change_pct": row.get("estimate_change_pct"),
		"timestamp": row.get("estimate_timestamp") or row.get("date"),
	}


def _select_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
	for field in ("unit_value", "cumulative_value", "close", "price"):
		if field in df.columns:
			series = pd.to_numeric(df[field], errors="coerce").dropna()
			if not series.empty:
				return series
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	if not numeric_cols.empty:
		series = df[numeric_cols[0]].dropna()
		return series
	return None


def _run_hwdp(series: pd.Series, params: dict) -> Optional[pd.Series]:
	alpha = float(params.get("alpha", 0.1))
	beta = float(params.get("beta", 0.0))
	gamma = float(params.get("gamma", 0.1))
	season_length = int(params.get("season_length", 20))
	if len(series) < season_length * 2:
		return None
	try:
		indicator = HWDP.run(
			series,
			alpha=alpha,
			beta=beta,
			gamma=gamma,
			season_length=season_length,
			multiplicative=True,
		)
	except Exception:
		return None
	return indicator.hwdp


def _coerce_percent(value: object) -> Optional[float]:
	if value is None:
		return None
	if isinstance(value, str):
		value = value.strip().replace("%", "")
		if not value:
			return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _build_fund_panel(
	fund_frames: Dict[str, pd.DataFrame],
	fund_names: Dict[str, str],
	meta_names: Dict[str, str],
	hw_params: Dict[str, dict],
) -> pd.DataFrame:
	rows: List[FundPanelRow] = []
	for code, frame in fund_frames.items():
		if frame.empty:
			continue
		latest = frame.iloc[-1]
		nav_date = pd.to_datetime(frame.index[-1])
		nav_change_pct = _coerce_percent(latest.get("daily_growth_rate")) * 100
		price_series = _select_price_series(frame)
		if nav_change_pct is None and price_series is not None and len(price_series) >= 2:
			nav_change_pct = (price_series.iloc[-1] / price_series.iloc[-2] - 1.0) * 100

		params = hw_params.get(code)
		current_hdp = None
		estimate_hdp = None
		estimate_change_pct = None
		if params and price_series is not None:
			hwdp_series = _run_hwdp(price_series, params)
			if hwdp_series is not None and not hwdp_series.empty:
				current_hdp = float(hwdp_series.iloc[-1]) * 100
			estimate = _fetch_estimate_snapshot(code)
			if estimate:
				est_change_pct = estimate.get("estimate_change_pct")
				if isinstance(est_change_pct, (int, float)):
					estimate_change_pct = float(est_change_pct) * 100
				estimate_value = estimate.get("estimate_value")
				timestamp = estimate.get("timestamp")
				if estimate_value and price_series is not None:
					augmented = price_series.copy()
					new_index = pd.to_datetime(timestamp) if timestamp is not None else pd.Timestamp.utcnow()
					if new_index in augmented.index:
						new_index += pd.Timedelta(seconds=1)
					augmented = pd.concat(
						[augmented, pd.Series([float(estimate_value)], index=[new_index])]
					)
					aug_hwdp = _run_hwdp(augmented, params)
					if aug_hwdp is not None and not aug_hwdp.empty:
						estimate_hdp = float(aug_hwdp.iloc[-1]) * 100
		else:
			estimate = _fetch_estimate_snapshot(code)
			if estimate:
				est_change_pct = estimate.get("estimate_change_pct")
				if isinstance(est_change_pct, (int, float)):
					estimate_change_pct = float(est_change_pct) * 100

		name = fund_names.get(code) or fund_names.get(code.lstrip("0"))
		if not name:
			name = meta_names.get(code) or meta_names.get(code.lstrip("0")) or "--"
		rows.append(
			FundPanelRow(
				name=name,
				code=code,
				nav_date=nav_date,
				nav_change_pct=nav_change_pct,
				est_change_pct=estimate_change_pct,
				estimate_hdp=estimate_hdp,
				current_hdp=current_hdp,
			)
		)

	if not rows:
		return pd.DataFrame()
	panel_records: List[Dict[str, object]] = []
	for row in rows:
		panel_records.append(
			{
				"基金": row.name,
				"代码": row.code,
				"净值日期": row.nav_date.strftime("%Y-%m-%d"),
				"净值变化%": row.nav_change_pct,
				"当日估值变化%": row.est_change_pct,
				"估值HDP": row.estimate_hdp,
				"当前HDP": row.current_hdp,
			}
		)
	return pd.DataFrame(panel_records)


def _render_fund_panel(panel_df: pd.DataFrame) -> None:
	if panel_df.empty:
		st.info("上传的资产中暂无可展示的基金。")
		return
	numeric_cols = ["净值变化%", "当日估值变化%", "估值HDP", "当前HDP"]

	def _pct_fmt(value: Optional[float]) -> str:
		if value is None or (isinstance(value, float) and math.isnan(value)):
			return "--"
		return f"{value:.2f}%"

	def _colorize(value: Optional[float]) -> str:
		if value is None or (isinstance(value, float) and math.isnan(value)):
			return ""
		if value > 0:
			return "color:#d74242;"
		if value < 0:
			return "color:#0b8b38;"
		return ""

	format_map = {col: _pct_fmt for col in numeric_cols if col in panel_df.columns}
	style = panel_df.style.format(format_map)
	for col in numeric_cols:
		if col in panel_df.columns:
			style = style.applymap(_colorize, subset=col)
	st.dataframe(style, use_container_width=True, hide_index=True)


def main() -> None:
	_auto_refresh()
	st.title("Holt-Winters 资产面板")
	if st.button("⬅️ 返回主面板", use_container_width=True):
		st.switch_page("dffc_app.py")
	st.write("上传基金资产 PKL 文件与 Holt-Winters 参数 JSON，系统将实时展示基金估值与 HDP 状态。")

	col_asset, col_param = st.columns(2)
	with col_asset:
		asset_file = st.file_uploader("上传资产文件 (PKL)", type=["pkl"], key="fund-pkl")
	with col_param:
		param_file = st.file_uploader("上传 Holt-Winters 参数 (JSON)", type=["json"], key="hw-json")

	if not asset_file:
		st.info("请先上传数据页导出的 FundData PKL 文件。")
		return

	try:
		fund_frames, fund_names = _load_assets_from_pickle(asset_file.getvalue())
	except Exception as exc:
		st.error(f"载入资产文件失败: {exc}")
		return

	params_map = _parse_hw_params(param_file.getvalue() if param_file else None)
	meta_names = _load_fund_meta()

	st.subheader("基金资产面板")
	if not params_map:
		st.warning("未上传或无法解析 Holt-Winters 参数，HDP 将以 '--' 展示。")

	fund_panel = _build_fund_panel(fund_frames, fund_names, meta_names, params_map)
	_render_fund_panel(fund_panel)

	st.subheader("股票资产面板")
	st.info("股票资产展示暂未实现，后续版本将补充。")

	st.caption(f"最后刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
	main()
