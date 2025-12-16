"""Streamlit page for Holt-Winters asset monitoring."""

from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qual
import streamlit as st

import dffc
from dffc._utils import validate_fund_code, validate_stock_code
from dffc.data_provider.eastmoney_provider import EastMoneyFundProvider, EastMoneyStockProvider
from dffc.data_provider.fund_est_val_crawler import FundEstimateProvider
from dffc.holt_winters._holt_winters import HW, HWDP

try:  # Streamlit >=1.18 ships this helper, otherwise users may install plugin.
	from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	st_autorefresh = None


APP_DIR = Path(__file__).resolve().parents[1]
META_DIR = APP_DIR / "_utils"
DATA_DIR = APP_DIR / "data"
FUND_CACHE_DIR = DATA_DIR / "fund"
STOCK_CACHE_DIR = DATA_DIR / "stock"
PARAM_CACHE_DIR = DATA_DIR / "fund_parameters"
ASSET_LIST_PATH = DATA_DIR / "asset_lists.json"
AUTO_REFRESH_MS = 30_000  # 半分钟刷新估值


st.set_page_config(
	page_title="HDP 资产监控面板",
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
	if st.session_state.get("hw_auto_refresh", False):
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


@st.cache_data(show_spinner=False)
def _load_stock_meta() -> Dict[str, str]:
	path = META_DIR / "stock_meta.csv"
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
	"""Parse parameter JSON payload; tolerate missing keys."""
	if not payload:
		return {}
	try:
		data = json.loads(payload.decode("utf-8"))
	except Exception as exc:  # pragma: no cover - Streamlit UI feedback path
		st.error(f"无法解析 Holt-Winters 参数文件: {exc}")
		return {}
	entries = data if isinstance(data, list) else [data]
	params: Dict[str, dict] = {}
	for item in entries:
		if not isinstance(item, dict):
			continue
		code = str(item.get("code", "")).strip()
		raw_params = item.get("params") or {}
		# fill missing keys with None for consistency
		param_set = {
			"alpha": raw_params.get("alpha"),
			"beta": raw_params.get("beta"),
			"gamma": raw_params.get("gamma"),
			"season_length": raw_params.get("season_length"),
			"mse": raw_params.get("mse"),
		}
		if not code:
			continue
		params[code] = param_set
	return params


def _load_cached_params() -> Dict[str, dict]:
	"""Load cached parameter JSON files under PARAM_CACHE_DIR."""
	if not PARAM_CACHE_DIR.exists():
		return {}
	param_map: Dict[str, dict] = {}
	for path in PARAM_CACHE_DIR.glob("*.json"):
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			code = str(data.get("code", "")).strip()
			raw_params = data.get("params") or {}
			param_set = {
				"alpha": raw_params.get("alpha"),
				"beta": raw_params.get("beta"),
				"gamma": raw_params.get("gamma"),
				"season_length": raw_params.get("season_length"),
				"mse": raw_params.get("mse"),
			}
			if code:
				param_map[code] = param_set
		except Exception:
			continue
	return param_map


def _normalize_param_entry(data: dict) -> Optional[Tuple[str, dict, dict]]:
	"""Return (code, full_entry, param_set) if valid; fill missing keys with None."""
	if not isinstance(data, dict):
		return None
	code = str(data.get("code", "")).strip()
	raw_params = data.get("params") or {}
	param_set = {
		"alpha": raw_params.get("alpha"),
		"beta": raw_params.get("beta"),
		"gamma": raw_params.get("gamma"),
		"season_length": raw_params.get("season_length"),
		"mse": raw_params.get("mse"),
	}
	if not code:
		return None
	full_entry = {
		"code": code,
		"status": data.get("status"),
		"data_points": data.get("data_points"),
		"params": param_set,
		"results_df": data.get("results_df", []),
	}
	return code, full_entry, param_set


def _save_param_json(payload: str) -> Optional[List[str]]:
	"""Validate and persist parameter JSON text, returning saved codes on success."""
	try:
		parsed = json.loads(payload)
	except Exception as exc:
		st.error(f"JSON 解析失败: {exc}")
		return None
	entries = parsed if isinstance(parsed, list) else [parsed]
	saved: List[str] = []
	for entry in entries:
		normalized = _normalize_param_entry(entry)
		if not normalized:
			continue
		code, full_entry, _ = normalized
		try:
			validate_fund_code(code)
		except Exception:
			# 仍然缓存，但提示用户
			st.warning(f"基金代码格式异常，已按原样保存：{code}")
		try:
			PARAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
			out_path = PARAM_CACHE_DIR / f"{code}.json"
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(full_entry, f, ensure_ascii=False, indent=2)
			saved.append(code)
		except Exception as exc:
			st.error(f"保存 {code} 失败: {exc}")
	if not saved:
		return None
	return saved


def _load_assets_from_pickle(payload: bytes) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
	tmp_path = ""
	try:
		with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
			tmp.write(payload)
			tmp_path = tmp.name
		
		try:
			obj = dffc.FundData.load(tmp_path)
		except Exception:
			with open(tmp_path, "rb") as f:
				obj = pickle.load(f)
	finally:
		if tmp_path and os.path.exists(tmp_path):
			try:
				os.unlink(tmp_path)
			except Exception:
				pass

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
		raise ValueError(f"上传的 PKL 文件不包含可识别的基金数据结构: {type(obj)}")
	return fund_frames, fund_names


def _normalize_codes(raw_inputs: List[str], validator) -> Tuple[List[str], List[str]]:
	valid_codes: List[str] = []
	invalid: List[str] = []
	for raw in raw_inputs:
		if not raw:
			continue
		clean_text = str(raw).replace("，", ",")
		parts = [p.strip() for p in clean_text.split(",") if p.strip()]
		for part in parts:
			try:
				valid_codes.append(validator(part))
			except Exception:
				invalid.append(part)
	return valid_codes, invalid


def _load_asset_lists() -> List[dict]:
	if not ASSET_LIST_PATH.exists():
		return []
	try:
		with open(ASSET_LIST_PATH, "r", encoding="utf-8") as f:
			data = json.load(f)
		if not isinstance(data, list):
			return []
		normalized = []
		for item in data:
			if not isinstance(item, dict):
				continue
			list_id = item.get("id")
			name = item.get("name") or "Unnamed"
			codes = item.get("codes") or item.get("list") or []
			if list_id is None:
				list_id = len(normalized) + 1
			codes = [str(c).strip() for c in codes if str(c).strip()]
			normalized.append({"id": list_id, "name": str(name), "codes": codes})
		return normalized
	except Exception:
		return []


def _save_asset_list(name: str, codes: List[str]) -> Optional[int]:
	if not codes:
		st.warning("当前基金列表为空，未保存。")
		return None
	name = name.strip() or "New List"
	codes_unique = sorted(set(codes))
	lists = _load_asset_lists()
	max_id = max([item.get("id", 0) for item in lists], default=0)
	new_id = max_id + 1
	entry = {"id": new_id, "name": name, "codes": codes_unique}
	lists.append(entry)
	try:
		with open(ASSET_LIST_PATH, "w", encoding="utf-8") as f:
			json.dump(lists, f, ensure_ascii=False, indent=2)
		return new_id
	except Exception as exc:
		st.error(f"保存资产列表失败: {exc}")
		return None


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


def _ensure_cache_dirs() -> None:
	FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	STOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	PARAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_cached_frame(cache_dir: Path, code: str) -> Optional[pd.DataFrame]:
	# 优先读取 pkl，如无则回退 csv
	pkl_path = cache_dir / f"{code}.pkl"
	csv_path = cache_dir / f"{code}.csv"
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


def _persist_cache(cache_dir: Path, code: str, df: pd.DataFrame) -> None:
	try:
		cache_dir.mkdir(parents=True, exist_ok=True)
		payload = df.copy()
		payload.index = pd.to_datetime(payload.index)
		payload.to_pickle(cache_dir / f"{code}.pkl")
	except Exception:
		pass


def _refresh_fund_history(code: str) -> Optional[pd.DataFrame]:
	_ensure_cache_dirs()
	cached = _read_cached_frame(FUND_CACHE_DIR, code)
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
		_persist_cache(FUND_CACHE_DIR, code, merged)
	return merged


def _refresh_stock_history(code: str) -> Optional[pd.DataFrame]:
	_ensure_cache_dirs()
	cached = _read_cached_frame(STOCK_CACHE_DIR, code)
	now = datetime.now()
	start_dt = now - timedelta(days=400)
	if cached is not None and not cached.empty:
		start_dt = pd.to_datetime(cached.index.max()) - timedelta(days=3)
	provider = EastMoneyStockProvider()
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
		_persist_cache(STOCK_CACHE_DIR, code, merged)
	return merged


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
		nav_change_pct = _coerce_percent(latest.get("daily_growth_rate"))
		if nav_change_pct is not None:
			nav_change_pct *= 100
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


def _build_stock_panel(
	stock_frames: Dict[str, pd.DataFrame],
	stock_names: Dict[str, str],
	meta_names: Dict[str, str],
) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	for code, frame in stock_frames.items():
		if frame.empty:
			continue
		latest = frame.iloc[-1]
		trade_date = pd.to_datetime(frame.index[-1])
		close_price = latest.get("close") or latest.get("price")
		change_pct = latest.get("change_percent")
		if isinstance(change_pct, str):
			try:
				change_pct = float(change_pct)
			except Exception:
				change_pct = None
		if isinstance(change_pct, (int, float)):
			change_pct = float(change_pct) * 100
		name = stock_names.get(code) or stock_names.get(code.lstrip("0"))
		if not name:
			name = meta_names.get(code) or meta_names.get(code.lstrip("0")) or "--"
		rows.append(
			{
				"股票": name,
				"代码": code,
				"日期": trade_date.strftime("%Y-%m-%d"),
				"收盘": close_price,
				"涨跌幅%": change_pct,
			}
		)
	return pd.DataFrame(rows)


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
	style = (
		panel_df.style.format(format_map)
		.set_properties(**{"text-align": "center"})
		.set_table_styles(
			[
				{"selector": "th", "props": [("text-align", "center")]},
				{"selector": "td", "props": [("text-align", "center")]},
			]
		)
	)
	for col in numeric_cols:
		if col in panel_df.columns:
			style = style.applymap(_colorize, subset=col)
	st.dataframe(style, use_container_width=True, hide_index=True)


def _render_stock_panel(panel_df: pd.DataFrame) -> None:
	if panel_df.empty:
		st.info("当前列表中暂无可展示的股票。")
		return

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

	style = panel_df.style.format({"涨跌幅%": _pct_fmt})
	if "涨跌幅%" in panel_df.columns:
		style = style.applymap(_colorize, subset="涨跌幅%")
	st.dataframe(style, use_container_width=True, hide_index=True)


def _make_hw_fig(
	selected_codes: List[str],
	fund_frames: Dict[str, pd.DataFrame],
	params_map: Dict[str, dict],
) -> Optional[go.Figure]:
	if not selected_codes:
		return None
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	palette = qual.Plotly
	color_map: Dict[str, str] = {}
	any_added = False
	for code in selected_codes:
		frame = fund_frames.get(code)
		if frame is None or frame.empty:
			st.warning(f"{code} 无可用数据，已跳过绘图。")
			continue
		price_series = _select_price_series(frame)
		if price_series is None or price_series.empty:
			st.warning(f"{code} 无法识别价格列，已跳过绘图。")
			continue
		params = params_map.get(code)
		if not params:
			st.warning(f"{code} 缺少 Holt-Winters 参数，已跳过绘图。")
			continue
		alpha = params.get("alpha", 0.1)
		beta = params.get("beta", 0.0)
		gamma = params.get("gamma", 0.1)
		season_length = int(params.get("season_length", 20) or 20)
		if len(price_series) < season_length * 2:
			st.warning(f"{code} 数据不足（需要 >= 2*season_length），已跳过绘图。")
			continue
		try:
			hw_indicator = HW.run(
				price_series,
				alpha=alpha,
				beta=beta,
				gamma=gamma,
				season_length=season_length,
				multiplicative=True,
			)
			hwdp_indicator = HWDP.run(
				price_series,
				alpha=alpha,
				beta=beta,
				gamma=gamma,
				season_length=season_length,
				multiplicative=True,
			)
		except Exception as exc:
			st.warning(f"{code} 绘图计算失败: {exc}")
			continue

		color = color_map.get(code)
		if not color:
			color = palette[len(color_map) % len(palette)]
			color_map[code] = color
		fig.add_trace(
			go.Scatter(
				x=price_series.index,
				y=price_series.values,
				name=f"{code} 价格",
				mode="lines",
				line=dict(color=color),
				opacity=1.0,
			),
			secondary_y=False,
		)
		fig.add_trace(
			go.Scatter(
				x=price_series.index,
				y=hw_indicator.hw,
				name=f"{code} HW平滑",
				mode="lines",
				line=dict(color=color, dash="dot"),
				opacity=0.75,
			),
			secondary_y=False,
		)
		fig.add_trace(
			go.Scatter(
				x=price_series.index,
				y=hwdp_indicator.hwdp * 100,
				name=f"{code} HDP%",
				mode="lines",
				line=dict(color=color),
				opacity=0.4,
			),
			secondary_y=True,
		)
		any_added = True

	if not any_added:
		return None
	fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	fig.update_yaxes(title_text="价格", secondary_y=False)
	fig.update_yaxes(title_text="HDP%", secondary_y=True, range=[-100, 700])
	return fig


def main() -> None:
	st.session_state.setdefault("hw_auto_refresh", False)
	_auto_refresh()

	# Compact refresh toggle to the right of title
	title_col, control_col = st.columns([6, 1])
	with title_col:
		st.title("Holtwinters资产监控面板")
	with control_col:
		button_label = "停止刷新" if st.session_state.get("hw_auto_refresh", False) else "开始刷新"
		if st.button(button_label, use_container_width=True):
			st.session_state["hw_auto_refresh"] = not st.session_state.get("hw_auto_refresh", False)
		st.caption(f"更新：{datetime.now().strftime('%H:%M:%S')}")

	st.write("可直接管理基金/股票列表，自动爬取最新数据并缓存到 data/fund 与 data/stock，或继续使用 PKL 上传方式。")

	_ensure_cache_dirs()

	meta_fund_names = _load_fund_meta()
	meta_stock_names = _load_stock_meta()
	st.session_state.setdefault("fund_panel_codes", [])
	st.session_state.setdefault("stock_panel_codes", [])
	cached_params = _load_cached_params()
	asset_lists = _load_asset_lists()

	with st.expander("参数输入与缓存", expanded=False):
		st.caption("导入或新增基金参数，文件将保存到 data/fund_parameters/{code}.json，缺少的键会写为空。")
		param_file = st.file_uploader("导入基金参数文件", type=["json"], key="param-import")
		if param_file:
			payload = param_file.getvalue().decode("utf-8", errors="ignore")
			saved_codes = _save_param_json(payload)
			if saved_codes:
				st.success(f"已导入并拆分参数：{', '.join(saved_codes)}")
				cached_params = _load_cached_params()

		popover = getattr(st, "popover", None)
		if popover is None:
			param_container = st.expander("填写单只基金参数")
		else:
			param_container = popover("添加参数", use_container_width=True)
		with param_container:
			code_input = st.text_input("基金代码")
			alpha = st.number_input("alpha", value=0.0, step=0.001, format="%f")
			beta = st.number_input("beta", value=0.0, step=0.001, format="%f")
			gamma = st.number_input("gamma", value=0.0, step=0.001, format="%f")
			season_length = st.number_input("season_length", value=0, step=1, format="%d")
			mse = st.number_input("mse", value=0.0, step=0.000001, format="%f")
			data_points = st.number_input("data_points", value=0, step=1, format="%d")
			status = st.text_input("status", value="success")
			save_manual = st.button("保存该基金参数", use_container_width=True, key="save-single-param")
			if save_manual:
				entry = {
					"code": code_input.strip(),
					"status": status.strip() or None,
					"data_points": data_points if data_points else None,
					"params": {
						"alpha": alpha if alpha != 0.0 else None,
						"beta": beta if beta != 0.0 else None,
						"gamma": gamma if gamma != 0.0 else None,
						"season_length": season_length if season_length else None,
						"mse": mse if mse != 0.0 else None,
					},
				}
				payload_text = json.dumps(entry, ensure_ascii=False)
				saved_codes = _save_param_json(payload_text)
				if saved_codes:
					st.success(f"已保存参数：{', '.join(saved_codes)}")
					cached_params = _load_cached_params()

	effective_params = {**cached_params}

	st.markdown("### 基金资产")
	apply_list_clicked = False
	if asset_lists:
		list_options = {f"{item['id']}: {item['name']}": item for item in asset_lists}
		selected_label = st.selectbox("选择历史基金列表", options=list_options.keys())
		apply_list = st.button("应用该列表", use_container_width=True)
		if apply_list:
			apply_list_clicked = True
			selected = list_options[selected_label]
			st.session_state["fund_panel_codes"] = selected.get("codes", [])
			st.success(f"已应用列表：{selected_label}")
	else:
		st.info("暂无历史基金列表，可在下方导出当前列表进行保存。")

	with st.form("fund-manage-form"):
		add_choices = st.multiselect(
			"添加基金代码",
			options=sorted(meta_fund_names.keys()),
			format_func=lambda x: f"{x} - {meta_fund_names.get(x, '')}".strip(" -"),
		)
		remove_choices = st.multiselect("移除基金", options=st.session_state["fund_panel_codes"])
		submit_fund = st.form_submit_button("更新基金列表", use_container_width=True)
	if submit_fund:
		added, invalid = _normalize_codes(add_choices, validate_fund_code)
		codes = set(st.session_state["fund_panel_codes"])
		codes.update(added)
		codes.difference_update(remove_choices)
		st.session_state["fund_panel_codes"] = sorted(codes)
		if invalid:
			st.warning(f"以下基金代码未通过校验，已忽略：{', '.join(invalid)}")
		st.success(f"基金列表已更新（{len(codes)}）")

	current_funds = st.session_state["fund_panel_codes"]
	if current_funds:
		needs_fund_refresh = False
		if st.session_state.get("hw_auto_refresh", False):
			needs_fund_refresh = True
		if submit_fund or apply_list_clicked:
			needs_fund_refresh = True
		cached_frames: Dict[str, pd.DataFrame] = st.session_state.get("fund_frames_cache", {})
		if not cached_frames:
			needs_fund_refresh = True

		if needs_fund_refresh:
			with st.spinner("正在更新基金数据与缓存…"):
				fund_frames: Dict[str, pd.DataFrame] = {}
				for code in current_funds:
					frame = _refresh_fund_history(code)
					if frame is None or frame.empty:
						st.warning(f"基金 {code} 暂无可用数据。")
						continue
					fund_frames[code] = frame
			if fund_frames:
				st.session_state["fund_frames_cache"] = fund_frames
			else:
				st.info("未获取到基金数据，请检查代码或网络。")

		fund_frames = st.session_state.get("fund_frames_cache", {})
		if fund_frames:
			fund_params = effective_params
			fund_panel = _build_fund_panel(fund_frames, {}, meta_fund_names, fund_params)
			_render_fund_panel(fund_panel)
		else:
			st.info("暂无基金数据可展示，请刷新或检查网络。")
	else:
		st.info("通过上方表单添加基金后将自动抓取数据并展示。")
		st.session_state.pop("fund_frames_cache", None)

	st.markdown("#### 基金曲线与 HDP")
	cache_frames = st.session_state.get("fund_frames_cache", {})
	if not cache_frames:
		st.info("请先加载至少一只基金数据。")
	else:
		plot_codes = st.multiselect("选择要绘图的基金", options=current_funds, default=current_funds[:1] if current_funds else [])
		fig = _make_hw_fig(plot_codes, cache_frames, effective_params)
		if fig is not None:
			st.plotly_chart(fig, use_container_width=True)
		else:
			st.info("未生成可绘制的数据，请检查参数或选择。")

	export_name = st.text_input("导出当前基金列表名称", value=datetime.now().strftime("List_%Y%m%d_%H%M"))
	if st.button("导出当前基金列表到 asset_lists.json", use_container_width=True):
		saved_id = _save_asset_list(export_name, st.session_state.get("fund_panel_codes", []))
		if saved_id:
			st.success(f"已保存列表 #{saved_id}")
			asset_lists = _load_asset_lists()

	st.markdown("### 股票资产")
	with st.form("stock-manage-form"):
		stock_choices = st.multiselect(
			"添加股票代码",
			options=sorted(meta_stock_names.keys()),
			format_func=lambda x: f"{x} - {meta_stock_names.get(x, '')}".strip(" -"),
		)
		manual_stock = st.text_input("手动输入股票代码（逗号分隔）")
		remove_stock = st.multiselect("移除股票", options=st.session_state["stock_panel_codes"])
		submit_stock = st.form_submit_button("更新股票列表", use_container_width=True)
	if submit_stock:
		added, invalid = _normalize_codes(stock_choices + [manual_stock], validate_stock_code)
		codes = set(st.session_state["stock_panel_codes"])
		codes.update(added)
		codes.difference_update(remove_stock)
		st.session_state["stock_panel_codes"] = sorted(codes)
		if invalid:
			st.warning(f"以下股票代码未通过校验，已忽略：{', '.join(invalid)}")
		st.success(f"股票列表已更新（{len(codes)}）")

	current_stocks = st.session_state["stock_panel_codes"]
	if current_stocks:
		with st.spinner("正在更新股票数据与缓存…"):
			stock_frames: Dict[str, pd.DataFrame] = {}
			for code in current_stocks:
				frame = _refresh_stock_history(code)
				if frame is None or frame.empty:
					st.warning(f"股票 {code} 暂无可用数据。")
					continue
				stock_frames[code] = frame
		if stock_frames:
			stock_panel = _build_stock_panel(stock_frames, {}, meta_stock_names)
			_render_stock_panel(stock_panel)
		else:
			st.info("未获取到股票数据，请检查代码或网络。")
	else:
		st.info("通过上方表单添加股票后将自动抓取数据并展示。")

	st.caption(f"最后刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
	main()
