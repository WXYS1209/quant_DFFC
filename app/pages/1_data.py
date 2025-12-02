"""Streamlit page for downloading fund/stock data."""

from __future__ import annotations

import io
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from dffc.data_provider.eastmoney_provider import (
	EastMoneyFundProvider,
	EastMoneyStockProvider,
)
from dffc.fund_data import FundData


APP_DIR = Path(__file__).resolve().parents[1]
META_DIR = APP_DIR / "_utils"

st.set_page_config(
	page_title="数据下载与存储",
	layout="wide",
	initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def _load_meta(filename: str) -> pd.DataFrame:
	"""Load metadata CSV once and cache it for selector use."""
	path = META_DIR / filename
	if not path.exists():
		return pd.DataFrame()
	df = pd.read_csv(path, dtype=str).fillna("")
	if "code" in df.columns:
		df = df.drop_duplicates(subset="code")
	return df


def _prepare_options(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
	if df.empty:
		return [], {}
	if "name" in df.columns:
		display_values = df["name"].fillna("")
	elif "abbr" in df.columns:
		display_values = df["abbr"].fillna("")
	else:
		display_values = pd.Series(["" for _ in range(len(df))])
	label_map: Dict[str, str] = {}
	for code, label in zip(df["code"], display_values):
		label_map[str(code)] = f"{code} - {label}".strip(" -")
	return list(label_map.keys()), label_map


def _to_datetime(value: Optional[date]) -> Optional[datetime]:
	if value is None:
		return None
	return datetime.combine(value, datetime.min.time())


def _download_fund_data(
	codes: List[str], start: Optional[datetime], end: Optional[datetime]
) -> Tuple[FundData, Dict[str, pd.DataFrame]]:
	provider = EastMoneyFundProvider()
	fund_data = FundData.download(
		codes,
		provider=provider,
		start=start or 0,
		end=end or "now",
	)
	frames: Dict[str, pd.DataFrame] = {}
	for code in fund_data.symbols:
		df = fund_data.data[code].copy()
		df.index = pd.to_datetime(df.index)
		df.sort_index(inplace=True)
		frames[str(code)] = df
	return fund_data, frames


def _download_stock_data(
	codes: List[str], start: Optional[datetime], end: Optional[datetime]
) -> Dict[str, pd.DataFrame]:
	provider = EastMoneyStockProvider()
	frames: Dict[str, pd.DataFrame] = {}
	for code in codes:
		df = provider.get_data(code, start_date=start, end_date=end)
		if df is None or df.empty:
			continue
		df.index = pd.to_datetime(df.index)
		df.sort_index(inplace=True)
		frames[str(code)] = df
	return frames


def _build_csv_archive(data_map: Dict[str, pd.DataFrame]) -> io.BytesIO:
	buffer = io.BytesIO()
	with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
		for code, df in data_map.items():
			if df.empty:
				continue
			payload = df.to_csv().encode("utf-8")
			zf.writestr(f"{code}.csv", payload)
	buffer.seek(0)
	return buffer


def _build_funddata_pickle(fund_data: FundData) -> io.BytesIO:
	with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
		fund_data.save(tmp.name)
		tmp.seek(0)
		data = tmp.read()
	return io.BytesIO(data)


def _update_preview_cache(data_map: Dict[str, pd.DataFrame]) -> None:
	for code, df in data_map.items():
		if df is not None and not df.empty:
			st.session_state["last_preview"] = {
				"code": code,
				"data": df.tail(20).copy(),
			}
			return
	st.session_state.pop("last_preview", None)


def _cache_csv_payload(buffer: io.BytesIO, file_name: str, asset_type: str, count: int) -> None:
	st.session_state["csv_download"] = {
		"data": buffer.getvalue(),
		"file_name": file_name,
		"asset_type": asset_type,
		"count": count,
	}
	st.session_state["export_basename"] = Path(file_name).stem


def _cache_pkl_payload(buffer: io.BytesIO, file_name: str) -> None:
	st.session_state["pkl_download"] = {
		"data": buffer.getvalue(),
		"file_name": file_name,
	}
	st.session_state.setdefault("export_basename", Path(file_name).stem)



def _render_cached_results() -> None:
	csv_payload = st.session_state.get("csv_download")
	if not csv_payload:
		return
	asset_type = csv_payload["asset_type"]
	count = csv_payload["count"]
	st.info(f"{asset_type} 数据已缓存，共 {count} 份。")
	preview = st.session_state.get("last_preview")
	if preview:
		st.caption(f"数据预览（{preview['code']}）")
		st.dataframe(preview["data"])

	default_base = st.session_state.get("export_basename") or Path(csv_payload["file_name"]).stem
	st.session_state.setdefault("export_basename", default_base)
	base_input = st.text_input("文件名（无需扩展名）", key="export_basename")
	base_name = (base_input or default_base).strip() or default_base

	csv_col, pkl_col = st.columns(2)
	with csv_col:
		st.download_button(
			"下载 CSV 压缩包",
			data=csv_payload["data"],
			file_name=f"{base_name}.zip",
			mime="application/zip",
			key="cached_csv_button",
		)

	pkl_payload = st.session_state.get("pkl_download")
	with pkl_col:
		if pkl_payload:
			st.download_button(
				"下载 FundData PKL",
				data=pkl_payload["data"],
				file_name=f"{base_name}.pkl",
				mime="application/octet-stream",
				key="cached_pkl_button",
			)
		else:
			st.button("无 PKL 数据", disabled=True, use_container_width=True)


def _clear_cached_outputs() -> None:
	for key in ("last_preview", "csv_download", "pkl_download", "export_basename"):
		st.session_state.pop(key, None)


def main() -> None:
	st.title("数据下载与存储")
	if st.button("⬅️ 返回主面板", use_container_width=True):
		st.switch_page("dffc_app.py")
	st.write("选择基金或股票，配置日期范围，即可下载原始净值数据。")

	fund_meta = _load_meta("fund_meta.csv")
	stock_meta = _load_meta("stock_meta.csv")
	fund_options, fund_labels = _prepare_options(fund_meta)
	stock_options, stock_labels = _prepare_options(stock_meta)

	asset_type = st.radio(
		"资产类型",
		options=("基金", "股票"),
		horizontal=True,
	)

	if asset_type == "基金":
		options = fund_options
		labels = fund_labels
		help_text = "从基金元数据中选择代码，支持多选。"
	else:
		options = stock_options
		labels = stock_labels
		help_text = "从股票元数据中选择代码，支持多选。"
		if not options:
			st.info("暂未加载股票元数据，可直接输入代码。")

	selected_codes: List[str] = []
	if options:
		selected_codes = st.multiselect(
			"基金/股票列表",
			options=options,
			format_func=lambda code: labels.get(code, code),
			help=help_text,
		)
	else:
		manual = st.text_input("手动输入代码（逗号分隔）")
		selected_codes = [code.strip() for code in manual.split(",") if code.strip()]

	if not selected_codes:
		st.info("请先选择或输入至少一个代码。")

	specify_range = st.checkbox("限定日期范围")
	start_date: Optional[date] = None
	end_date: Optional[date] = None
	if specify_range:
		start_date = st.date_input("开始日期", value=None)
		end_date = st.date_input("结束日期", value=None)
		if end_date and start_date and end_date < start_date:
			st.error("结束日期不能早于开始日期。")


	fetch_clicked = st.button("获取数据", type="primary", use_container_width=True)
	if fetch_clicked:
		_clear_cached_outputs()

	_render_cached_results()

	if not fetch_clicked:
		return

	codes = selected_codes
	if not codes:
		st.warning("请选择有效的代码。")
		return
	if specify_range and start_date and end_date and end_date < start_date:
		st.warning("请修正日期范围。")
		return

	start_dt = _to_datetime(start_date)
	end_dt = _to_datetime(end_date)

	try:
		with st.spinner("正在获取数据…"):
			fund_data_obj = None
			if asset_type == "基金":
				fund_data_obj, data_map = _download_fund_data(codes, start_dt, end_dt)
			else:
				data_map = _download_stock_data(codes, start_dt, end_dt)

		if not data_map:
			st.warning("未获取到任何数据，请确认代码或日期范围。")
			return

		csv_archive = _build_csv_archive(data_map)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		csv_name = f"{asset_type}_data_{timestamp}.zip"
		_update_preview_cache(data_map)
		_cache_csv_payload(csv_archive, csv_name, asset_type, len(data_map))
		if asset_type == "基金" and fund_data_obj is not None:
			fund_buffer = _build_funddata_pickle(fund_data_obj)
			pkl_name = f"funddata_{timestamp}.pkl"
			_cache_pkl_payload(fund_buffer, pkl_name)
		else:
			st.session_state.pop("pkl_download", None)
		_render_cached_results()
	except Exception as exc:
		st.error(f"数据下载失败: {exc}")


if __name__ == "__main__":
	main()
