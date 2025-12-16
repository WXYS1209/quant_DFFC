"""Main Streamlit entry point for the DFFC toolkit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
META_DIR = APP_DIR / "_utils"

@st.cache_data(show_spinner=False)
def _meta_counts() -> dict:
    """Return lightweight stats for fund/stock metadata files."""
    stats = {"funds": 0, "stocks": 0}
    fund_meta = META_DIR / "fund_meta.csv"
    stock_meta = META_DIR / "stock_meta.csv"
    if fund_meta.exists():
        stats["funds"] = pd.read_csv(fund_meta, usecols=["code"], dtype=str).code.nunique()
    if stock_meta.exists():
        stats["stocks"] = pd.read_csv(stock_meta, usecols=["code"], dtype=str).code.nunique()
    return stats

def main() -> None:
    st.set_page_config(page_title="DFFC 主面板", layout="wide", initial_sidebar_state="collapsed")
    st.title("DFFC 研究工作台")

    meta = _meta_counts()
    col_a, col_b = st.columns(2)
    col_a.metric("基金元数据", f"{meta['funds']:,}")
    col_b.metric("股票元数据", f"{meta['stocks']:,}")

    st.markdown("---")
    st.subheader("快速导航")
    pages_dir = APP_DIR / "pages"
    n_pages = len([p for p in pages_dir.glob("*.py") if p.is_file()]) if pages_dir.exists() else 0
    nav_cols = st.columns(max(n_pages, 2))
    if nav_cols[0].button("📈 1. 监控面板", use_container_width=True):
        st.switch_page("pages/1_monitor.py")
    if nav_cols[1].button("📈 2. 优化器", use_container_width=True):
        st.switch_page("pages/2_optimizer.py")
if __name__ == "__main__":
	main()
