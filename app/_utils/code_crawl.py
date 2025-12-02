"""Minimal helpers for downloading fund and stock catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import tushare as ts

def fetch_fund_catalog(timeout: float = 30.0) -> pd.DataFrame:
    """Return the full fund catalog as a DataFrame."""

    fund_code_url = "https://fund.eastmoney.com/js/fundcode_search.js"
    response = requests.get(fund_code_url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Unable to parse fund catalog payload")
    json_text = text[start : end + 1]
    data = json.loads(json_text)

    rows: List[dict] = []
    for item in data:
        if len(item) < 5:
            continue
        rows.append({
            "code": item[0],
            "abbr": item[1],
            "name": item[2],
            "type": item[3],
            "pinyin": item[4],
        })

    if not rows:
        raise ValueError("Fund catalog is empty")

    return pd.DataFrame(rows)


def fetch_stock_catalog(
    tushare_token: Optional[str] = None,
    tushare_exchange: str = "",
    tushare_list_status: str = "L",
) -> pd.DataFrame:
    """Return the A-share stock catalog via the Tushare SDK."""
    ts.set_token(tushare_token)
    pro = ts.pro_api()
    request_kwargs = {
        "exchange": tushare_exchange,
        "list_status": tushare_list_status,
    }
    df = pro.stock_basic(**request_kwargs)
    if df.empty:
        raise ValueError("Tushare returned an empty result")
    if "symbol" in df.columns:
        df = df.rename(columns={"symbol": "code"})
    return df


def update_fund_meta(timeout: float = 30.0) -> None:
    """Fetch fund catalog and persist it to fund_meta.csv next to this file."""

    df = fetch_fund_catalog(timeout=timeout)
    target_path = Path(__file__).with_name("fund_meta.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return None


def update_stock_meta(
    tushare_token: Optional[str] = None,
    tushare_exchange: str = "",
    tushare_list_status: str = "L",
) -> None:
    """Fetch stock catalog and persist it to stock_meta.csv next to this file."""

    df = fetch_stock_catalog(
        tushare_token=tushare_token,
        tushare_exchange=tushare_exchange,
        tushare_list_status=tushare_list_status,
    )
    target_path = Path(__file__).with_name("stock_meta.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return None

def main():
    update_fund_meta()
    update_stock_meta(
        tushare_token="8684e1cef850f509325fbea62e890448bb11c265f4ad9e5af50d9daf"
    )

if __name__ == "__main__":
    main()