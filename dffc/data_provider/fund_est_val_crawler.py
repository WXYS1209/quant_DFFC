"""Lightweight data provider for Eastmoney fund `fundgz` estimates.

This module adapts the thin `fundgz` endpoint to the project-wide
``DataProvider`` contract so that callers can interact with it through the
same ``get_data`` interface used elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import time

import requests
import pandas as pd

from .base import BS4DataProvider, DataProviderConfig
from .._utils import DataFetchError, safe_float_convert, validate_fund_code

class FundEstimateProvider(BS4DataProvider):
	"""Provide the latest Eastmoney fund NAV estimate via `fundgz`."""

	def __init__(self, config: Optional[DataProviderConfig] = None):
		if config is None:
			config = DataProviderConfig(
				timeout=15,
				retry_count=3,
				rate_limit=0.3,
				headers={
					"User-Agent": (
						"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
						"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
					),
					"Referer": "https://fund.eastmoney.com/",
				},
				base_url="https://fundgz.1234567.com.cn/js/{fundcode}.js",
			)
		super().__init__(config)

	def fetch_raw_data(self, code: str, start_date=None, end_date=None) -> List[Dict[str, Any]]:  # type: ignore[override]
		validated_code = validate_fund_code(code)
		url = self.config.base_url.format(fundcode=validated_code)
		payload = self._make_request(url, params={"rt": int(time.time() * 1000)})
		json_payload = self._parse_jsonp(payload.text)
		
		if json_payload is None:
			raise DataFetchError(f"Unexpected JSONP payload for fund {validated_code}")
		
		return [json_payload]

	def parse_data(self, raw_data: List[Dict[str, Any]]):
		df = pd.DataFrame(raw_data)
		df.rename(
            columns={
                'fundcode': 'code',
                'jzrq': 'date',
                'dwjz': 'unit_value',
                'gsz': 'estimate_value', 
                'gszzl': 'estimate_change_pct', 
                'gztime': 'estimate_timestamp'
            }, 
            inplace=True
        )
		df[['unit_value', 'estimate_value', 'estimate_change_pct']] = df[['unit_value', 'estimate_value', 'estimate_change_pct']].map(safe_float_convert)
		df['estimate_change_pct'] = df['estimate_change_pct'] / 100.0  # Convert percentage to decimal
		df['date'] = pd.to_datetime(df['date'])
		df['estimate_timestamp'] = pd.to_datetime(df['estimate_timestamp'])
		# df.set_index('estimate_timestamp', inplace=True)
		
		return df

	@staticmethod
	def _parse_jsonp(text: str) -> Optional[Dict[str, Any]]:
		stripped = text.strip()
		if stripped.startswith("jsonpgz(") and stripped.endswith(");"):
			stripped = stripped[len("jsonpgz("):-2]
		try:
			return json.loads(stripped)
		except json.JSONDecodeError:
			return None
