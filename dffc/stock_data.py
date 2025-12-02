"""Stock data wrapper built on top of vectorbt's Data class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union, Hashable

import numpy as np
import pandas as pd
from vectorbt.data.base import Data
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime_ import to_tzaware_datetime
from vectorbt import _typing as tp

from dffc.data_provider.base import DataProvider
from dffc.data_provider.eastmoney_provider import EastMoneyStockProvider
from dffc._utils import validate_stock_code


class StockData(Data):
	"""VectorBT-compatible stock time-series container backed by Eastmoney."""

	_expected_keys = (
		"open",
		"high",
		"low",
		"close",
		"volume",
		"amount",
		"change",
		"change_percent",
	)

	_default_timezone = "Asia/Shanghai"

	@classmethod
	def download_symbol(
		cls,
		symbol: Hashable,
		provider: Optional[DataProvider] = None,
		start: tp.DatetimeLike = 0,
		end: tp.DatetimeLike = "now",
		**kwargs,
	) -> pd.DataFrame:
		"""Download a single stock's OHLCV frame."""

		if provider is None:
			raise ValueError("provider must be provided")

		validated_symbol = validate_stock_code(symbol)
		start_date = to_tzaware_datetime(start)
		end_date = to_tzaware_datetime(end)
		frame = provider.get_data(validated_symbol, start_date, end_date)

		for key in cls._expected_keys:
			if key not in frame.columns:
				frame[key] = np.nan
		if not isinstance(frame.index, pd.DatetimeIndex):
			frame.index = pd.to_datetime(frame.index)
		return frame

	@classmethod
	def download(
		cls,
		symbols: Union[Hashable, Sequence[Hashable]],
		provider: Optional[EastMoneyStockProvider] = None,
		**kwargs,
	) -> "StockData":
		"""Batch download stocks similar to ``FundData.download``."""

		from vectorbt.utils.config import get_func_kwargs

		provider_kwargs: Dict[str, Any] = {}
		try:
			for key in get_func_kwargs(EastMoneyStockProvider.__init__):
				if key in kwargs and key != "self":
					provider_kwargs[key] = kwargs.pop(key)
		except Exception:
			common = ["headers", "timeout", "retry_times", "rate_limit"]
			for key in common:
				if key in kwargs:
					provider_kwargs[key] = kwargs.pop(key)

		if "names" in kwargs:
			names = kwargs.pop("names")
			stock_names: Dict[str, str] = {}
			if isinstance(symbols, (str, int)):
				symbol_list = [symbols]
			else:
				symbol_list = list(symbols)
			if isinstance(names, str):
				if len(symbol_list) != 1:
					raise ValueError("Single name provided but multiple symbols given")
				stock_names[str(symbol_list[0])] = names
			elif isinstance(names, dict):
				stock_names = {str(k): v for k, v in names.items()}
			elif isinstance(names, list):
				if len(names) != len(symbol_list):
					raise ValueError("Length of names list must match symbols length")
				stock_names = {str(sym): name for sym, name in zip(symbol_list, names)}
			cls.names = stock_names

		if provider is None:
			provider = EastMoneyStockProvider(**provider_kwargs)

		data_download_kwargs: Dict[str, Any] = {}
		for key in ["missing_index", "missing_columns", "tz_localize", "tz_convert", "wrapper_kwargs"]:
			if key in kwargs:
				data_download_kwargs[key] = kwargs.pop(key)

		if "tz_localize" not in data_download_kwargs:
			data_download_kwargs["tz_localize"] = cls._default_timezone
			data_download_kwargs["tz_convert"] = cls._default_timezone

		return super(StockData, cls).download(symbols, provider=provider, **data_download_kwargs, **kwargs)

	def get(self, column: Optional[str] = None, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
		"""Return data in the same shape as FundData.get for consistency."""

		if len(self.symbols) == 1:
			if column is None:
				return self.data[self.symbols[0]]
			return self.data[self.symbols[0]][column]

		concat_data = self.concat(**kwargs)
		if column is not None:
			if isinstance(column, list):
				return {c: concat_data[c] for c in column}
			return concat_data[column]
		return concat_data

	def get_stock_info(self, symbol: str) -> Dict[str, Any]:
		"""Return basic metadata for a tracked stock."""

		if symbol not in self.symbols:
			raise ValueError(f"Stock {symbol} not found in data")
		frame = self.get("close")[symbol]
		name = ""
		if getattr(self, "names", None):
			name = self.names.get(str(symbol), "")
		return {
			"symbol": symbol,
			"name": name,
			"start_date": frame.index.min(),
			"end_date": frame.index.max(),
			"total_days": len(frame),
			"latest_close": frame.iloc[-1] if len(frame) else None,
		}

	def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.Frame:
		"""Refresh an existing symbol by reusing download kwargs."""

		download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
		download_kwargs["start"] = self.data[symbol].index[-1]
		download_kwargs["end"] = "now"
		kwargs = merge_dicts(download_kwargs, kwargs)
		return self.download_symbol(symbol, **kwargs)

	@property
	def provider_name(self) -> str:
		return "EastMoney Stock Provider"

	def __repr__(self) -> str:
		return f"<StockData: {len(self.symbols)} stocks, {self.provider_name}>"
