"""Utility helpers for merging fund estimates with existing price data."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from dffc.data_provider.base import DataProvider


def append_estimates_to_prices(
    price_data: pd.DataFrame,
    codes: Iterable[str],
    provider: DataProvider,
    *,
    value_column: str = "estimate_value",
    timestamp_column: str = "estimate_timestamp",
    target_timezone: Optional[str] = None,
) -> pd.DataFrame:
    """Append latest estimate rows that are missing from ``price_data``.

    Parameters
    ----------
    price_data
        Historical unit value data indexed by date (``DatetimeIndex``).
    codes
        Iterable of fund codes whose estimates should be queried.
    provider
        Any provider exposing ``get_data(code) -> DataFrame`` with estimate
        information.
    value_column
        Column name in the provider output containing the estimate value.
    timestamp_column
        Column name holding the estimate timestamp. Values must be convertible
        to ``DatetimeIndex``.
    target_timezone
        Optional timezone name. When omitted the timezone of ``price_data`` is
        used. If both are ``None`` the timestamps remain naive.

    Returns
    -------
    pandas.DataFrame
        Price data with any missing estimate rows appended.

    Raises
    ------
    ValueError
        If estimate data is missing, timestamps are invalid, values are missing,
        or no estimate rows are available to append.
    KeyError
        If the provider output does not contain ``timestamp_column``.
    """

    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise TypeError("price_data must be indexed by a DatetimeIndex")

    target_tz = target_timezone if target_timezone is not None else price_data.index.tz

    codes_seq: Sequence[str] = tuple(str(code) for code in codes)
    estimate_series: dict[str, pd.Series] = {}

    for code in codes_seq:
        est_df = provider.get_data(code)
        if est_df is None or est_df.empty:
            raise ValueError(f"No estimate data returned for {code}.")

        if timestamp_column not in est_df.columns:
            raise KeyError(f"Estimate data for {code} missing column '{timestamp_column}'.")
        if value_column not in est_df.columns:
            raise KeyError(f"Estimate data for {code} missing column '{value_column}'.")

        timestamps = pd.to_datetime(est_df[timestamp_column], errors="coerce")
        valid_mask = timestamps.notna()
        if not valid_mask.any():
            raise ValueError(f"Skip {code}: missing valid estimate timestamp.")

        values = pd.to_numeric(est_df.loc[valid_mask, value_column], errors="coerce")

        if values.isna().all():
            raise ValueError(f"Skip {code}: missing estimate value.")

        ts_index = pd.DatetimeIndex(timestamps[valid_mask])
        if target_tz is not None:
            if ts_index.tz is None:
                ts_index = ts_index.tz_localize(target_tz)
            else:
                ts_index = ts_index.tz_convert(target_tz)
        else:
            if ts_index.tz is not None:
                ts_index = ts_index.tz_convert("UTC").tz_localize(None)

        estimate_timestamp = ts_index.max()
        print(f"Estimate data fetched for {code} on {estimate_timestamp}")
        ts_index = ts_index.normalize()

        ts_index = ts_index.rename(price_data.index.name)

        estimate_series[code] = pd.Series(values.to_numpy(), index=ts_index, name=code)
    
    if not estimate_series:
        raise ValueError("No estimate data added.")

    est_frame = pd.DataFrame(estimate_series)
    est_frame = est_frame.reindex(columns=[code for code in codes_seq if code in est_frame.columns])
    est_frame = est_frame[~est_frame.index.duplicated(keep="last")]
    
    missing_mask = ~est_frame.index.isin(price_data.index)
    new_rows = est_frame.loc[missing_mask]

    if new_rows.empty:
        print("No new estimate rows to append.")
        return price_data.copy()

    updated_price = pd.concat([price_data, new_rows], axis=0).sort_index()
    added_dates = ", ".join(date.strftime("%Y-%m-%d") for date in new_rows.index)
    print(f"Added estimate rows for dates: {added_dates}.")

    return updated_price
