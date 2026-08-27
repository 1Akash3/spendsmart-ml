# spendsmart_ml/features/temporal.py
"""Temporal feature extraction utilities.

This module provides functions to compute cyclical time encodings and various
inter‑transaction interval features for a DataFrame that follows the canonical
schema defined in ``canonical_schema.Transaction``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to pandas datetime (UTC) safely."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for hour, day‑of‑week, day‑of‑month, month.

    The original ``timestamp`` column must be present. New columns are:
    - hour_sin, hour_cos
    - dow_sin, dow_cos (day‑of‑week)
    - dom_sin, dom_cos (day‑of‑month)
    - month_sin, month_cos
    """
    df = df.copy()
    ts = _to_datetime(df["timestamp"])
    # Hour of day (0‑23)
    hour = ts.dt.hour.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # Day of week (0‑6 Mon=0)
    dow = ts.dt.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    # Day of month (1‑31)
    dom = ts.dt.day.astype(float)
    df["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31)
    df["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31)
    # Month (1‑12)
    month = ts.dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


def add_inter_transaction_features(df: pd.DataFrame, user_col: str = "user_id") -> pd.DataFrame:
    """Compute time‑gap features for each transaction per user.

    Features added:
    - `time_since_prev` (seconds since previous transaction of same user)
    - `time_since_prev_merchant` (seconds since previous transaction with same merchant)
    - `time_since_prev_category` (seconds since previous transaction of same category)
    - `transactions_past_7d`, `transactions_past_30d` (counts in rolling windows)
    The function is safe for users with a single transaction (gaps become NaN).
    """
    df = df.copy()
    df = df.sort_values([user_col, "timestamp"]).reset_index(drop=True)
    df["timestamp_dt"] = _to_datetime(df["timestamp"]).astype("int64") // 1_000_000_000  # seconds epoch
    # time since previous transaction per user
    df["prev_ts_user"] = df.groupby(user_col)["timestamp_dt"].shift(1)
    df["time_since_prev"] = df["timestamp_dt"] - df["prev_ts_user"]
    # time since previous same merchant
    if "merchant_raw" in df.columns:
        df["prev_ts_merchant"] = df.groupby([user_col, "merchant_raw"])["timestamp_dt"].shift(1)
        df["time_since_prev_merchant"] = df["timestamp_dt"] - df["prev_ts_merchant"]
    # time since previous same category
    if "category" in df.columns:
        df["prev_ts_category"] = df.groupby([user_col, "category"])["timestamp_dt"].shift(1)
        df["time_since_prev_category"] = df["timestamp_dt"] - df["prev_ts_category"]
    # Rolling transaction counts (7d and 30d)
    df.set_index(pd.to_datetime(df["timestamp_dt"], unit="s"), inplace=True)
    for win in [7, 30]:
        col_name = f"transactions_past_{win}d"
        df[col_name] = (
            df.groupby(user_col)["amount"].rolling(f"{win}D", min_periods=1).count()
        )
    df.reset_index(drop=True, inplace=True)
    # Clean up temporary columns
    df.drop(columns=["prev_ts_user", "prev_ts_merchant", "prev_ts_category"], errors="ignore", inplace=True)
    return df
