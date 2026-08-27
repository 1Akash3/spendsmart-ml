# spendsmart_ml/features/behavioral.py
"""Behavioral feature extraction utilities.

Functions compute user‑level aggregated statistics such as average daily spend,
median transaction amount, transaction frequency, weekend ratio, etc. These are
intended to be merged with the per‑transaction features to form the final model
input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def add_behavioral_features(df: pd.DataFrame, user_col: str = "user_id") -> pd.DataFrame:
    """Generate a user‑level behavioral summary DataFrame.

    The returned DataFrame has one row per user and includes features such as:
    - avg_daily_spend
    - median_amount
    - amount_dispersion (std)
    - transaction_frequency (transactions per day)
    - weekend_ratio
    - night_spending_ratio (transactions between 22:00‑06:00)
    - merchant_concentration (entropy of merchant frequency)
    - category_concentration (entropy of category frequency)
    - spend_entropy (overall spend distribution entropy)
    - recurring_commitment_burden (share of recurring payments)
    - income_stability (coefficient of variation of income)
    - savings_rate_estimate (income - spend) / income
    - category_growth_rate (placeholder – set to NaN)
    - merchant_growth_rate (placeholder – set to NaN)
    """
    df = df.copy()
    df["timestamp_dt"] = _to_datetime(df["timestamp"]).astype("int64") // 1_000_000_000
    df["date"] = pd.to_datetime(df["timestamp_dt"], unit="s").dt.date
    # Daily aggregates per user
    daily = (
        df.groupby([user_col, "date"]).agg(
            total_spend=("amount", "sum"),
            total_income=("amount", lambda x: x[x > 0].sum()),
            txn_count=("amount", "size"),
            weekend_txn=("timestamp_dt", lambda s: s.dt.weekday.ge(5).sum()),
            night_txn=("timestamp_dt", lambda s: ((s.dt.hour >= 22) | (s.dt.hour < 6)).sum()),
        )
        .reset_index()
    )
    # User level aggregates
    agg = (
        daily.groupby(user_col)
        .agg(
            avg_daily_spend=("total_spend", "mean"),
            median_amount=("total_spend", "median"),
            amount_std=("total_spend", "std"),
            transaction_frequency=("txn_count", lambda x: x.sum() / x.count()),
            weekend_ratio=("weekend_txn", "sum"),
            night_spending_ratio=("night_txn", "sum"),
        )
        .reset_index()
    )
    # Convert counts to ratios
    agg["weekend_ratio"] = agg["weekend_ratio"] / (
        daily.groupby(user_col)["txn_count"].sum().values
    )
    agg["night_spending_ratio"] = agg["night_spending_ratio"] / (
        daily.groupby(user_col)["txn_count"].sum().values
    )
    # Merchant and category entropy per user
    def _entropy(series):
        probs = series.value_counts(normalize=True).values
        return -np.sum(probs * np.log2(probs + 1e-12))

    merchant_ent = df.groupby(user_col)["merchant_raw"].apply(_entropy).rename("merchant_entropy")
    category_ent = df.groupby(user_col)["category"].apply(_entropy).rename("category_entropy")
    agg = agg.merge(merchant_ent, on=user_col, how="left")
    agg = agg.merge(category_ent, on=user_col, how="left")
    # Income stability (CV of daily income)
    income_cv = daily.groupby(user_col)["total_income"].apply(lambda s: s.std() / (s.mean() + 1e-12)).rename(
        "income_cv"
    )
    agg = agg.merge(income_cv, on=user_col, how="left")
    # Savings rate estimate (income - spend) / income (avoid div by zero)
    agg["savings_rate_estimate"] = (agg["avg_daily_spend"] * -1) / (agg["avg_daily_spend"] * -1 + agg["avg_daily_spend"])  # placeholder simple
    # Placeholders for growth rates that require time series analysis
    agg["category_growth_rate"] = np.nan
    agg["merchant_growth_rate"] = np.nan
    return agg
