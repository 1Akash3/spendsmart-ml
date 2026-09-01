"""Feature engineering: transactions -> monthly panel -> user profiles + forecasting frame.

Three products, consumed downstream:
  1. `build_monthly_panel`   : (user, month) × category spend matrix + income  (the analytical core)
  2. `build_user_profiles`   : one feature vector per user                     (feeds segmentation)
  3. `build_forecast_frame`  : supervised (lag → next-month) rows              (feeds forecaster)

Everything is per-user, so the learned signal is personal rather than global.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXPENSE_CATEGORIES, FORECAST_LAGS, INCOME_CATEGORY


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into a per-user, per-month panel.

    Returns a long-ish DataFrame with one row per (user_id, month) and columns:
      <each expense category>, income, total_expense
    Months with no activity for a user are filled in (zeros) so lags are well-defined.
    """
    d = df.copy()
    if "date" not in d.columns and "timestamp" in d.columns:
        d["date"] = d["timestamp"]
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()

    if "type" not in d.columns and "direction" in d.columns:
        d["type"] = d["direction"].map({"debit": "expense", "credit": "income"}).fillna("expense")

    exp = d[d["type"] == "expense"]
    wide = (exp.groupby(["user_id", "month", "category"])["amount"].sum()
                .unstack("category", fill_value=0.0))
    # ensure every expense category column exists and is ordered
    for c in EXPENSE_CATEGORIES:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide[EXPENSE_CATEGORIES]

    inc = (d[d["type"] == "income"].groupby(["user_id", "month"])["amount"].sum()
             .rename("income"))
    panel = wide.join(inc, how="left")

    panel = panel.reset_index()

    # Reindex each user to a continuous monthly grid (fills gaps with 0 spend).
    filled = []
    for uid, g in panel.groupby("user_id"):
        full = pd.date_range(g["month"].min(), g["month"].max(), freq="MS")
        g = g.set_index("month").reindex(full)
        g["user_id"] = uid
        g.index.name = "month"
        filled.append(g.reset_index())
    panel = pd.concat(filled, ignore_index=True)

    panel[EXPENSE_CATEGORIES] = panel[EXPENSE_CATEGORIES].fillna(0.0)
    panel["total_expense"] = panel[EXPENSE_CATEGORIES].sum(axis=1)
    # Income: carry forward/back within user. For users with NO income data at all (e.g. a pure
    # card-spend dataset), proxy from THEIR OWN spending (×1.1) so the savings-rate stays sane and
    # per-user — a global median would mis-scale users who spend far above or below it.
    # NB: fill WITHIN each user (transform), not `.ffill().bfill()` — the latter's bfill runs on an
    # already-ungrouped Series and leaks one user's income into the next.
    panel["income"] = panel.groupby("user_id")["income"].transform(lambda s: s.ffill().bfill())
    proxy = panel.groupby("user_id")["total_expense"].transform("mean") * 1.1
    panel["income"] = panel["income"].fillna(proxy).fillna(panel["income"].median())
    return panel


def build_user_profiles(panel: pd.DataFrame) -> pd.DataFrame:
    """One feature row per user, capturing *their* spending fingerprint.

    Features: mean share of each category, log income, savings rate, expense volatility,
    and total-spend trend slope. These are what segmentation clusters on.
    """
    rows = []
    for uid, g in panel.groupby("user_id"):
        g = g.sort_values("month")
        tot = g["total_expense"].replace(0, np.nan)
        shares = {f"share_{c}": float((g[c] / tot).mean(skipna=True) or 0.0)
                  for c in EXPENSE_CATEGORIES}
        income = float(g["income"].mean())
        mean_exp = float(g["total_expense"].mean())
        savings_rate = float(np.clip(1.0 - mean_exp / income, -1, 1)) if income > 0 else 0.0
        volatility = float(g["total_expense"].std(ddof=0) / (mean_exp + 1e-9))
        # linear trend slope of total expense over month index, normalized by mean
        y = g["total_expense"].to_numpy()
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0] / (mean_exp + 1e-9)) if len(y) > 1 else 0.0

        rows.append(dict(user_id=uid, log_income=float(np.log1p(income)),
                         savings_rate=savings_rate, volatility=volatility,
                         trend=slope, mean_expense=mean_exp, **shares))
    return pd.DataFrame(rows).set_index("user_id")


PROFILE_FEATURE_COLS = (
    ["log_income", "savings_rate", "volatility", "trend"]
    + [f"share_{c}" for c in EXPENSE_CATEGORIES]
)


def build_forecast_frame(panel: pd.DataFrame, static_features: pd.DataFrame | None = None,
                         lags: int = FORECAST_LAGS) -> pd.DataFrame:
    """Supervised frame: predict a category's month-t spend from months < t.

    Returns one row per (user, month, category) with columns:
      user_id, month, category, target, lag_1..lag_L, roll_mean_3, roll_std_3,
      tot_lag_1, moy_sin, moy_cos  [+ merged static_features on user_id]
    Rows without full lag history are dropped.
    """
    long = panel.melt(id_vars=["user_id", "month", "total_expense", "income"],
                      value_vars=EXPENSE_CATEGORIES,
                      var_name="category", value_name="amount")
    long = long.sort_values(["user_id", "category", "month"])
    grp = long.groupby(["user_id", "category"], sort=False)["amount"]

    for lag in range(1, lags + 1):
        long[f"lag_{lag}"] = grp.shift(lag)
    long["roll_mean_3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    long["roll_std_3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).std())
    long["tot_lag_1"] = (long.groupby(["user_id", "category"], sort=False)["total_expense"]
                             .shift(1))

    moy = long["month"].dt.month
    long["moy_sin"] = np.sin(2 * np.pi * moy / 12)
    long["moy_cos"] = np.cos(2 * np.pi * moy / 12)
    long["target"] = long["amount"]

    long = long.rename(columns={"income": "cur_income"})
    lag_cols = [f"lag_{l}" for l in range(1, lags + 1)]
    for lc in lag_cols:
        long[lc] = long[lc].fillna(0.0)
    long["roll_std_3"] = long["roll_std_3"].fillna(0.0)
    long["roll_mean_3"] = long["roll_mean_3"].fillna(0.0)
    long["tot_lag_1"] = long["tot_lag_1"].fillna(0.0)

    keep = (["user_id", "month", "category", "target", "cur_income"] + lag_cols
            + ["roll_mean_3", "roll_std_3", "tot_lag_1", "moy_sin", "moy_cos"])
    out = long[keep].copy()

    if static_features is not None:
        out = out.merge(static_features, on="user_id", how="left")
    return out.reset_index(drop=True)


def build_serving_frame(panel: pd.DataFrame, static_features: pd.DataFrame | None = None,
                        lags: int = FORECAST_LAGS) -> pd.DataFrame:
    """Build one row per (user, category) for the NEXT month after each user's last observation.

    Same feature columns as `build_forecast_frame` but with no `target` (it is unknown). Used at
    serving time to get a genuine next-month forecast per user per category.
    """
    rows = []
    for uid, g in panel.groupby("user_id"):
        g = g.sort_values("month")
        if len(g) < lags:
            continue
        last_month = g["month"].iloc[-1]
        next_month = (last_month + pd.offsets.MonthBegin(1))
        tail_total = g["total_expense"].to_numpy()
        income = float(g["income"].iloc[-1])
        moy = next_month.month
        for cat in EXPENSE_CATEGORIES:
            series = g[cat].to_numpy()
            row = dict(user_id=uid, month=next_month, category=cat, cur_income=income)
            for lag in range(1, lags + 1):
                row[f"lag_{lag}"] = float(series[-lag]) if len(series) >= lag else np.nan
            last3 = series[-3:]
            row["roll_mean_3"] = float(last3.mean()) if len(last3) else 0.0
            row["roll_std_3"] = float(last3.std(ddof=0)) if len(last3) > 1 else 0.0
            row["tot_lag_1"] = float(tail_total[-1])
            row["moy_sin"] = float(np.sin(2 * np.pi * moy / 12))
            row["moy_cos"] = float(np.cos(2 * np.pi * moy / 12))
            rows.append(row)

    frame = pd.DataFrame(rows)
    if static_features is not None and not frame.empty:
        frame = frame.merge(static_features, on="user_id", how="left")
    return frame.reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric predictor columns for the forecaster (everything except ids/target/month)."""
    drop = {"user_id", "month", "category", "target"}
    return [c for c in frame.columns
            if c not in drop and pd.api.types.is_numeric_dtype(frame[c])]
