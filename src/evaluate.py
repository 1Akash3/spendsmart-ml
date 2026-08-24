"""Evaluation harness — turns 'accurate' into a number, benchmarked against naive baselines.

Three evaluations:
  • Categorizer  : accuracy + macro-F1 on a held-out split.
  • Forecaster   : MAE / MAPE vs a last-month naive baseline (skill = how much better).
  • Recommender  : precision/recall/F1 of the overspend detector on a held-out month,
                   vs the same detector driven by the naive baseline.

A model is only "proven" if it beats the baseline — that comparison is the point of this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXPENSE_CATEGORIES, OVERSPEND_Z, TEST_FRACTION
from features import build_forecast_frame, feature_columns
from forecaster import PersonalizedForecaster


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _mape(y_true, y_pred, eps=1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _wape(y_true, y_pred) -> float:
    """Weighted Absolute % Error = sum|err| / sum|actual|. Robust to $0-spend months
    (unlike MAPE, which explodes when many actuals are ~0). This is the headline % metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def _smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0,2]: mean( |err| / ((|a|+|p|)/2) ). Bounded even at zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    return float(np.mean(np.abs(y_true - y_pred)[mask] / denom[mask])) if mask.any() else float("nan")


def _time_split_months(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Global time-based split: the latest `test_fraction` of distinct months are the test set."""
    months = np.sort(frame["month"].unique())
    n_test = max(1, int(round(len(months) * test_fraction)))
    cutoff = months[-n_test]
    train = frame[frame["month"] < cutoff]
    test = frame[frame["month"] >= cutoff]
    return train, test, cutoff


def _time_split_peruser(frame: pd.DataFrame):
    """Per-user temporal split: hold out EACH user's last observed month as test, train on
    their earlier months. Robust to users with different date ranges (a global month cutoff
    breaks when datasets span different periods) and is the natural eval for personalization."""
    f = frame.sort_values(["user_id", "month"])
    last = f.groupby("user_id")["month"].transform("max")
    return f[f["month"] < last], f[f["month"] == last], "per-user-last-month"


def backtest_forecaster(panel: pd.DataFrame, static_features: pd.DataFrame) -> dict:
    """Train on past months, test on held-out recent months. Compare to a naive baseline."""
    frame = build_forecast_frame(panel, static_features=static_features)
    train, test, cutoff = _time_split_peruser(frame)
    if len(train) == 0 or len(test) == 0:
        return {"error": "insufficient history for backtest"}

    fc = PersonalizedForecaster().fit(train)
    pred = fc.predict(test)
    naive = test["lag_1"].to_numpy()          # predict "same as last month"
    roll = test["roll_mean_3"].to_numpy()     # predict "recent 3-month average"

    model_mae, naive_mae, roll_mae = _mae(test["target"], pred), _mae(test["target"], naive), _mae(test["target"], roll)
    per_cat = {}
    for c in EXPENSE_CATEGORIES:
        m = (test["category"] == c).to_numpy()
        if m.sum() == 0:
            continue
        per_cat[c] = {"model_mae": round(_mae(test["target"][m], pred[m]), 2),
                      "naive_mae": round(_mae(test["target"][m], naive[m]), 2),
                      "n": int(m.sum())}

    return {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "cutoff_month": str(cutoff),
        "model_mae": round(model_mae, 2),
        "naive_mae": round(naive_mae, 2),
        "rolling_mean_mae": round(roll_mae, 2),
        # WAPE is the headline % metric — robust to $0-spend months (MAPE is not, kept for reference).
        "model_wape": round(_wape(test["target"], pred), 4),
        "naive_wape": round(_wape(test["target"], naive), 4),
        "model_smape": round(_smape(test["target"], pred), 4),
        "model_mape_ref": round(_mape(test["target"], pred), 4),
        "skill_vs_naive": round(1 - model_mae / (naive_mae + 1e-9), 4),   # >0 → better than naive
        "per_category_mae": per_cat,
    }


def evaluate_overspend_detector(panel: pd.DataFrame, static_features: pd.DataFrame) -> dict:
    """Recommendation-quality proxy: can we flag next month's category overspends?

    Ground truth for a (user, category, test-month): actual spend exceeds the user's trailing
    baseline mean + OVERSPEND_Z·std. Predicted: the forecaster's prediction exceeds the same
    threshold. Compared against the naive (last-month) predictor.
    """
    frame = build_forecast_frame(panel, static_features=static_features)
    train, test, _ = _time_split_peruser(frame)
    if len(train) == 0 or len(test) == 0:
        return {"error": "insufficient history"}

    fc = PersonalizedForecaster().fit(train)
    pred = fc.predict(test)
    test = test.copy()
    test["pred"] = pred

    # Threshold from each row's own trailing stats (roll_mean_3 as baseline, roll_std_3 as spread).
    base = test["roll_mean_3"].to_numpy()
    spread = test["roll_std_3"].to_numpy()
    thresh = base + OVERSPEND_Z * spread

    y_true = (test["target"].to_numpy() > thresh) & (spread > 0)
    y_model = (test["pred"].to_numpy() > thresh) & (spread > 0)
    y_naive = (test["lag_1"].to_numpy() > thresh) & (spread > 0)

    def prf(y_true, y_hat):
        tp = int(np.sum(y_true & y_hat))
        fp = int(np.sum(~y_true & y_hat))
        fn = int(np.sum(y_true & ~y_hat))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn}

    return {
        "n_test_rows": int(len(test)),
        "positive_rate": round(float(y_true.mean()), 4),
        "model": prf(y_true, y_model),
        "naive_baseline": prf(y_true, y_naive),
    }
