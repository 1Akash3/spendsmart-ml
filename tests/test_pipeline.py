"""Fast smoke test — exercises the whole pipeline on a tiny dataset.

Run:  python -m pytest tests/ -q        (or)   python tests/test_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from categorizer import TransactionCategorizer
from evaluate import backtest_forecaster, evaluate_overspend_detector
from features import (build_forecast_frame, build_monthly_panel, build_serving_frame,
                      build_user_profiles, PROFILE_FEATURE_COLS)
from forecaster import PersonalizedForecaster
from recommender import PersonalizedRecommender, RealTimeState
from segmentation import UserSegmenter
from synth import generate_transactions
from config import EXPENSE_CATEGORIES


def _small():
    return generate_transactions(n_users=40, months=14, seed=0)


def test_synth_shape():
    df = _small()
    assert len(df) > 1000
    assert set(["user_id", "date", "amount", "category", "description", "type"]).issubset(df.columns)
    assert df["amount"].min() >= 0


def test_categorizer_learns():
    df = _small()
    Xtr, Xte, ytr, yte = train_test_split(df["description"], df["category"],
                                          test_size=0.25, random_state=0, stratify=df["category"])
    cat = TransactionCategorizer().fit(Xtr, ytr)
    m = cat.evaluate(Xte, yte)
    assert m["accuracy"] > 0.75, m["accuracy"]           # merchants are strong signal
    labels, conf = cat.predict_with_confidence(Xte[:5])
    assert len(labels) == 5 and (conf <= 1).all()


def test_panel_and_profiles():
    df = _small()
    panel = build_monthly_panel(df)
    assert set(EXPENSE_CATEGORIES).issubset(panel.columns)
    assert (panel["total_expense"] >= 0).all()
    profiles = build_user_profiles(panel)
    assert len(profiles) == df["user_id"].nunique()
    assert set(PROFILE_FEATURE_COLS).issubset(profiles.columns)


def test_segmentation():
    df = _small()
    profiles = build_user_profiles(build_monthly_panel(df))
    seg = UserSegmenter(n_cohorts=4).fit(profiles)
    assigned = seg.assign(profiles)
    assert assigned["cohort"].nunique() >= 2
    assert seg.cohort_category_norms_.shape[1] == len(EXPENSE_CATEGORIES)


def test_forecaster_beats_naive():
    # Needs a realistic number of users for the per-category models to learn (a handful of
    # users overfits); at ~120+ users the personalized forecaster reliably beats the naive
    # last-month baseline.
    df = generate_transactions(n_users=150, months=16, seed=0)
    panel = build_monthly_panel(df)
    profiles = build_user_profiles(panel)
    seg = UserSegmenter(n_cohorts=5).fit(profiles)
    static = seg.assign(profiles)[PROFILE_FEATURE_COLS + ["cohort"]].reset_index()
    res = backtest_forecaster(panel, static)
    assert "model_mae" in res
    assert res["skill_vs_naive"] > 0, res       # strictly better than last-month naive


def test_recommender_and_realtime():
    df = _small()
    panel = build_monthly_panel(df)
    profiles = build_user_profiles(panel)
    seg = UserSegmenter(n_cohorts=4).fit(profiles)
    static = seg.assign(profiles)[PROFILE_FEATURE_COLS + ["cohort"]].reset_index()
    frame = build_forecast_frame(panel, static_features=static)
    fc = PersonalizedForecaster().fit(frame)
    serving = build_serving_frame(panel, static_features=static)
    forecasts = fc.predict_by_user_category(serving)

    uid = int(profiles.index[0])
    upanel = panel[panel["user_id"] == uid]
    cohort = int(static.loc[static["user_id"] == uid, "cohort"].iloc[0])
    out = PersonalizedRecommender().recommend(
        upanel, forecasts[uid], cohort_norms_row=seg.cohort_category_norms_.loc[cohort],
        savings_goal_rate=0.3)
    assert "recommendations" in out and "summary" in out
    assert isinstance(out["recommendations"], list)

    # Real-time layer
    mean = {c: float(upanel[c].mean()) for c in EXPENSE_CATEGORIES}
    std = {c: float(upanel[c].std(ddof=0)) for c in EXPENSE_CATEGORIES}
    rt = RealTimeState(mean, std, income=float(upanel["income"].mean()))
    alerts = rt.update("food_dining", mean["food_dining"] * 3, day_of_month=5)
    assert isinstance(alerts, list)
    assert set(rt.projected_month_end().keys()) == set(EXPENSE_CATEGORIES)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll smoke tests passed.")
