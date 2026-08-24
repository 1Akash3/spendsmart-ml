"""Personalized per-category spending forecaster.

One gradient-boosted regressor per expense category, predicting next-month spend. The dominant
signal is the *user's own* lagged spend (lag_1..lag_L, rolling stats), so every prediction is
conditioned on that individual's history — with cohort + profile features adding "users like you"
context. HistGradientBoosting is fast, CPU-only, and handles missing values natively.

Accuracy is proven in `evaluate.py` by benchmarking MAE/MAPE against a naive last-month baseline.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import ARTIFACTS_DIR, EXPENSE_CATEGORIES, HGBR_PARAMS, RANDOM_SEED
from features import feature_columns


class PersonalizedForecaster:
    def __init__(self, params: dict | None = None) -> None:
        self.params = {**HGBR_PARAMS, **(params or {})}
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self.feat_cols: list[str] | None = None
        self.global_mean_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, min_rows: int = 40) -> "PersonalizedForecaster":
        self.feat_cols = feature_columns(frame)
        for cat in EXPENSE_CATEGORIES:
            sub = frame[frame["category"] == cat]
            self.global_mean_[cat] = float(sub["target"].mean()) if len(sub) else 0.0
            if len(sub) < min_rows:
                self.models[cat] = None  # fall back to mean at predict time
                continue
            model = HistGradientBoostingRegressor(random_state=RANDOM_SEED, **self.params)
            model.fit(sub[self.feat_cols], sub["target"])
            self.models[cat] = model
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict `target` for every row of `frame` (row order preserved)."""
        if self.feat_cols is None:
            raise RuntimeError("Forecaster is not fitted.")
        out = np.zeros(len(frame), dtype=float)
        for cat in EXPENSE_CATEGORIES:
            mask = (frame["category"] == cat).to_numpy()
            if not mask.any():
                continue
            model = self.models.get(cat)
            if model is None:
                out[mask] = self.global_mean_.get(cat, 0.0)
            else:
                out[mask] = np.clip(model.predict(frame.loc[mask, self.feat_cols]), 0, None)
        return out

    def predict_by_user_category(self, serving_frame: pd.DataFrame) -> dict[int, dict[str, float]]:
        """Return {user_id: {category: predicted_next_month_spend}} from a serving frame."""
        preds = self.predict(serving_frame)
        result: dict = {}
        for (uid, cat), p in zip(zip(serving_frame["user_id"], serving_frame["category"]), preds):
            result.setdefault(uid, {})[cat] = float(p)   # uid kept native (str ids like 'cc_..' or ints)
        return result

    def save(self, path=None):
        path = path or (ARTIFACTS_DIR / "forecaster.joblib")
        joblib.dump(dict(models=self.models, feat_cols=self.feat_cols,
                         params=self.params, global_mean=self.global_mean_), path)
        return path

    @classmethod
    def load(cls, path=None) -> "PersonalizedForecaster":
        path = path or (ARTIFACTS_DIR / "forecaster.joblib")
        blob = joblib.load(path)
        obj = cls(params=blob["params"])
        obj.models, obj.feat_cols = blob["models"], blob["feat_cols"]
        obj.global_mean_ = blob["global_mean"]
        return obj
