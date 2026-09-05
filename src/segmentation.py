"""User segmentation — the 'users like you' layer that keeps peer comparisons fair.

KMeans over standardized user profiles groups financially similar users (a student cohort, a
family cohort, ...). The recommender uses each user's *cohort norms* — never a global average —
so personalization holds even when comparing against peers. Reports silhouette score.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS_DIR, EXPENSE_CATEGORIES, N_COHORTS, RANDOM_SEED
from src.features import PROFILE_FEATURE_COLS


class UserSegmenter:
    def __init__(self, n_cohorts: int = N_COHORTS) -> None:
        self.n_cohorts = n_cohorts
        self.scaler = StandardScaler()
        self.km = KMeans(n_clusters=n_cohorts, n_init=10, random_state=RANDOM_SEED)
        self.silhouette_: float | None = None
        self.cohort_category_norms_: pd.DataFrame | None = None

    def fit(self, profiles: pd.DataFrame) -> "UserSegmenter":
        X = self.scaler.fit_transform(profiles[PROFILE_FEATURE_COLS])
        labels = self.km.fit_predict(X)
        try:
            self.silhouette_ = float(silhouette_score(X, labels))
        except Exception:
            self.silhouette_ = float("nan")

        # Cohort norms: median category share per cohort → peer benchmark for recommendations.
        share_cols = [f"share_{c}" for c in EXPENSE_CATEGORIES]
        tmp = profiles[share_cols].copy()
        tmp["cohort"] = labels
        self.cohort_category_norms_ = tmp.groupby("cohort").median()
        self.cohort_category_norms_.columns = EXPENSE_CATEGORIES
        return self

    def predict(self, profiles: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(profiles[PROFILE_FEATURE_COLS])
        return self.km.predict(X)

    def assign(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Return profiles with a 'cohort' column appended."""
        out = profiles.copy()
        out["cohort"] = self.predict(profiles)
        return out

    def save(self, path=None):
        path = path or (ARTIFACTS_DIR / "segmenter.joblib")
        joblib.dump(dict(scaler=self.scaler, km=self.km,
                         silhouette=self.silhouette_,
                         norms=self.cohort_category_norms_), path)
        return path

    @classmethod
    def load(cls, path=None) -> "UserSegmenter":
        path = path or (ARTIFACTS_DIR / "segmenter.joblib")
        blob = joblib.load(path)
        obj = cls()
        obj.scaler, obj.km = blob["scaler"], blob["km"]
        obj.silhouette_, obj.cohort_category_norms_ = blob["silhouette"], blob["norms"]
        obj.n_cohorts = obj.km.n_clusters
        return obj
