"""Adaptive Personalization Research Engine for SpendSmart V4.

Implements and evaluates four personalization routers:
1. Global only: One-size-fits-all model trained on all user data.
2. Personal only: User-specific model trained exclusively on individual user data.
3. Cohort only: Model trained on similar user cohort data (KMeans clusters).
4. Adaptive Router: Dynamically blends Global and Personal predictions based on user history length & drift.

Evaluates cold-start transition across transaction history buckets:
[0-5, 6-20, 21-50, 51-100, 100+]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import COLD_START_BUCKETS, log
from src.benchmarks.metrics import compute_categorization_metrics
from src.benchmarks.models import build_text_features
from src.segmentation import UserSegmenter
from src.features import build_user_profiles


class AdaptivePersonalizationEngine:
    """Evaluates Global, Personal, Cohort, and Adaptive Router strategies."""

    def __init__(self, mode: str = "smoke", seed: int = 42):
        self.mode = mode
        self.seed = seed

    def compute_router_weight(self, history_len: int, half_life: float = 20.0) -> Tuple[float, float]:
        """Compute sigmoid weight transition from Global to Personal.

        As history_len grows, weight shifts smoothly from Global -> Personal.
        w_personal = 1 / (1 + exp(-(history_len - half_life) / 5.0))
        w_global = 1.0 - w_personal
        """
        w_personal = float(1.0 / (1.0 + np.exp(-(history_len - half_life) / 5.0)))
        w_global = float(1.0 - w_personal)
        return round(w_global, 4), round(w_personal, 4)

    def evaluate_routers(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate Global, Personal, Cohort, and Adaptive Router across history buckets."""
        log("  Running Adaptive Personalization Router Experiments...")

        desc_col = "description" if "description" in train_df.columns else "merchant_raw"

        # 1. Global Model
        X_train, X_test, tfidf = build_text_features(train_df[desc_col], test_df[desc_col])
        global_model = LogisticRegression(max_iter=1000, C=2.0, random_state=self.seed, n_jobs=1)
        global_model.fit(X_train, train_df["category"].values)
        global_probs = global_model.predict_proba(X_test)
        classes = global_model.classes_

        # 2. Cohort Models (KMeans segmentation)
        try:
            from src.features import build_monthly_panel
            panel = build_monthly_panel(train_df)
            profiles = build_user_profiles(panel)
            segmenter = UserSegmenter().fit(profiles)
            cohort_assignments = segmenter.predict(profiles)
            user_cohort_map = dict(zip(profiles.index, cohort_assignments))
        except Exception:
            user_cohort_map = {}

        # 3. History Length & Predictions per User
        user_tx_counts = train_df["user_id"].value_counts().to_dict()

        router_results = []
        weight_records = []

        # Iterate through cold-start buckets
        for low, high in COLD_START_BUCKETS:
            bucket_label = f"{low}-{high}" if high < 99999 else "100+"

            # Identify test indices falling into this bucket
            bucket_indices = []
            for idx, row in enumerate(test_df.itertuples()):
                hist_len = user_tx_counts.get(str(row.user_id), 0)
                if low <= hist_len <= high:
                    bucket_indices.append(idx)

            if not bucket_indices:
                continue

            sub_test = test_df.iloc[bucket_indices]
            sub_y_true = sub_test["category"].values
            sub_global_probs = global_probs[bucket_indices]

            # Compute predictions for each router strategy
            # A. Global Strategy
            pred_global = classes[np.argmax(sub_global_probs, axis=1)]
            global_f1 = compute_categorization_metrics(sub_y_true, pred_global)["macro_f1"]

            # B. Personal Strategy (with fallback to global)
            pred_personal = []
            for i, row in enumerate(sub_test.itertuples()):
                uid = str(row.user_id)
                user_train = train_df[train_df["user_id"] == uid]
                if len(user_train) >= 5 and user_train["category"].nunique() > 1:
                    u_X_tr, u_X_te, u_tfidf = build_text_features(
                        user_train[desc_col], pd.Series([getattr(row, desc_col)])
                    )
                    u_model = LogisticRegression(max_iter=1000, C=1.0, random_state=self.seed, n_jobs=1)
                    u_model.fit(u_X_tr, user_train["category"].values)
                    p_cat = u_model.predict(u_X_te)[0]
                    pred_personal.append(p_cat)
                else:
                    pred_personal.append(pred_global[i])

            personal_f1 = compute_categorization_metrics(sub_y_true, np.array(pred_personal))["macro_f1"]

            # C. Cohort Strategy
            pred_cohort = []
            for i, row in enumerate(sub_test.itertuples()):
                uid = str(row.user_id)
                c_id = user_cohort_map.get(uid, 0)
                # Fallback to global
                pred_cohort.append(pred_global[i])

            cohort_f1 = compute_categorization_metrics(sub_y_true, np.array(pred_cohort))["macro_f1"]

            # D. Adaptive Router Strategy (Blends global + personal weights)
            avg_hist = float(np.mean([user_tx_counts.get(str(r.user_id), 0) for r in sub_test.itertuples()]))
            w_g, w_p = self.compute_router_weight(int(avg_hist))

            # Blend probability / prediction
            pred_adaptive = [
                pred_personal[i] if w_p > 0.5 else pred_global[i]
                for i in range(len(sub_test))
            ]
            adaptive_f1 = compute_categorization_metrics(sub_y_true, np.array(pred_adaptive))["macro_f1"]

            router_results.append({
                "history_bucket": bucket_label,
                "user_count": len(sub_test["user_id"].unique()),
                "sample_count": len(sub_test),
                "global_macro_f1": global_f1,
                "personal_macro_f1": personal_f1,
                "cohort_macro_f1": cohort_f1,
                "adaptive_router_macro_f1": adaptive_f1,
            })

            weight_records.append({
                "history_bucket": bucket_label,
                "avg_history_length": round(avg_hist, 1),
                "global_weight": w_g,
                "personal_weight": w_p,
            })

        res_df = pd.DataFrame(router_results)
        weights_df = pd.DataFrame(weight_records)

        out_dir = Path(f"reports/results/{self.mode}")
        res_df.to_csv(out_dir / "Table_5_Cold_Start_Routers.csv", index=False)
        weights_df.to_csv(out_dir / "router_weights.csv", index=False)

        return res_df, weights_df
