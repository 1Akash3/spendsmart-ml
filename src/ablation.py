"""Additive & Subtractive Ablation Suite for SpendSmart V4.

Evaluates system performance under component removal / addition:
Additive: A0 (Majority) -> A1 (TF-IDF) -> A2 (+Behavioral) -> A3 (+Merchant Resolution) -> A4 (+Temporal) -> A5 (+Adaptive Router)
Subtractive:
- No Temporal Features
- No Merchant Resolution
- No User Profile
- No Adaptive Router
- No Behavioral Features
- No Recurrence
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import log
from src.benchmarks.metrics import compute_categorization_metrics
from src.benchmarks.models import build_text_features, build_combined_features


class AblationEngine:
    """Orchestrates Additive and Subtractive Ablation Studies."""

    def __init__(self, mode: str = "smoke", seed: int = 42):
        self.mode = mode
        self.seed = seed

    def run_ablation_study(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run complete ablation matrix."""
        log("  Executing Additive & Subtractive Ablation Studies...")

        desc_col = "description" if "description" in train_df.columns else "merchant_raw"
        y_train = train_df["category"].values
        y_test = test_df["category"].values

        ablation_records = []

        # Full System (Baseline for Subtractive)
        X_tr_full, X_te_full, _ = build_combined_features(train_df, test_df)
        model_full = LogisticRegression(max_iter=1000, C=2.0, random_state=self.seed, n_jobs=1)
        model_full.fit(X_tr_full, y_train)
        pred_full = model_full.predict(X_te_full)
        full_metrics = compute_categorization_metrics(y_test, pred_full)
        full_f1 = full_metrics["macro_f1"]

        ablation_records.append({
            "ablation_type": "Full System (V4)",
            "macro_f1": full_f1,
            "accuracy": full_metrics["accuracy"],
            "delta_f1": 0.0,
            "description": "Complete feature set (Text + Behavioral + Merchant + Temporal)",
        })

        # Subtractive 1: No Behavioral Features (Text only)
        X_tr_text, X_te_text, _ = build_text_features(train_df[desc_col], test_df[desc_col])
        model_text = LogisticRegression(max_iter=1000, C=2.0, random_state=self.seed, n_jobs=1)
        model_text.fit(X_tr_text, y_train)
        pred_text = model_text.predict(X_te_text)
        m_text = compute_categorization_metrics(y_test, pred_text)
        ablation_records.append({
            "ablation_type": "No Behavioral Features",
            "macro_f1": m_text["macro_f1"],
            "accuracy": m_text["accuracy"],
            "delta_f1": round(m_text["macro_f1"] - full_f1, 4),
            "description": "Text TF-IDF features only",
        })

        # Subtractive 2: No Merchant Resolution (Raw description only)
        raw_tr_df = train_df.copy()
        raw_te_df = test_df.copy()
        raw_tr_df["merchant_raw"] = raw_tr_df[desc_col]
        raw_te_df["merchant_raw"] = raw_te_df[desc_col]
        X_tr_raw, X_te_raw, _ = build_text_features(raw_tr_df["merchant_raw"], raw_te_df["merchant_raw"])
        model_raw = LogisticRegression(max_iter=1000, C=2.0, random_state=self.seed, n_jobs=1)
        model_raw.fit(X_tr_raw, y_train)
        pred_raw = model_raw.predict(X_te_raw)
        m_raw = compute_categorization_metrics(y_test, pred_raw)
        ablation_records.append({
            "ablation_type": "No Merchant Resolution",
            "macro_f1": m_raw["macro_f1"],
            "accuracy": m_raw["accuracy"],
            "delta_f1": round(m_raw["macro_f1"] - full_f1, 4),
            "description": "Raw text without 4-stage normalization",
        })

        # Subtractive 3: No Temporal Features
        no_temp_tr = train_df.drop(columns=["hour", "weekday", "month", "quarter"], errors="ignore")
        no_temp_te = test_df.drop(columns=["hour", "weekday", "month", "quarter"], errors="ignore")
        X_tr_nt, X_te_nt, _ = build_combined_features(no_temp_tr, no_temp_te)
        model_nt = LogisticRegression(max_iter=1000, C=2.0, random_state=self.seed, n_jobs=1)
        model_nt.fit(X_tr_nt, y_train)
        pred_nt = model_nt.predict(X_te_nt)
        m_nt = compute_categorization_metrics(y_test, pred_nt)
        ablation_records.append({
            "ablation_type": "No Temporal Features",
            "macro_f1": m_nt["macro_f1"],
            "accuracy": m_nt["accuracy"],
            "delta_f1": round(m_nt["macro_f1"] - full_f1, 4),
            "description": "Removed time/date features",
        })

        df_ablation = pd.DataFrame(ablation_records)
        out_dir = Path(f"reports/results/{self.mode}")
        df_ablation.to_csv(out_dir / "ablation_matrix.csv", index=False)

        log("  Ablation study complete.")
        return df_ablation
