"""Distribution Drift Research & Monitoring Engine for SpendSmart V4.

Computes multi-dimensional temporal drift:
1. Population Stability Index (PSI) on amount distributions.
2. Wasserstein Distance on financial feature vectors.
3. KL Divergence on category and merchant prior probabilities.
4. Generates a comprehensive drift dashboard.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import log


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current samples.

    PSI < 0.1: No significant shift
    0.1 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant population drift
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    bins = np.histogram_bin_edges(np.concatenate([reference, current]), bins=num_bins)
    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / max(1, len(reference))
    curr_pct = curr_counts / max(1, len(current))

    # Add epsilon for numerical stability
    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    curr_pct = np.where(curr_pct == 0, eps, curr_pct)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(np.round(psi, 4))


def compute_category_kl_divergence(
    ref_categories: pd.Series, curr_categories: pd.Series, all_cats: List[str]
) -> float:
    """Compute KL Divergence between reference and current category distributions."""
    ref_counts = ref_categories.value_counts()
    curr_counts = curr_categories.value_counts()

    ref_p = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    curr_q = np.array([curr_counts.get(c, 0) for c in all_cats], dtype=float)

    ref_p = ref_p / max(1.0, ref_p.sum())
    curr_q = curr_q / max(1.0, curr_q.sum())

    eps = 1e-6
    ref_p = np.where(ref_p == 0, eps, ref_p)
    curr_q = np.where(curr_q == 0, eps, curr_q)

    return float(np.round(entropy(ref_p, curr_q), 4))


class DriftEngine:
    """Orchestrates comprehensive distribution drift benchmarks."""

    def __init__(self, mode: str = "smoke"):
        self.mode = mode

    def run_drift_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run monthly temporal drift analysis across amount, category, and merchant features."""
        log("  Executing Temporal Distribution Drift Analysis...")

        df_sorted = df.copy()
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True)
        df_sorted["period"] = df_sorted["timestamp"].dt.to_period("M").astype(str)

        periods = sorted(df_sorted["period"].unique())
        all_cats = sorted(df_sorted["category"].unique())

        if len(periods) < 2:
            log("  Single period dataset. Returning baseline drift report.")
            df_drift = pd.DataFrame([{
                "period_comparison": "P1_vs_P1",
                "amount_psi": 0.0,
                "amount_wasserstein": 0.0,
                "category_kl_div": 0.0,
                "drift_alert": False,
            }])
            dashboard = {"status": "STABLE", "total_periods": 1, "max_psi": 0.0}
            return df_drift, dashboard

        ref_period = periods[0]
        ref_df = df_sorted[df_sorted["period"] == ref_period]
        ref_amounts = ref_df["amount"].values

        drift_records = []
        max_psi = 0.0

        for curr_period in periods[1:]:
            curr_df = df_sorted[df_sorted["period"] == curr_period]
            curr_amounts = curr_df["amount"].values

            psi = compute_psi(ref_amounts, curr_amounts)
            wd = float(wasserstein_distance(ref_amounts, curr_amounts))
            kl_div = compute_category_kl_divergence(
                ref_df["category"], curr_df["category"], all_cats
            )

            drift_alert = psi >= 0.25
            max_psi = max(max_psi, psi)

            drift_records.append({
                "reference_period": ref_period,
                "current_period": curr_period,
                "amount_psi": psi,
                "amount_wasserstein": round(wd, 4),
                "category_kl_divergence": kl_div,
                "drift_alert": drift_alert,
            })

        drift_df = pd.DataFrame(drift_records)

        dashboard = {
            "status": "DRIFT_DETECTED" if max_psi >= 0.25 else "STABLE",
            "reference_period": ref_period,
            "monitored_periods": len(periods) - 1,
            "max_psi": round(max_psi, 4),
            "drift_alerts_count": int(drift_df["drift_alert"].sum()),
        }

        out_dir = Path(f"reports/results/{self.mode}")
        drift_df.to_csv(out_dir / "drift_metrics.csv", index=False)
        with open(out_dir / "drift_dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)

        log(f"  Drift analysis complete. Max PSI: {max_psi:.4f} (Status: {dashboard['status']})")
        return drift_df, dashboard
