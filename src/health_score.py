"""Financial Health Score Engine for SpendSmart V4.

Computes a transparent, 5-component Financial Health Score (0-100):
1. Savings Consistency (25 pts): Savings rate relative to target.
2. Cash Flow Stability (20 pts): Income vs expense volatility.
3. Essential Spending Burden (20 pts): Housing/utilities/groceries share of income.
4. Discretionary Spending Burden (20 pts): Dining/shopping/entertainment control.
5. Income Regularity (15 pts): Frequency & predictability of income streams.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DISCRETIONARY_CATEGORIES, EXPENSE_CATEGORIES, INCOME_CATEGORY
from src.benchmarks import log


class FinancialHealthCalculator:
    """Calculates 5-component transparent financial health score."""

    def compute_score(self, user_profile: Dict[str, float]) -> Dict[str, Any]:
        """Compute 0-100 financial health score for a user profile dict."""
        savings_rate = user_profile.get("savings_rate", 0.15)
        volatility = user_profile.get("volatility", 0.20)
        cashflow_ratio = user_profile.get("cashflow_ratio", 1.2)

        # 1. Savings Consistency (max 25)
        savings_score = min(25.0, max(0.0, (savings_rate / 0.20) * 25.0))

        # 2. Cash Flow Stability (max 20)
        stability_score = min(20.0, max(0.0, 20.0 * (1.0 - min(1.0, volatility))))

        # 3. Essential Spending Burden (max 20)
        essential_share = (
            user_profile.get("share_groceries", 0.15)
            + user_profile.get("share_housing", 0.20)
            + user_profile.get("share_utilities", 0.10)
        )
        essential_score = min(20.0, max(0.0, 20.0 * (1.0 - min(1.0, essential_share / 0.60))))

        # 4. Discretionary Control (max 20)
        disc_share = sum(user_profile.get(f"share_{c}", 0.05) for c in DISCRETIONARY_CATEGORIES)
        disc_score = min(20.0, max(0.0, 20.0 * (1.0 - min(1.0, disc_share / 0.40))))

        # 5. Income Regularity (max 15)
        income_score = min(15.0, max(0.0, min(1.5, cashflow_ratio) * 10.0))

        total_score = round(savings_score + stability_score + essential_score + disc_score + income_score, 1)

        tier = "Excellent" if total_score >= 80 else ("Good" if total_score >= 65 else "Fair" if total_score >= 50 else "Needs Attention")

        return {
            "total_health_score": total_score,
            "health_tier": tier,
            "components": {
                "savings_consistency": round(savings_score, 1),
                "cashflow_stability": round(stability_score, 1),
                "essential_burden": round(essential_score, 1),
                "discretionary_control": round(disc_score, 1),
                "income_regularity": round(income_score, 1),
            },
            "max_components": {
                "savings_consistency": 25.0,
                "cashflow_stability": 20.0,
                "essential_burden": 20.0,
                "discretionary_control": 20.0,
                "income_regularity": 15.0,
            },
        }


def generate_health_reports(panel: pd.DataFrame, mode: str = "smoke") -> pd.DataFrame:
    """Generate financial health scores for all users in the dataset."""
    log("  Computing Financial Health Scores across panel...")
    calc = FinancialHealthCalculator()
    records = []

    from src.features import build_user_profiles
    profiles = build_user_profiles(panel)

    for uid, row in profiles.iterrows():
        prof_dict = row.to_dict()
        res = calc.compute_score(prof_dict)
        rec = {
            "user_id": str(uid),
            "health_score": res["total_health_score"],
            "tier": res["health_tier"],
            **res["components"],
        }
        records.append(rec)

    df_health = pd.DataFrame(records)
    out_dir = Path(f"reports/results/{mode}")
    df_health.to_csv(out_dir / "financial_health_scores.csv", index=False)
    return df_health
