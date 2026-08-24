"""Central configuration: taxonomy, paths, hyperparameters.

Paths are resolved relative to THIS file so the project works identically whether run from a
local shell (`python -m src.train`) or a Colab cell, regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, SYNTH_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CURRENCY = "₹"  # ₹ ; purely cosmetic for printed reports

# --------------------------------------------------------------------------------------
# Category taxonomy
# --------------------------------------------------------------------------------------
# `EXPENSE_CATEGORIES` are the ones we forecast and make recommendations on.
# INCOME is tracked separately (it funds the savings-goal optimizer, never "recommended down").
EXPENSE_CATEGORIES = [
    "food_dining",
    "groceries",
    "transportation",
    "shopping",
    "entertainment",
    "healthcare",
    "utilities",
    "housing",
    "subscriptions",
    "financial",
    "misc",
]
INCOME_CATEGORY = "income"
ALL_CATEGORIES = EXPENSE_CATEGORIES + [INCOME_CATEGORY]

# Human-friendly labels for reports.
CATEGORY_LABELS = {
    "food_dining": "Food & Dining",
    "groceries": "Groceries",
    "transportation": "Transportation",
    "shopping": "Shopping",
    "entertainment": "Entertainment",
    "healthcare": "Healthcare",
    "utilities": "Utilities",
    "housing": "Housing / Rent",
    "subscriptions": "Subscriptions",
    "financial": "Financial / Fees",
    "misc": "Miscellaneous",
    "income": "Income",
}

# Categories a user can realistically cut in the short term (flexibility for the optimizer).
# Housing/utilities/healthcare/financial are treated as largely fixed.
DISCRETIONARY_CATEGORIES = [
    "food_dining",
    "shopping",
    "entertainment",
    "subscriptions",
    "misc",
]

# --------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------
SYNTH_N_USERS = 800
SYNTH_MONTHS = 18          # months of history per user
SYNTH_START = "2024-07-01"

# --------------------------------------------------------------------------------------
# Model hyperparameters
# --------------------------------------------------------------------------------------
# Segmentation
N_COHORTS = 6

# Forecaster (per-category HistGradientBoostingRegressor)
FORECAST_LAGS = 3          # months of lag features
FORECAST_MIN_MONTHS = 6    # a user needs at least this much history to be forecastable
HGBR_PARAMS = dict(max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0)

# Recommender
FLEX_STD_MULT = 1.0        # a category can flex by ±(this × the user's own monthly std)
OVERSPEND_Z = 0.75         # flag when forecast exceeds user baseline by this many std devs
TOP_K_RECOMMENDATIONS = 5

# Evaluation
TEST_FRACTION = 0.2        # last fraction of months held out per user for forecasting backtest
