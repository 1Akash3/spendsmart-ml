"""Persona-based synthetic transaction generator.

Real per-user, multi-month spending panels are rare in public data. This generator fabricates
them with *distinct per-user spending DNA* so that a personalized model has genuine signal to
learn (and so personalization can beat a one-size-fits-all baseline in evaluation).

Each user is assigned a **persona** (student, family, saver, ...) that sets category shares and
income. On top of that sit: individual variation, a slow multi-month trend for some categories,
month-of-year seasonality, fixed recurring subscriptions, and transaction-level noise (merchants +
free-text descriptions) that also trains the categorizer.

Output: a tidy transaction-level DataFrame — the single source of truth every other module reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    EXPENSE_CATEGORIES,
    INCOME_CATEGORY,
    RANDOM_SEED,
    SYNTH_MONTHS,
    SYNTH_N_USERS,
    SYNTH_START,
)

# --------------------------------------------------------------------------------------
# Personas: relative category weights (unnormalized) + monthly income range.
# --------------------------------------------------------------------------------------
PERSONAS = {
    "student": {
        "income": (12_000, 25_000),
        "weights": dict(food_dining=3.0, groceries=1.2, transportation=1.5, shopping=1.4,
                        entertainment=2.5, healthcare=0.4, utilities=0.8, housing=1.5,
                        subscriptions=2.2, financial=0.3, misc=1.0),
        "save_rate": (0.02, 0.10),
    },
    "young_professional": {
        "income": (45_000, 90_000),
        "weights": dict(food_dining=2.6, groceries=1.6, transportation=1.8, shopping=2.4,
                        entertainment=2.0, healthcare=0.8, utilities=1.2, housing=3.0,
                        subscriptions=1.8, financial=1.2, misc=1.1),
        "save_rate": (0.10, 0.28),
    },
    "family": {
        "income": (60_000, 130_000),
        "weights": dict(food_dining=1.8, groceries=3.5, transportation=2.2, shopping=2.0,
                        entertainment=1.2, healthcare=2.4, utilities=2.2, housing=3.6,
                        subscriptions=1.2, financial=1.5, misc=1.4),
        "save_rate": (0.08, 0.22),
    },
    "saver": {
        "income": (40_000, 95_000),
        "weights": dict(food_dining=1.0, groceries=1.6, transportation=1.0, shopping=0.8,
                        entertainment=0.7, healthcare=1.0, utilities=1.1, housing=2.2,
                        subscriptions=0.7, financial=2.6, misc=0.7),
        "save_rate": (0.30, 0.55),
    },
    "high_spender": {
        "income": (80_000, 180_000),
        "weights": dict(food_dining=3.4, groceries=1.8, transportation=2.0, shopping=4.2,
                        entertainment=3.2, healthcare=1.0, utilities=1.4, housing=3.4,
                        subscriptions=2.6, financial=1.0, misc=1.8),
        "save_rate": (0.02, 0.14),
    },
    "retiree": {
        "income": (30_000, 60_000),
        "weights": dict(food_dining=1.2, groceries=2.4, transportation=0.9, shopping=1.2,
                        entertainment=1.0, healthcare=3.2, utilities=2.0, housing=1.6,
                        subscriptions=0.9, financial=1.8, misc=1.0),
        "save_rate": (0.05, 0.20),
    },
}

# --------------------------------------------------------------------------------------
# Merchants + description templates per category (feed the text categorizer).
# --------------------------------------------------------------------------------------
MERCHANTS = {
    "food_dining":   ["Swiggy", "Zomato", "Dominos", "Starbucks", "KFC", "Cafe Coffee Day",
                      "McDonalds", "Barbeque Nation", "Local Dhaba", "Pizza Hut"],
    "groceries":     ["BigBasket", "DMart", "Reliance Fresh", "More Supermarket", "Blinkit",
                      "Zepto", "Spencers", "Nature's Basket"],
    "transportation":["Uber", "Ola", "IndianOil Petrol", "HP Petrol", "Metro Card", "Rapido",
                      "BluSmart", "IRCTC"],
    "shopping":      ["Amazon", "Flipkart", "Myntra", "Ajio", "Croma", "Nykaa", "IKEA", "H&M",
                      "Decathlon"],
    "entertainment": ["BookMyShow", "PVR Cinemas", "Steam", "PlayStation Store", "Dave & Buster",
                      "Wonderla", "Concert Tickets"],
    "healthcare":    ["Apollo Pharmacy", "PharmEasy", "1mg", "Practo", "Manipal Hospital",
                      "Dr Lal PathLabs", "MedPlus"],
    "utilities":     ["Electricity Board", "Airtel Broadband", "Jio Fiber", "Gas Bill",
                      "Water Board", "BSNL", "Tata Power"],
    "housing":       ["Monthly Rent", "Housing Society", "Home Loan EMI", "Maintenance Fee"],
    "subscriptions": ["Netflix", "Spotify", "Amazon Prime", "Disney+ Hotstar", "YouTube Premium",
                      "Gym Membership", "iCloud", "Adobe CC", "LinkedIn Premium"],
    "financial":     ["SIP Mutual Fund", "Insurance Premium", "Bank Charges", "Credit Card Fee",
                      "Brokerage", "PPF Deposit"],
    "misc":          ["ATM Withdrawal", "Gift Shop", "Donation", "Misc UPI", "Stationery",
                      "Pet Store"],
    INCOME_CATEGORY: ["Salary Credit", "Monthly Payroll", "Freelance Payout", "Bonus", "Interest Credit"],
}

_DESC_TEMPLATES = [
    "{m}", "{m} UPI", "POS {m}", "{m} payment", "{m} online", "PAYTM*{m}", "{m} #{ref}",
    "UPI/{m}/{ref}", "{m} auto-debit",
]
_CITIES = ["Mumbai", "Pune", "Bengaluru", "Delhi", "Hyderabad", "Chennai"]

# Categories billed as fixed monthly recurring charges.
_RECURRING_CATS = {"subscriptions", "housing", "utilities"}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_description(rng: np.random.Generator, merchant: str) -> str:
    tmpl = rng.choice(_DESC_TEMPLATES)
    ref = int(rng.integers(1000, 99999))
    desc = tmpl.format(m=merchant, ref=ref)
    if rng.random() < 0.25:
        desc = f"{desc} {rng.choice(_CITIES)}"
    return desc


def _seasonal_multiplier(category: str, month_idx_in_year: int) -> float:
    """Month-of-year (0=Jan) seasonality; returns a multiplier around 1.0."""
    m = month_idx_in_year
    if category in ("shopping", "entertainment") and m in (10, 11):   # festive/Nov-Dec
        return 1.35
    if category == "food_dining" and m in (11, 0):                    # holidays
        return 1.20
    if category == "healthcare" and m in (0, 1):                      # new-year checkups
        return 1.15
    if category == "transportation" and m in (3, 4):                  # summer travel
        return 1.18
    return 1.0


def generate_transactions(
    n_users: int = SYNTH_N_USERS,
    months: int = SYNTH_MONTHS,
    start: str = SYNTH_START,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a transaction-level DataFrame.

    Columns: user_id, date, amount, merchant, description, category, type, is_recurring,
             persona, income
    """
    rng = _rng(seed)
    persona_names = list(PERSONAS.keys())
    start_ts = pd.Timestamp(start)
    rows: list[dict] = []

    for uid in range(n_users):
        persona = persona_names[uid % len(persona_names)] if uid < len(persona_names) \
            else rng.choice(persona_names)
        p = PERSONAS[persona]

        income = float(rng.uniform(*p["income"]))
        save_rate = float(rng.uniform(*p["save_rate"]))
        spendable = income * (1.0 - save_rate)

        # Per-user category shares: persona weights × individual variation, normalized.
        w = np.array([p["weights"][c] for c in EXPENSE_CATEGORIES], dtype=float)
        w = w * rng.uniform(0.8, 1.2, size=w.shape)          # personal idiosyncrasy
        shares = w / w.sum()
        base_budget = dict(zip(EXPENSE_CATEGORIES, shares * spendable))

        # Slow per-category trends: a couple of categories drift up/down over the window.
        trend = {c: 0.0 for c in EXPENSE_CATEGORIES}
        for c in rng.choice(EXPENSE_CATEGORIES, size=2, replace=False):
            trend[c] = float(rng.uniform(-0.010, 0.020))      # monthly compounding drift

        # Fixed recurring amounts (stable month to month).
        recurring_amt = {c: base_budget[c] * rng.uniform(0.85, 1.0)
                         for c in _RECURRING_CATS}

        for m in range(months):
            date0 = start_ts + pd.DateOffset(months=m)
            moy = date0.month - 1

            # Income (1 salary + occasional bonus/freelance).
            sal_day = date0.replace(day=1) + pd.Timedelta(days=int(rng.integers(0, 3)))
            rows.append(dict(user_id=uid, date=sal_day, amount=round(income, 2),
                             merchant="Salary Credit",
                             description=_make_description(rng, "Salary Credit"),
                             category=INCOME_CATEGORY, type="income", is_recurring=True,
                             persona=persona, income=round(income, 2)))
            if rng.random() < 0.15:
                bonus_m = rng.choice(["Bonus", "Freelance Payout", "Interest Credit"])
                rows.append(dict(user_id=uid,
                                 date=sal_day + pd.Timedelta(days=int(rng.integers(3, 20))),
                                 amount=round(income * rng.uniform(0.1, 0.5), 2),
                                 merchant=bonus_m, description=_make_description(rng, bonus_m),
                                 category=INCOME_CATEGORY, type="income", is_recurring=False,
                                 persona=persona, income=round(income, 2)))

            for c in EXPENSE_CATEGORIES:
                growth = (1.0 + trend[c]) ** m
                season = _seasonal_multiplier(c, moy)
                monthly = base_budget[c] * growth * season * rng.uniform(0.82, 1.18)
                if monthly <= 0:
                    continue

                if c in _RECURRING_CATS:
                    # One (or few) fixed recurring charges.
                    amt = recurring_amt[c] * growth * rng.uniform(0.98, 1.02)
                    merch = rng.choice(MERCHANTS[c])
                    day = date0.replace(day=min(int(rng.integers(1, 6)), 28))
                    rows.append(dict(user_id=uid, date=day, amount=round(amt, 2),
                                     merchant=merch, description=_make_description(rng, merch),
                                     category=c, type="expense", is_recurring=True,
                                     persona=persona, income=round(income, 2)))
                    # A couple of extra subscriptions for realism.
                    if c == "subscriptions" and rng.random() < 0.7:
                        for _ in range(int(rng.integers(1, 4))):
                            merch2 = rng.choice(MERCHANTS[c])
                            amt2 = amt * rng.uniform(0.1, 0.4)
                            day2 = date0.replace(day=min(int(rng.integers(1, 26)), 28))
                            rows.append(dict(user_id=uid, date=day2, amount=round(amt2, 2),
                                             merchant=merch2,
                                             description=_make_description(rng, merch2),
                                             category=c, type="expense", is_recurring=True,
                                             persona=persona, income=round(income, 2)))
                    continue

                # Variable category: split monthly total into several transactions.
                n_tx = max(1, int(rng.poisson({"food_dining": 12, "groceries": 6,
                                               "transportation": 8, "shopping": 4,
                                               "entertainment": 3, "healthcare": 2,
                                               "financial": 2, "misc": 3}.get(c, 4))))
                splits = rng.dirichlet(np.ones(n_tx)) * monthly
                for amt in splits:
                    if amt < 1:
                        continue
                    merch = rng.choice(MERCHANTS[c])
                    day = date0 + pd.Timedelta(days=int(rng.integers(0, 27)))
                    rows.append(dict(user_id=uid, date=day, amount=round(float(amt), 2),
                                     merchant=merch, description=_make_description(rng, merch),
                                     category=c, type="expense", is_recurring=False,
                                     persona=persona, income=round(income, 2)))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df


def load_or_generate(path=None, **kwargs) -> pd.DataFrame:
    """Generate (and cache) the synthetic dataset as parquet."""
    from config import SYNTH_DIR
    path = path or (SYNTH_DIR / "transactions.parquet")
    path = pd.io.common.stringify_path(path)
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        df = generate_transactions(**kwargs)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.to_csv(str(path).replace(".parquet", ".csv"), index=False)
        return df


if __name__ == "__main__":
    d = generate_transactions(n_users=20, months=12)
    print(d.head(12).to_string())
    print(f"\n{len(d):,} transactions | {d.user_id.nunique()} users | "
          f"categories: {sorted(d.category.unique())}")
