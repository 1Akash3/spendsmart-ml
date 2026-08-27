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


def _make_timestamp(rng: np.random.Generator, base_date: pd.Timestamp) -> pd.Timestamp:
    h = int(rng.integers(8, 22))
    m = int(rng.integers(0, 60))
    s = int(rng.integers(0, 60))
    return base_date + pd.Timedelta(hours=h, minutes=m, seconds=s)


def _make_row(uid: int, ts: pd.Timestamp, amt: float, merch: str, desc: str, cat: str,
              tp: str, is_recurring: bool, persona: str, income: float, row_idx: int) -> dict:
    direction = "credit" if tp == "income" else "debit"
    merch_key = str(merch).lower().strip().replace(" ", "_")
    return dict(
        user_id=uid,
        timestamp=ts,
        date=ts,
        amount=round(float(amt), 2),
        direction=direction,
        merchant=merch,
        merchant_raw=merch,
        merchant_key=merch_key,
        description=desc,
        category=cat,
        type=tp,
        is_recurring=is_recurring,
        persona=persona,
        income=round(float(income), 2),
        source="synthetic",
        currency="INR",
        transaction_id=f"tx_{uid}_{row_idx}"
    )


def generate_transactions(
    n_users: int = SYNTH_N_USERS,
    months: int = SYNTH_MONTHS,
    start: str = SYNTH_START,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a transaction-level DataFrame.

    Columns: user_id, timestamp, date, amount, direction, merchant, merchant_raw, merchant_key,
             description, category, type, is_recurring, persona, income, source, currency, transaction_id
    """
    rng = _rng(seed)
    persona_names = list(PERSONAS.keys())
    start_ts = pd.Timestamp(start)
    rows: list[dict] = []
    row_count = 0

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
            sal_day = _make_timestamp(rng, date0.replace(day=1) + pd.Timedelta(days=int(rng.integers(0, 3))))
            row_count += 1
            rows.append(_make_row(uid, sal_day, income, "Salary Credit",
                                  _make_description(rng, "Salary Credit"),
                                  INCOME_CATEGORY, "income", True, persona, income, row_count))
            if rng.random() < 0.15:
                bonus_m = rng.choice(["Bonus", "Freelance Payout", "Interest Credit"])
                bonus_day = _make_timestamp(rng, sal_day + pd.Timedelta(days=int(rng.integers(3, 20))))
                row_count += 1
                rows.append(_make_row(uid, bonus_day, income * rng.uniform(0.1, 0.5),
                                      bonus_m, _make_description(rng, bonus_m),
                                      INCOME_CATEGORY, "income", False, persona, income, row_count))

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
                    day = _make_timestamp(rng, date0.replace(day=min(int(rng.integers(1, 6)), 28)))
                    row_count += 1
                    rows.append(_make_row(uid, day, amt, merch, _make_description(rng, merch),
                                          c, "expense", True, persona, income, row_count))
                    # A couple of extra subscriptions for realism.
                    if c == "subscriptions" and rng.random() < 0.7:
                        for _ in range(int(rng.integers(1, 4))):
                            merch2 = rng.choice(MERCHANTS[c])
                            amt2 = amt * rng.uniform(0.1, 0.4)
                            day2 = _make_timestamp(rng, date0.replace(day=min(int(rng.integers(1, 26)), 28)))
                            row_count += 1
                            rows.append(_make_row(uid, day2, amt2, merch2,
                                                  _make_description(rng, merch2),
                                                  c, "expense", True, persona, income, row_count))
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
                    day = _make_timestamp(rng, date0 + pd.Timedelta(days=int(rng.integers(0, 27))))
                    row_count += 1
                    rows.append(_make_row(uid, day, float(amt), merch,
                                          _make_description(rng, merch),
                                          c, "expense", False, persona, income, row_count))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"]
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
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
