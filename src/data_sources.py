"""Registry + loaders for REAL public datasets, plus taxonomy normalization.

The pipeline runs on synthetic data by default. These loaders are optional and only import their
heavy dependencies (`datasets`, `kagglehub`) lazily, so `import data_sources` never fails on a
clean environment. (Named `data_sources` — NOT `datasets` — to avoid shadowing HuggingFace's
`datasets` library, which the HF loader imports.) Each loader returns a DataFrame normalized to
the project's columns where possible: at minimum `description` + `category`.

See data/README.md for the full researched catalog.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from config import EXPENSE_CATEGORIES, INCOME_CATEGORY, RAW_DIR


# --------------------------------------------------------------------------------------
# Category normalization: map external label text -> project taxonomy.
# --------------------------------------------------------------------------------------
CATEGORY_ALIASES = {
    # HuggingFace mitulshah/transaction-categorization (10 classes)
    "food & dining": "food_dining",
    "transportation": "transportation",
    "shopping & retail": "shopping",
    "entertainment & recreation": "entertainment",
    "healthcare & medical": "healthcare",
    "utilities & services": "utilities",
    "financial services": "financial",
    "income": INCOME_CATEGORY,
    "government & legal": "misc",
    "charity & donations": "misc",
    # Common Kaggle variants
    "food": "food_dining",
    "dining": "food_dining",
    "restaurants": "food_dining",
    "groceries": "groceries",
    "grocery": "groceries",
    "transport": "transportation",
    "travel": "transportation",
    "fuel": "transportation",
    "shopping": "shopping",
    "retail": "shopping",
    "clothing": "shopping",
    "entertainment": "entertainment",
    "recreation": "entertainment",
    "health": "healthcare",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "utilities": "utilities",
    "bills": "utilities",
    "rent": "housing",
    "housing": "housing",
    "mortgage": "housing",
    "subscription": "subscriptions",
    "subscriptions": "subscriptions",
    "finance": "financial",
    "investment": "financial",
    "insurance": "financial",
    "salary": INCOME_CATEGORY,
    "wage": INCOME_CATEGORY,
}


def normalize_category(raw: str) -> str:
    """Map an arbitrary external category string to the project taxonomy (fallback: 'misc')."""
    if raw is None:
        return "misc"
    key = str(raw).strip().lower()
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    # token containment fallback
    for token, mapped in CATEGORY_ALIASES.items():
        if token in key:
            return mapped
    valid = set(EXPENSE_CATEGORIES) | {INCOME_CATEGORY}
    return key if key in valid else "misc"


# --------------------------------------------------------------------------------------
# Loaders (lazy heavy imports inside each function)
# --------------------------------------------------------------------------------------
def load_hf_transaction_categorization(sample: Optional[int] = 200_000) -> pd.DataFrame:
    """HuggingFace `mitulshah/transaction-categorization` — ~4.5M rows, MIT.

    Returns columns: description, category (normalized). `sample` caps rows for speed.
    Requires: `pip install datasets` and possibly `huggingface-cli login`.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset("mitulshah/transaction-categorization", split="train")
    if sample:
        ds = ds.shuffle(seed=42).select(range(min(sample, len(ds))))
    df = ds.to_pandas()
    df = df.rename(columns={"transaction_description": "description"})
    df["category"] = df["category"].map(normalize_category)
    return df[["description", "category"]].dropna()


# Real Kaggle credit-card categories -> project taxonomy (used by load_real_transactions).
CC_CATEGORY_MAP = {
    "grocery_pos": "groceries", "grocery_net": "groceries",
    "gas_transport": "transportation", "travel": "transportation",
    "food_dining": "food_dining",
    "entertainment": "entertainment",
    "shopping_net": "shopping", "shopping_pos": "shopping",
    "health_fitness": "healthcare",
    "home": "housing",
    "personal_care": "misc", "kids_pets": "misc", "misc_net": "misc", "misc_pos": "misc",
}


def load_real_transactions(sample_users: Optional[int] = None) -> pd.DataFrame:
    """Load REAL per-user transactions (Kaggle credit-card dataset), normalized to the
    pipeline's schema so the whole pipeline can train on real data.

    Returns columns: user_id, date (datetime), type ('expense'), category (project taxonomy),
    amount, description (merchant). 983 users × 18 months × ~1.3M real transactions.
    Requires: `pip install kagglehub` + a Kaggle token at ~/.kaggle/ (access_token or kaggle.json).
    """
    import glob
    import kagglehub  # lazy

    path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")
    csvs = sorted(glob.glob(f"{path}/**/*.csv", recursive=True))
    if not csvs:
        raise FileNotFoundError(f"No CSV in credit-card-transactions-dataset at {path}")
    df = pd.read_csv(csvs[0], usecols=["cc_num", "trans_date_trans_time", "merchant", "category", "amt"])
    df["date"] = pd.to_datetime(df["trans_date_trans_time"])
    df["category"] = df["category"].map(CC_CATEGORY_MAP)
    df = df.dropna(subset=["category"]).copy()
    df["user_id"] = df["cc_num"]
    df["type"] = "expense"
    df["amount"] = df["amt"].astype(float)
    df["description"] = df["merchant"].astype(str).str.replace("fraud_", "", regex=False)
    if sample_users:
        keep = (df["user_id"].drop_duplicates()
                .sample(min(sample_users, df["user_id"].nunique()), random_state=42))
        df = df[df["user_id"].isin(keep)]
    return df[["user_id", "date", "type", "category", "amount", "description"]].reset_index(drop=True)


# khushikyad personal-finance-tracker categories -> project taxonomy.
PF_CATEGORY_MAP = {
    "dining out": "food_dining", "groceries": "groceries", "transportation": "transportation",
    "entertainment": "entertainment", "healthcare": "healthcare", "utilities": "utilities",
    "rent": "housing", "insurance": "financial", "investments": "financial", "education": "misc",
}


def load_pf_tracker_transactions() -> pd.DataFrame:
    """khushikyad001/personal-finance-tracker-dataset -> transaction schema.

    Adds the categories the credit-card set lacks: real monthly **income**, **housing** (rent),
    **financial** (loan + investment), and **utilities/insurance**. Monthly-aggregate rows are
    expanded into (user, month, category) transactions. Users are prefixed `pf_`.
    """
    import glob
    import kagglehub  # lazy
    p = kagglehub.dataset_download("khushikyad001/personal-finance-tracker-dataset")
    csv = sorted(glob.glob(f"{p}/**/*.csv", recursive=True))[0]
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    def num(row, col):
        try:
            return float(row.get(col, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    rows = []
    for _, r in df.iterrows():
        uid = f"pf_{int(r['user_id'])}"
        d = r["date"]
        inc = num(r, "monthly_income")
        if inc > 0:
            rows.append((uid, d, "income", INCOME_CATEGORY, inc, "monthly income"))
        # Allocate the REAL monthly expense total across categories so per-user savings rate
        # matches the dataset (housing = rent, financial = loan, remainder = the row's category).
        total = num(r, "monthly_expense_total")
        if total <= 0:
            total = num(r, "essential_spending") + num(r, "discretionary_spending")
        housing = min(num(r, "rent_or_mortgage"), total)
        rem = total - housing
        fin = min(num(r, "loan_payment"), rem)
        rem -= fin
        cat = PF_CATEGORY_MAP.get(str(r.get("category", "")).strip().lower(), "misc")
        if cat in ("housing", "financial"):
            cat = "misc"
        if housing > 0:
            rows.append((uid, d, "expense", "housing", housing, "rent/mortgage"))
        if fin > 0:
            rows.append((uid, d, "expense", "financial", fin, "loan payment"))
        if rem > 0:
            rows.append((uid, d, "expense", cat, rem, f"{r.get('category')} spend"))
    return pd.DataFrame(rows, columns=["user_id", "date", "type", "category", "amount", "description"])


def load_all_real_transactions(sample_users: Optional[int] = None) -> pd.DataFrame:
    """Union of ALL usable real transaction sources, normalized to the pipeline schema.

    credit-card (rich per-category expenses, 983 users) + personal-finance-tracker (real income,
    housing, financial, utilities). Together they cover income + every expense category, fixing the
    single-source gaps. `sample_users` optionally caps the credit-card portion for a faster run.
    """
    cc = load_real_transactions(sample_users=sample_users)
    cc["user_id"] = "cc_" + cc["user_id"].astype(str)
    try:
        pf = load_pf_tracker_transactions()
    except Exception:
        pf = cc.iloc[0:0]
    both = pd.concat([cc, pf], ignore_index=True)
    return both


# Indian expense/merchant datasets -> project taxonomy (for UPI-domain categorization).
_PRASAD_MAP = {
    "Food": "food_dining", "Transportation": "transportation", "Household": "groceries",
    "subscription": "subscriptions", "Other": "misc", "Investment": "financial",
    "Health": "healthcare", "Family": "misc", "Apparel": "shopping", "Money transfer": "misc",
    "Gift": "shopping", "Recurring Deposit": "financial", "Beauty": "shopping", "Education": "misc",
    "maid": "misc", "Festivals": "misc", "Culture": "entertainment",
    "Public Provident Fund": "financial", "Tourism": "transportation", "Rent": "housing",
    "Cook": "misc", "Grooming": "shopping", "water (jar /tanker)": "utilities",
    "Self-development": "misc", "Documents": "misc", "garbage disposal": "utilities",
    "Social Life": "entertainment",
}
_HARUN_MAP = {
    "Streaming Service": "subscriptions", "Education Fee": "misc", "Hotel Booking": "transportation",
    "Water Bill": "utilities", "Movie Ticket": "entertainment", "Food Delivery": "food_dining",
    "Taxi Fare": "transportation", "Electricity Bill": "utilities", "Rent Payment": "housing",
    "Gas Bill": "utilities", "Loan Repayment": "financial", "Online Shopping": "shopping",
    "Mobile Recharge": "utilities", "Grocery Shopping": "groceries", "Bus Ticket": "transportation",
    "Internet Bill": "utilities", "Gaming Credits": "entertainment", "Insurance Premium": "financial",
    "Gift Card": "shopping", "Flight Booking": "transportation",
}


# belbino Indian banking merchant_category -> project taxonomy.
_BELBINO_MAP = {
    "Food & Dining": "food_dining", "Fuel": "transportation", "Travel": "transportation",
    "E-Commerce": "shopping", "Retail": "shopping", "Entertainment": "entertainment",
    "Healthcare": "healthcare", "Utilities": "utilities", "Real Estate": "housing",
    "Insurance": "financial", "Investment": "financial", "Education": "misc",
    "Government": "misc",
    # 'Salary' -> income (handled separately); 'Peer Transfer' -> dropped (no category).
}


def load_indian_bank_transactions(sample_users: Optional[int] = None,
                                  min_months: int = 4) -> pd.DataFrame:
    """REAL Indian per-user transaction history — belbino/indian-banking-transactions-20192024.

    ~550k transactions, ~80k Indian customers, 2019-2024, with merchant categories AND a real
    'Salary' credit stream, so Indian users get genuine income + multi-month panels. This is what
    makes the forecaster / segmentation / recommender work on Indian data rather than US cohorts.

    Returns the pipeline schema: user_id, date, type, category, amount, description.
    """
    import glob
    import kagglehub  # lazy
    p = kagglehub.dataset_download("belbino/indian-banking-transactions-20192024")
    csv = sorted(glob.glob(f"{p}/**/*.csv", recursive=True))[0]
    df = pd.read_csv(csv, usecols=["customer_id", "transaction_date", "transaction_amount",
                                   "transaction_direction", "merchant_category"])
    df["date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["date"])

    is_salary = df["merchant_category"].eq("Salary") & df["transaction_direction"].eq("Credit")
    inc = df[is_salary].copy()
    inc["type"] = "income"
    inc["category"] = INCOME_CATEGORY

    exp = df[df["transaction_direction"].eq("Debit")].copy()
    exp["category"] = exp["merchant_category"].map(_BELBINO_MAP)
    exp = exp.dropna(subset=["category"])          # drops 'Peer Transfer' (uncategorizable)
    exp["type"] = "expense"

    out = pd.concat([exp, inc], ignore_index=True)
    out["user_id"] = "in_" + out["customer_id"].astype(str)
    out["amount"] = out["transaction_amount"].astype(float)
    out["description"] = out["merchant_category"].astype(str)

    # keep users with enough monthly history for lag features / forecasting
    months = out.assign(_m=out["date"].dt.tz_localize(None).dt.to_period("M")).groupby("user_id")["_m"].nunique()
    keep = months[months >= min_months].index
    out = out[out["user_id"].isin(keep)]
    if sample_users:
        sel = pd.Series(out["user_id"].unique())
        sel = sel.sample(min(sample_users, len(sel)), random_state=42)
        out = out[out["user_id"].isin(set(sel))]
    return out[["user_id", "date", "type", "category", "amount", "description"]].reset_index(drop=True)


# shriyashjagtap Indian personal-finance columns -> project taxonomy.
_SHRIYASH_MAP = {
    "Rent": "housing", "Groceries": "groceries", "Transport": "transportation",
    "Eating_Out": "food_dining", "Entertainment": "entertainment", "Utilities": "utilities",
    "Healthcare": "healthcare", "Education": "misc", "Miscellaneous": "misc",
    "Loan_Repayment": "financial", "Insurance": "financial",
}


def load_indian_profiles() -> pd.DataFrame:
    """REAL Indian user profiles for fair peer comparison —
    shriyashjagtap/indian-personal-finance-and-spending-habits (20,000 Indian users).

    Each row is one user's monthly income + spend per category, so it yields the
    profile feature vector the segmenter clusters on (log income, savings rate, category shares).
    This is what makes "users like you" mean *Indian* users rather than US card holders.

    Note: this dataset has no shopping/subscriptions columns, so those shares are 0 — callers
    should fall back to a global norm for those two categories (see build_indian_segmenter).
    """
    import glob
    import kagglehub  # lazy
    import numpy as np
    p = kagglehub.dataset_download("shriyashjagtap/indian-personal-finance-and-spending-habits")
    d = pd.read_csv(sorted(glob.glob(f"{p}/**/*.csv", recursive=True))[0])

    cats = {c: 0.0 for c in EXPENSE_CATEGORIES}
    out = pd.DataFrame({c: pd.Series(0.0, index=d.index) for c in cats})
    for col, cat in _SHRIYASH_MAP.items():
        if col in d.columns:
            out[cat] = out[cat] + pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    income = pd.to_numeric(d["Income"], errors="coerce").fillna(0.0)
    total = out[EXPENSE_CATEGORIES].sum(axis=1)
    prof = pd.DataFrame({"user_id": ["ind_%d" % i for i in range(len(d))]})
    prof["log_income"] = np.log1p(income)
    prof["savings_rate"] = np.clip(1.0 - total / income.replace(0, np.nan), -1, 1).fillna(0.0)
    prof["volatility"] = 0.0          # single-month snapshot: no within-user time series
    prof["trend"] = 0.0
    prof["mean_expense"] = total
    for c in EXPENSE_CATEGORIES:
        prof[f"share_{c}"] = (out[c] / total.replace(0, np.nan)).fillna(0.0)
    return prof.set_index("user_id")


def load_indian_labeled() -> pd.DataFrame:
    """Real INDIAN (description -> category) pairs so the categorizer handles UPI-domain text:
    prasad22/daily-transactions (real expense-tracker notes) + harunrai/digital-wallet
    (real merchant names). Returns columns: description, category (project taxonomy)."""
    import glob
    import kagglehub  # lazy
    frames = []
    try:  # real Indian expense notes
        p = kagglehub.dataset_download("prasad22/daily-transactions-dataset")
        d = pd.read_csv(sorted(glob.glob(f"{p}/**/*.csv", recursive=True))[0])
        d = d[d.get("Income/Expense") == "Expense"].copy()
        d["description"] = (d["Subcategory"].fillna("").astype(str) + " " + d["Note"].fillna("").astype(str)).str.strip()
        d["category"] = d["Category"].map(_PRASAD_MAP)
        frames.append(d[["description", "category"]])
    except Exception:
        pass
    try:  # real merchant names + product
        p = kagglehub.dataset_download("harunrai/digital-wallet-transactions")
        d = pd.read_csv(sorted(glob.glob(f"{p}/**/*.csv", recursive=True))[0])
        d["description"] = (d["merchant_name"].astype(str) + " " + d["product_name"].astype(str)).str.strip()
        d["category"] = d["product_category"].map(_HARUN_MAP)
        frames.append(d[["description", "category"]])
    except Exception:
        pass
    if not frames:
        return pd.DataFrame(columns=["description", "category"])
    out = pd.concat(frames, ignore_index=True).dropna()
    return out[out["description"].str.len() > 1]


def load_all_labeled_descriptions(cc_sample: int = 40_000, upi_augment: bool = True) -> pd.DataFrame:
    """Combined REAL (description -> category) pairs for the categorizer, from every labeled source:
    credit-card merchants + HuggingFace natural-language expense messages + real Indian expense
    text, plus (by default) UPI-shaped renderings so the model handles GPay/PhonePe merchant
    strings. `cc_sample` is capped so the 1.3M US card merchants do not swamp the Indian signal."""
    import json as _json
    frames = []
    # credit-card merchants
    cc = load_real_transactions()[["description", "category"]].dropna()
    if len(cc) > cc_sample:
        cc = cc.sample(cc_sample, random_state=42)
    frames.append(cc)
    # HuggingFace natural-language expense messages (public)
    try:
        from datasets import load_dataset, concatenate_datasets
        hf_map = {"food_and_drink": "food_dining", "groceries": "groceries",
                  "transportation": "transportation", "travel": "transportation", "fuel": "transportation",
                  "electronics": "shopping", "clothing": "shopping", "shopping": "shopping",
                  "entertainment": "entertainment", "subscriptions": "subscriptions",
                  "pharmacy_and_health": "healthcare", "utilities": "utilities",
                  "books_and_education": "misc", "other": "misc"}
        d = load_dataset("razisayyed/expense-extraction-dataset")
        ds = concatenate_datasets([d[s] for s in d.keys()])
        recs = []
        for r in ds:
            try:
                exps = _json.loads(r["assistant_response"]).get("expenses", [])
                if len(exps) == 1 and exps[0].get("category") in hf_map:
                    recs.append((r["user_message"].strip(), hf_map[exps[0]["category"]]))
            except Exception:
                pass
        if recs:
            frames.append(pd.DataFrame(recs, columns=["description", "category"]))
    except Exception:
        pass
    # Real Indian expense/merchant text so the categorizer handles UPI-domain descriptions.
    indian = load_indian_labeled()
    if len(indian):
        frames.append(indian)

    out = pd.concat(frames, ignore_index=True).dropna()

    if upi_augment:
        from indian_lexicon import generate_upi_training_data, upi_variants
        # (a) Indian brand + business-word patterns rendered in UPI shape.
        gen = pd.DataFrame(generate_upi_training_data(), columns=["description", "category"])
        # (b) UPI-shaped renderings of the real Indian corpus (space-stripped / upper-case),
        #     so the model sees the same semantics in the token shape statements actually use.
        ind_aug = []
        for d, c in zip(indian["description"], indian["category"]):
            for v in upi_variants(d)[:2]:      # joined-upper + joined
                ind_aug.append((v, c))
        aug = pd.DataFrame(ind_aug, columns=["description", "category"])
        out = pd.concat([out, gen, aug], ignore_index=True)

    out = out.drop_duplicates(subset=["description", "category"])
    return out[out["description"].astype(str).str.len() > 1].reset_index(drop=True)


def load_kaggle(slug: str, sample: Optional[int] = None) -> pd.DataFrame:
    """Download a Kaggle dataset via kagglehub and load the first CSV found.

    `slug` e.g. "sahideseker/personal-expense-classification-dataset".
    Requires: `pip install kagglehub` + Kaggle credentials (kaggle.json).
    Returns the raw DataFrame (caller maps columns); does light column-name normalization.
    """
    import glob
    import kagglehub  # lazy

    path = kagglehub.dataset_download(slug)
    csvs = sorted(glob.glob(f"{path}/**/*.csv", recursive=True))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in downloaded dataset '{slug}' at {path}")
    df = pd.read_csv(csvs[0])
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@dataclass
class DatasetSpec:
    key: str
    name: str
    source: str
    url: str
    rows: str
    license: str
    use: str
    loader: Optional[Callable[..., pd.DataFrame]] = field(default=None)


# Public registry mirrored from data/README.md — programmatic catalog.
REGISTRY: dict[str, DatasetSpec] = {
    "hf_txn_cat": DatasetSpec(
        key="hf_txn_cat",
        name="Transaction Categorization",
        source="HuggingFace",
        url="https://huggingface.co/datasets/mitulshah/transaction-categorization",
        rows="~4.5M", license="MIT", use="categorizer",
        loader=load_hf_transaction_categorization,
    ),
    "kaggle_expense_cls": DatasetSpec(
        key="kaggle_expense_cls",
        name="Personal Expense Classification",
        source="Kaggle",
        url="https://www.kaggle.com/datasets/sahideseker/personal-expense-classification-dataset",
        rows="synthetic", license="Kaggle terms", use="categorizer",
        loader=lambda **k: load_kaggle("sahideseker/personal-expense-classification-dataset", **k),
    ),
    "kaggle_finance_tracker": DatasetSpec(
        key="kaggle_finance_tracker",
        name="Personal Finance Tracker",
        source="Kaggle",
        url="https://www.kaggle.com/datasets/khushikyad001/personal-finance-tracker-dataset",
        rows="~3k users", license="Kaggle terms", use="personalization",
        loader=lambda **k: load_kaggle("khushikyad001/personal-finance-tracker-dataset", **k),
    ),
    "kaggle_expenses_income": DatasetSpec(
        key="kaggle_expenses_income",
        name="Financial Transactions (Expenses & Income)",
        source="Kaggle",
        url="https://www.kaggle.com/datasets/artemkabseu/financial-transactions-dataset-expenses-and-income",
        rows="—", license="Kaggle terms", use="panel",
        loader=lambda **k: load_kaggle("artemkabseu/financial-transactions-dataset-expenses-and-income", **k),
    ),
    "kaggle_credit_card": DatasetSpec(
        key="kaggle_credit_card",
        name="Credit Card Transactions",
        source="Kaggle",
        url="https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset",
        rows="~1.85M", license="Kaggle terms", use="behavior/segmentation",
        loader=lambda **k: load_kaggle("priyamchoksi/credit-card-transactions-dataset", **k),
    ),
}


def print_catalog() -> None:
    print(f"{'key':<22}{'name':<38}{'rows':<10}{'license':<14}use")
    print("-" * 100)
    for s in REGISTRY.values():
        print(f"{s.key:<22}{s.name:<38}{s.rows:<10}{s.license:<14}{s.use}")


if __name__ == "__main__":
    print_catalog()
    print(f"\nRaw download dir: {RAW_DIR}")
    print("Loaders are optional — install `datasets` / `kagglehub` + creds to use them.")
