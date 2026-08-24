# Dataset Catalog

Public datasets researched for training the personalized spending‑recommendation model.
The pipeline runs on **synthetic data by default** (no credentials); the real sets below are
optional and wired up in [`../src/data_sources.py`](../src/data_sources.py).

Two data needs, two kinds of source:

- **Categorization** (map a transaction description → category): needs labeled text.
- **Personalization / forecasting** (learn a user's baseline & future): needs *per‑user, multi‑period*
  history — this is rarer and is why the synthetic generator matters.

---

## Tier 1 — recommended, permissively licensed

| Dataset | Source | Rows | Key fields | License | Best for |
|---|---|---|---|---|---|
| **Transaction Categorization** | [HuggingFace `mitulshah/transaction-categorization`](https://huggingface.co/datasets/mitulshah/transaction-categorization) | ~4.50M | `transaction_description`, `category` (10), `country`, `currency` | **MIT** | Categorizer training (large, clean) |
| **Personal Finance Tracker** | [Kaggle `khushikyad001/...`](https://www.kaggle.com/datasets/khushikyad001/personal-finance-tracker-dataset) | ~3,000 users × monthly | per‑user monthly financial snapshot incl. category spend | Kaggle terms | Personalization / forecasting (multi‑user panel) |
| **Personal Expense Classification** | [Kaggle `sahideseker/...`](https://www.kaggle.com/datasets/sahideseker/personal-expense-classification-dataset) | synthetic | merchant, description → category | Kaggle terms | Categorizer (merchant‑level) |
| **Financial Transactions (Expenses & Income)** | [Kaggle `artemkabseu/...`](https://www.kaggle.com/datasets/artemkabseu/financial-transactions-dataset-expenses-and-income) | — | dated income/expense transactions w/ categories | Kaggle terms | End‑to‑end panel |

## Tier 2 — large / behavioral (heavier, weaker labels)

| Dataset | Source | Rows | Notes |
|---|---|---|---|
| **Credit Card Transactions** | [Kaggle `priyamchoksi/...`](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data) | ~1.85M | time, amount, merchant, anonymized user — good for behavior/segmentation |
| **Bank Transaction (Fraud Detection)** | [Kaggle `valakhorasani/...`](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) | ~2,512 | rich behavioral fields; small |
| **Personal Budget Transactions** | [Kaggle `ismetsemedov/...`](https://www.kaggle.com/datasets/ismetsemedov/personal-budget-transactions-dataset) | — | budget vs actual by category |
| **BudgetWise Personal Finance** | [Kaggle `mohammedarfathr/...`](https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset) | — | intentionally messy (typos, dupes, outliers) — preprocessing practice |

## Tier 3 — reference / open banking

- [Open Banking Tracker — free datasets](https://www.openbankingtracker.com/open-banking-datasets)
- Academic: *"Application of Recommender System for Spending‑Habits‑Based Campaign Management"*
  (662,088 transactions, 4,997 customers) — methodology reference for spend‑based recommenders.

---

## Category taxonomy mapping

External datasets use different category names. `src/data_sources.py::CATEGORY_ALIASES` normalizes them
to the project taxonomy defined in `config.py` (e.g. HF's *"Food & Dining"* → `food_dining`,
*"Shopping & Retail"* → `shopping`). Anything unmatched falls back to `misc`.

## Notes on access

- **HuggingFace** set may require accepting terms + a token: `huggingface-cli login`, then
  `load_dataset("mitulshah/transaction-categorization")`.
- **Kaggle** sets download via `kagglehub.dataset_download("<owner>/<slug>")` after placing
  `kaggle.json` credentials (Colab: use `kagglehub` + `google.colab` auth, or upload the token).
- Downloaded files go to `data/raw/` (gitignored). None are committed to this repo.
