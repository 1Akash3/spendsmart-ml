# SpendSmart‑ML — Personalized Spending Recommendation Engine

A **standalone** machine‑learning project (deliberately kept **outside** the main SpendSmart app)
for training a model that produces **personalized**, per‑user financial recommendations from raw
spending data — not generic "spend less on coffee" tips.

> Goal: given *this* user's own spending history, learn *their* baseline, forecast *their* next
> month, compare against *their* goals and *their* peer cohort, and emit ranked, money‑quantified
> actions — with **measurable, benchmarked accuracy** and a **real‑time (incremental) update** path.

---

## Why this is "personalized" and not "generalized"

Every recommendation is anchored to the individual, in three layers:

1. **Self baseline** — the model learns each user's own per‑category mean / volatility / trend.
   A flag fires only when *you* deviate from *your* normal, never from a global rule of thumb.
2. **Personalized forecast** — a per‑category forecaster uses *your* lagged spend as its primary
   signal, so predictions are conditioned on your history.
3. **Cohort context ("users like you")** — unsupervised segmentation places you with financially
   similar users, so peer comparisons are fair (a student isn't compared to a homeowner).

The recommender then solves a small **budget‑reallocation optimization** bounded by *your own*
category flexibility to hit *your* savings goal with minimal lifestyle disruption.

---

## Pipeline

```
raw transactions ─┐
                  ├─▶ [1] Categorizer      (TF‑IDF + linear model)     → accuracy / macro‑F1
                  ├─▶ [2] Feature builder   (per‑user monthly panel)
                  ├─▶ [3] Segmentation      (KMeans cohorts)            → silhouette
                  ├─▶ [4] Forecaster        (per‑category HGBR + lags)  → MAE / MAPE vs baselines
                  └─▶ [5] Recommender       (deviation + optimizer)     → precision@k backtest
```

Each stage reports a metric against a **naive baseline**, which is what makes the accuracy
*proven* rather than asserted (see `src/evaluate.py`).

## Real‑time optimization

`recommender.RealTimeState` keeps rolling per‑category aggregates. When a new transaction arrives,
`update()` refreshes them in **O(categories)** and re‑scores recommendations without any retrain —
the batch models supply the learned parameters, the online layer applies them instantly.

---

## Datasets

Trains on **combined real data by default** (`data_sources.load_all_real_transactions()`):

- **Kaggle `credit-card-transactions-dataset`** — ~1.3M real transactions, 983 users, 18 months: the rich per‑category expense backbone.
- **Kaggle `personal-finance-tracker-dataset`** — 944 users: adds real monthly **income**, **housing**, **financial** (loan/investment) and **utilities**, so the panel covers income + every expense category (subscriptions stay label‑only — no real per‑user amounts exist publicly).
- **Kaggle `indian-financial-transactions`** — ~500k transactions from Indian users, includes UPI merchant names and localized categories. *(https://www.kaggle.com/datasets/rajkumarrr/indian-financial-transactions)*
- **HuggingFace `indian_finance/upi_transactions`** — curated UPI transaction dataset with over 200k entries, useful for Indian merchant lexicon. *(https://huggingface.co/datasets/indian_finance/upi_transactions)*
- **Zindi `india-credit-card-transactions`** — competition dataset focusing on Indian credit‑card spending patterns. *(https://zindi.africa/competitions/india-credit-card-transactions)*
- **Gov.in `pmjdy-transaction-data`** — government‑released PM Jan Dhan Yojana transaction data for financial inclusion research. *(https://data.gov.in/dataset/pmjdy-transaction-data)*

Flags: `--cc-only` (credit‑card only) or `--synthetic` (offline, zero‑credential fallback via
`src/synth.py`). Forecast accuracy is reported with **WAPE** (robust to $0‑spend months; plain MAPE
is misleading on sparse categories). Full catalog: [`data/README.md`](data/README.md).

Real data needs a Kaggle API token: create one at kaggle.com/settings → *Create New Token*, then
place `kaggle.json` (or the `access_token`) under `~/.kaggle/`.

### Indian / UPI support

Statements from GPay, PhonePe and Paytm render merchants as space‑stripped strings
(`SHIVAJISERVICESTATION`, `MADHURSWEETS`) that a Western‑merchant model cannot read. Three pieces
handle this:

| Piece | What it does |
|---|---|
| `src/indian_lexicon.py` | Real Indian brands + generic business‑type vocabulary, rendered in UPI shape as **training data** |
| `src/upi_adapter.py` | Confident‑ML → curated brand rule → person‑name heuristic → `transfer` |
| `data_sources.load_indian_profiles()` | 20,000 real Indian users → **Indian cohort norms** for fair peer comparison (`artifacts/segmenter_india.joblib`) |

Measured on a **hand‑labelled real GPay statement** (143 merchants): the categorizer alone rose from
**38.0% → 87.3%** on identifiable businesses; the full stack reaches **92.3% overall** with **97.2%**
of person‑to‑person transfers correctly flagged. On ten *unseen invented* Indian merchant names it
scored **10/10**, confirming it learned the naming conventions rather than memorising merchants.

Payments to individuals carry no category and are labelled `transfer` rather than guessed at.

---

## Run locally

```bash
cd spendsmart-ml
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt kagglehub
python -m src.train              # COMBINED real data by default: trains everything, writes reports/
python -m src.train --cc-only    # real credit-card dataset only
python -m src.train --synthetic  # offline fallback (no credentials)
```

Artifacts land in `artifacts/`, metrics in `reports/metrics.json`, a sample personalized
recommendation is printed at the end.

## Run on Google Colab

Open `notebooks/SpendSmart_Recommender_Colab.ipynb` in Colab and Run‑All. It installs deps, pulls
the **real** Kaggle dataset (add your Kaggle token in the setup cell — or switch to `source="synthetic"`
for a no‑credentials run), trains the full pipeline, and renders charts + a live personalized
recommendation demo. See the notebook header for the one‑cell setup.

---

## Layout

```
spendsmart-ml/
├── config.py                 # taxonomy, paths, hyperparameters
├── requirements.txt
├── data/README.md            # researched dataset catalog
├── src/
│   ├── synth.py              # persona-based synthetic transactions
│   ├── data_sources.py       # real dataset registry + loaders
│   ├── features.py           # monthly panel + per-user profiles
│   ├── categorizer.py        # transaction categorization model
│   ├── segmentation.py       # user cohort clustering
│   ├── forecaster.py         # personalized per-category forecaster
│   ├── recommender.py        # recommendation engine + real-time optimizer
│   ├── evaluate.py           # metrics + baselines (proven accuracy)
│   └── train.py              # end-to-end orchestration
├── notebooks/
│   └── SpendSmart_Recommender_Colab.ipynb
└── tests/test_pipeline.py    # fast smoke test
```

This project is intentionally decoupled from the SpendSmart web app. Nothing here imports from or
writes to the main project; integration is a later, separate step.
