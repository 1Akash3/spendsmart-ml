# SpendSmart‑ML V4 — Master Personalized Financial Intelligence Engine

A **publication-grade machine learning system** for personalized financial transaction categorization, multi-horizon spending forecasting, uncertainty estimation, distribution drift detection, and budget optimization.

> Goal: Transform raw transaction feeds into empirical evidence, paper-ready publication artifacts (Tables 1–10, Figures 1–16), and sub-50ms production serving APIs with 100% reproducible execution.

---

## 🚀 SpendSmart V4 Quick Start

### 1. Run Data Preprocessing & Splitting Pipeline
```bash
python -m src.run_data_pipeline --mode smoke
```

### 2. Execute Master Research Pipeline
```bash
python -m src.run_research_pipeline --mode smoke
```

### 3. Run Benchmark Engine Only
```bash
python -m src.run_baselines --mode smoke
```

### 4. Run PyTorch Unit & Integration Tests
```bash
pytest tests/ -v
```

---

## 📊 SpendSmart V4 Core Architecture

```
Raw Feed ──▶ [1] Data Pipeline (Canonical Schema, Hashing, Leakage Audit)
            ├──▶ [2] Baseline Engine (A0-A5 Categorization, F0-F4 Forecasting)
            ├──▶ [3] PATFormer Causal Transformer (PyTorch AMP, Cosine Scheduler, Checkpoints)
            ├──▶ [4] Adaptive Personalization (Global, Personal, Cohort, Adaptive Router)
            ├──▶ [5] Temporal Drift Engine (PSI, Wasserstein, KL Divergence)
            ├──▶ [6] Robustness Suite (5%, 10%, 20% Noise Injections)
            ├──▶ [7] Explainability & Counterfactual Engine (Permutation, Attention Heatmaps)
            ├──▶ [8] Financial Health Score (Transparent 5-Component Breakdown)
            ├──▶ [9] Multi-Seed Statistical Validation (Seeds 42-46, Bootstrap 95% CIs)
            └──▶ [10] Publication Artifacts Generator (Tables 1–10, Figures 1–16, Cards)
```

---

## 🛠️ Execution Modes

| Mode | Target | Description |
|------|--------|-------------|
| `smoke` | CPU / Local | Rapid software pipeline verification across 1–3 epochs. *Never used for final paper claims.* |
| `development` | GPU / Colab | Hyperparameter sweeps, full dataset training, and multi-seed ablations. |
| `final` | GPU / Locked | Immutable evaluation enforcing `LockedTestGuard` manifest verification. |

---

## 📄 Generated Paper Artifacts

Executing the master research pipeline automatically produces publication-ready files:

- **Tables 1–10**: CSV format under `reports/results/{mode}/` and `reports/tables/`
- **Figures 1–16**: High-resolution 300 DPI PNG plots under `reports/figures/`
- **Model Card**: [`reports/MODEL_CARD.md`](reports/MODEL_CARD.md)
- **Data Card**: [`reports/DATA_CARD.md`](reports/DATA_CARD.md)
- **Limitations**: [`reports/LIMITATIONS.md`](reports/LIMITATIONS.md)
- **Error Analysis**: [`reports/ERROR_ANALYSIS.md`](reports/ERROR_ANALYSIS.md)
- **Reproducibility Report**: [`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md)
- **Paper Readiness Audit**: [`reports/PAPER_READINESS_AUDIT.md`](reports/PAPER_READINESS_AUDIT.md)

---

## 🌐 Production Serving Layer API

SpendSmart V4 includes a production-ready serving layer (`src/serving.py`) with sub-50ms latency SLAs:

```python
from src.serving import SpendSmartServingAPI

api = SpendSmartServingAPI("smoke")

# 1. Categorize transaction
tx_res = api.predict_transaction("Swiggy Order", amount=450.0)

# 2. Monthly forecast
fc_res = api.forecast_user(user_id="user_123")

# 3. Financial Health Score
health_res = api.health_score(user_id="user_123")
```

---

## 🔬 Scientific Validation SLA

- **Zero Mock Metrics**: Every F1, MAE, PSI, ECE, and p-value is computed directly from model predictions.
- **Statistical Significance**: Paired t-tests and Wilcoxon signed-rank tests across 5 random seeds (42–46).
- **Verification Gate**: Master execution fails automatically if any generated CSV, table, figure, or document is missing or empty.
