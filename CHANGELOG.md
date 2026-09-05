# SpendSmart V4 CHANGELOG

## [4.1.1] - SpendSmart V4.1.1 Research Patch

### Fixed
- Fixed missing `drift_metrics.csv` and `drift_dashboard.json` generation logic.
- Corrected PATFormer learning rate scheduler step order in `neural_engine.py` (resolves PyTorch warning).
- Improved Logistic Regression training stability (`max_iter=1000`, `n_jobs=-1`) and removed deprecated `multi_class` argument.
- Silenced Seaborn `palette` deprecation warnings by explicitly setting `hue`.
- Resolved pandas timezone conversion warnings (`dt.tz_localize(None)`) prior to monthly aggregation.
- Implemented Research Gate 6 verification in `run_baselines.py` to assert creation of all empirical benchmark artifacts.
- Added explicit `benchmark_metadata.json` provenance tracking.
- Improved Google Colab compatibility for CLI execution.

## [4.0.0] - 2026-09-01

### Added
- **Master Research Pipeline (`src/run_research_pipeline.py`)**: 14-stage orchestrator executing end-to-end research workflows across smoke, development, and final modes.
- **PATFormer Neural Engine (`src/neural_engine.py`)**: PyTorch causal transformer training with mixed precision (`torch.amp`), CosineAnnealingLR scheduler, gradient clipping, early stopping, and `MODEL-SPLIT-SEED-EPOCH.pt` epoch checkpointing.
- **Adaptive Personalization Engine (`src/personalization.py`)**: Implementation of Global, Personal, Cohort, and Adaptive Routers across cold-start history length buckets (0–5, 6–20, 21–50, 51–100, 100+).
- **Distribution Drift Detection (`src/drift.py`)**: Multi-dimensional monthly drift analysis using Population Stability Index (PSI), Wasserstein Distance, and KL Divergence.
- **Explainability Suite (`src/explainability.py`)**: Permutation importance, counterfactual explanation generator, and PATFormer attention matrix extraction.
- **Transparent Financial Health Score (`src/health_score.py`)**: 5-component scoring (Savings consistency, Cashflow stability, Essential burden, Discretionary control, Income regularity).
- **Production Serving API (`src/serving.py`)**: Sub-50ms JSON serving layer profiling 5 production endpoints.
- **Additive & Subtractive Ablation Engine (`src/ablation.py`)**: Evaluates component removal/addition impact (No temporal, No merchant, No user profile, No router).
- **Statistical Validation Engine (`src/stats.py`)**: Multi-seed (42-46) statistical aggregation, paired t-tests, Wilcoxon signed-rank tests, and bootstrap 95% CIs.
- **Publication Artifacts Auto-Generator (`src/publication_artifacts.py`)**: Produces Tables 1–10, Figures 1–16 (PNG 300 DPI), Model Card, Data Card, Limitations, Error Analysis, Reproducibility Report, Paper Readiness Audit, and Research Gate Verification.
- **Master Research Notebook (`notebooks/SpendSmart_Research_Complete.ipynb`)**: Notebook orchestrator rendering generated artifacts without code duplication.
- **V4 Integration Test Suite (`tests/test_research_pipeline.py`)**: Verifies master pipeline execution, tables 1–10, figures 1–16, paper documents, and serving API SLAs.

### Verified
- **100% Test Pass Rate**: 46 out of 46 unit & integration tests passing cleanly.
- **Compile Checks**: Zero syntax or import errors across `src/`.
