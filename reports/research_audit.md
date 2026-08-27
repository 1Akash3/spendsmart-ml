# SpendSmart ML Codebase Audit Report

## 1. Current Architecture
The current architecture orchestrates a linear pipeline for training and evaluation:
1.  **Data Loading:** Generates synthetic transactions (`synth.py`) or loads real datasets (`data_sources.py`).
2.  **Categorization Training:** Fits models on descriptions (`categorizer.py`).
3.  **Feature Engineering:** Aggregates transactions into a monthly panel and user profiles (`features.py`).
4.  **Segmentation:** Applies KMeans clustering to define cohorts (`segmentation.py`).
5.  **Forecasting & Backtesting:** Prepares supervised lag frames and backtests `PersonalizedForecaster` (`evaluate.py`, `forecaster.py`).
6.  **Recommender Evaluation:** Tests the accuracy of the overspend detector (`evaluate.py`).
7.  **Recommendation Generation:** Recommends alerts using profiles, forecasts, and cohort norms (`recommender.py`).
8.  **Persistence:** Saves models via `joblib` and metrics as JSON.

## 2. Current Models
*   **TransactionCategorizer:** TF-IDF (word+char n-grams) → LogisticRegression. CPU-only, fast.
*   **HybridTransactionCategorizer:** TF-IDF + SentenceTransformer embeddings → LogisticRegression. Requires PyTorch.
*   **PersonalizedForecaster:** `HistGradientBoostingRegressor` trained per expense category.
*   **UserSegmenter:** Standard KMeans clustering on static user profile features.

## 3. Current Datasets
*   **Synthetic Data:** Generated via `synth.py` with simplistic seasonal/noise multipliers.
*   **Kaggle/HF:** Functions exist in `data_sources.py` to lazily pull from HuggingFace and Kaggle (e.g., Indian bank datasets, general PF trackers), heavily dependent on available API tokens and schema consistency.

## 4. Current Metrics
*   **Categorizers:** Evaluated via Accuracy and Macro-F1.
*   **Forecaster:** MAE, WAPE, sMAPE, MAPE, skill_vs_naive (relative to naive baseline), per_category_mae.
*   **Overspend Detector:** Precision, Recall, F1.

## 5. Current Splits
*   **Categorizer:** Uses standard `train_test_split` (random stratified split).
*   **Forecaster:** Uses temporal walk-forward split holding out recent months for backtesting.

## 6. Current Leakage Risks
*   **Features:** `build_monthly_panel` currently operates on the whole dataset at once before train/test splitting, creating high risk of future-data leakage into static user profiles (like average income over the whole period) applied to historical predictions.
*   **Segmentation:** KMeans uses full user profiles incorporating all time periods, leaking future user behavior into early-history segment assignment.

## 7. Current Limitations
*   GPay parsing discards exact timestamps and dates.
*   No standard canonical schema; assumes arbitrary formats depending on the dataset.
*   Categorizer random split overestimates real-world generalization (fails to simulate unseen merchants).
*   Zero uncertainty/probability calibration natively available for the forecasts.
*   No robust mechanism to adjust the balance between global baseline patterns and sparse personal history (Adaptive Personalization).

## 8. Current Deployment Path
*   Currently uses batch scripts (`train.py`) producing static artifacts (`.joblib` models and JSON metrics).
*   Inference scripts like `infer_unlabeled.py` strictly assume `HybridTransactionCategorizer` without robust fallback mechanisms.

## 9. Current Reproducibility Issues
*   `sys.path.insert(0, ...)` logic makes modules location-dependent.
*   Random splits lack guaranteed seed enforcement across all layers.
*   Missing rigorous environment and data hash tracking (e.g. Kaggle datasets might change).

## 10. Current Research Gaps
*   No adaptive personalization balancing global/local models for cold-start users.
*   No sequential deep models (Transformers/LSTMs) baselined against the regressors.
*   Lack of uncertainty/interval forecasting.
*   Lack of rigorous drift and robustness metrics.

## 11. Recommended Modifications
*   Implement strict Canonical Transaction Schema.
*   Fix GPay parser to retain full timestamps.
*   Build rigorous splits (Temporal, Unseen Merchant, Cross-source).
*   Implement Adaptive Router for personalization.
*   Develop PATFormer (Compact Causal Transformer).
*   Refactor feature generation to strictly avoid forward-looking temporal leakage.
