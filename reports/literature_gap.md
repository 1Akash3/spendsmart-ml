# Literature Gap & Research Contribution

## Established Approaches in Personal Finance ML
1. **Transaction Categorization:** Rule-based heuristics, TF-IDF + Logistic Regression, and standard word embeddings (Word2Vec/FastText) applied to transaction descriptions.
2. **Time Series Forecasting:** Standard global regressors (XGBoost, Random Forest), statistical models (ARIMA, Prophet), and recurrent neural networks (LSTMs) for expense and cash flow forecasting.
3. **Anomaly Detection:** Isolation forests or standard statistical z-score methods applied to user spending amounts.
4. **Personalization:** Typically achieved by treating user IDs as categorical embeddings, or training entirely distinct models per user (which fails on cold start).

## Closest Competing Approaches
* **Global Sequence Models:** Transformer architectures applied across massive population datasets (e.g., Tabular Transformers). 
    * *What they solve:* Captures broad consumer trends, seasonal behaviors, and generalizes well to unseen users with standard spending patterns.
    * *What they do not solve:* Often override highly specific, idiosyncratic user habits. Struggle with the extreme data sparsity of new users (cold start).
* **Local / Personal Models:** Separate models trained strictly on a single user's history.
    * *What they solve:* Perfectly capture idiosyncratic, highly personal recurring expenses and specific merchant habits.
    * *What they do not solve:* Cannot generalize to unseen merchants. Cannot produce forecasts for users with fewer than a few dozen transactions.

## SpendSmart's Proposed Contribution
We propose a **Compact Causal Transaction Transformer (PATFormer) equipped with an Adaptive Personalization Router**. 

Unlike approaches that use a fixed ensemble or static embeddings, the Adaptive Router dynamically calculates a `global_weight` and `personal_weight` continuously at prediction time. The router balances the Global Expert against the Personal Expert based on explicit sparsity metrics: history length, behavioral volatility, merchant entropy, and recent temporal drift.

## Why the Contribution is Meaningful
Real-world personal finance intelligence faces extreme heterogeneity:
1. **Cold-Start:** New users have almost zero history, requiring global priors.
2. **Data Sparsity:** Even mature users frequently encounter entirely unseen merchants or new spending categories.
3. **Distribution Shift:** Individual spending behaviors drift rapidly due to life events (moving, job change) and macro factors (inflation, seasonality).

By making the personalization weighting *adaptive*, the model can gracefully degrade to global knowledge under uncertainty or sparsity, and tightly fit personal history when sufficient, stable evidence exists.

## Experiments Required for Validation
To validate this contribution, we must execute the following controlled experiments:
1. **The Cold-Start Curve:** Plot forecasting performance (MAE/WAPE) against user history length (0-5, 6-20, ..., 250+). We must demonstrate that the Adaptive Router relies on global weights early and shifts to personal weights later, improving overall MAE across the curve.
2. **Unseen Merchant Generalization:** Evaluate specifically on merchant entities completely disjoint from the training set to ensure the global model can cover gaps in personal history.
3. **Temporal Distribution Shift:** Train on months $1..T$ and evaluate on $T+1..N$.
4. **Ablation Study:** Explicitly remove the Adaptive Router and the continuous user representation to quantify their direct contribution to the final performance metrics.
