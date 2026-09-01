"""All categorization and forecasting model implementations.

Every model follows the same interface:
    train(X_train, y_train, config) -> fitted_model
    predict(model, X_test) -> predictions
    predict_proba(model, X_test) -> probability_matrix (where applicable)

No mock metrics. Every function trains a real model.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse, hstack as sp_hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.benchmarks import log

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Feature Builders
# ============================================================================

def build_text_features(
    train_desc: pd.Series,
    test_desc: pd.Series,
    max_features: int = 10000,
) -> Tuple[Any, Any, Any]:
    """Build TF-IDF features (word + char n-grams)."""
    tfidf = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, sublinear_tf=True, min_df=2,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=max_features // 2, sublinear_tf=True, min_df=2,
        )),
    ])
    X_train = tfidf.fit_transform(train_desc.fillna("").astype(str))
    X_test = tfidf.transform(test_desc.fillna("").astype(str))
    return X_train, X_test, tfidf


def build_behavioral_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Extract behavioral numeric features for tree-based models."""
    numeric_cols = []
    for col in ["amount", "hour", "weekday", "is_weekend", "month", "quarter",
                "day_of_month", "is_salary_week", "cumulative_spend",
                "rolling_tx_count", "user_mean_amount", "user_std_amount",
                "merchant_frequency", "merchant_mean_amount",
                "days_since_previous", "running_balance_proxy",
                "history_length", "cashflow_ratio", "recurring_ratio"]:
        if col in train_df.columns:
            numeric_cols.append(col)

    if not numeric_cols:
        # Fallback: use amount only
        for col in ["amount"]:
            if col in train_df.columns:
                numeric_cols.append(col)

    X_train = train_df[numeric_cols].fillna(0).values.astype(float)
    X_test = test_df[numeric_cols].fillna(0).values.astype(float)
    return X_train, X_test, numeric_cols


def build_combined_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 10000,
) -> Tuple[Any, Any, list]:
    """Build combined TF-IDF + behavioral features."""
    desc_col = "description" if "description" in train_df.columns else "merchant_raw"

    X_text_train, X_text_test, _ = build_text_features(
        train_df[desc_col], test_df[desc_col], max_features
    )
    X_behav_train, X_behav_test, num_cols = build_behavioral_features(
        train_df, test_df
    )

    # Combine sparse text + dense behavioral
    from scipy.sparse import csr_matrix
    X_train = sp_hstack([X_text_train, csr_matrix(X_behav_train)])
    X_test = sp_hstack([X_text_test, csr_matrix(X_behav_test)])
    return X_train, X_test, num_cols


# ============================================================================
# CATEGORIZATION MODELS
# ============================================================================

def train_majority(X_train, y_train, seed=42, **kwargs):
    """A0: Majority classifier."""
    model = DummyClassifier(strategy="most_frequent", random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_tfidf_lr(
    train_df: pd.DataFrame, y_train: np.ndarray, seed: int = 42,
    max_features: int = 10000, **kwargs,
) -> Tuple[Any, Any, Any]:
    """A1: TF-IDF + Logistic Regression. Returns (model, tfidf, label_encoder)."""
    desc_col = "description" if "description" in train_df.columns else "merchant_raw"
    tfidf = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, sublinear_tf=True, min_df=2,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=max_features // 2, sublinear_tf=True, min_df=2,
        )),
    ])
    X = tfidf.fit_transform(train_df[desc_col].fillna("").astype(str))
    model = LogisticRegression(
        max_iter=1000, C=4.0, class_weight="balanced",
        random_state=seed, solver="lbfgs", multi_class="multinomial",
    )
    model.fit(X, y_train)
    return model, tfidf, desc_col


def predict_tfidf_lr(model_tuple, test_df: pd.DataFrame):
    """Predict with TF-IDF LR model."""
    model, tfidf, desc_col = model_tuple
    X = tfidf.transform(test_df[desc_col].fillna("").astype(str))
    return model.predict(X)


def predict_proba_tfidf_lr(model_tuple, test_df: pd.DataFrame):
    """Predict probabilities with TF-IDF LR model."""
    model, tfidf, desc_col = model_tuple
    X = tfidf.transform(test_df[desc_col].fillna("").astype(str))
    return model.predict_proba(X)


def train_tfidf_svm(
    train_df: pd.DataFrame, y_train: np.ndarray, seed: int = 42,
    max_features: int = 10000, **kwargs,
) -> Tuple[Any, Any, Any]:
    """A2: TF-IDF + Linear SVM."""
    desc_col = "description" if "description" in train_df.columns else "merchant_raw"
    tfidf = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, sublinear_tf=True, min_df=2,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=max_features // 2, sublinear_tf=True, min_df=2,
        )),
    ])
    X = tfidf.fit_transform(train_df[desc_col].fillna("").astype(str))
    model = LinearSVC(
        max_iter=2000, C=1.0, class_weight="balanced",
        random_state=seed, dual="auto",
    )
    model.fit(X, y_train)
    return model, tfidf, desc_col


def predict_tfidf_svm(model_tuple, test_df: pd.DataFrame):
    """Predict with TF-IDF SVM."""
    model, tfidf, desc_col = model_tuple
    X = tfidf.transform(test_df[desc_col].fillna("").astype(str))
    return model.predict(X)


def train_random_forest_cat(
    train_df: pd.DataFrame, y_train: np.ndarray, seed: int = 42,
    n_estimators: int = 100, **kwargs,
) -> Tuple[Any, list]:
    """A3: Random Forest on behavioral features."""
    X_train, _, num_cols = build_combined_features(train_df, train_df)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, num_cols, train_df


def predict_rf_cat(model_tuple, test_df: pd.DataFrame):
    """Predict with Random Forest."""
    model, num_cols, ref_train = model_tuple
    _, X_test, _ = build_combined_features(ref_train, test_df)
    return model.predict(X_test)


def predict_proba_rf_cat(model_tuple, test_df: pd.DataFrame):
    """Predict probabilities with Random Forest."""
    model, num_cols, ref_train = model_tuple
    _, X_test, _ = build_combined_features(ref_train, test_df)
    return model.predict_proba(X_test)


def train_xgboost_cat(
    train_df: pd.DataFrame, y_train: np.ndarray, seed: int = 42,
    n_estimators: int = 100, **kwargs,
) -> Tuple[Any, Any, LabelEncoder, pd.DataFrame]:
    """A4: XGBoost Classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log("  XGBoost not available, falling back to Random Forest")
        return train_random_forest_cat(train_df, y_train, seed, n_estimators)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    X_train, _, num_cols = build_combined_features(train_df, train_df)

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
        random_state=seed, eval_metric="mlogloss",
        tree_method="hist", device=device,
        use_label_encoder=False,
    )
    model.fit(X_train, y_enc)
    return model, le, num_cols, train_df


def predict_xgb_cat(model_tuple, test_df: pd.DataFrame):
    """Predict with XGBoost."""
    if len(model_tuple) == 3:  # RF fallback
        return predict_rf_cat(model_tuple, test_df)
    model, le, num_cols, ref_train = model_tuple
    _, X_test, _ = build_combined_features(ref_train, test_df)
    y_enc = model.predict(X_test)
    return le.inverse_transform(y_enc)


def predict_proba_xgb_cat(model_tuple, test_df: pd.DataFrame):
    """Predict probabilities with XGBoost."""
    if len(model_tuple) == 3:  # RF fallback
        return predict_proba_rf_cat(model_tuple, test_df)
    model, le, num_cols, ref_train = model_tuple
    _, X_test, _ = build_combined_features(ref_train, test_df)
    return model.predict_proba(X_test)


def train_lightgbm_cat(
    train_df: pd.DataFrame, y_train: np.ndarray, seed: int = 42,
    n_estimators: int = 100, **kwargs,
) -> Tuple[Any, Any, Any, pd.DataFrame]:
    """A5: LightGBM Classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log("  LightGBM not available, falling back to Random Forest")
        return train_random_forest_cat(train_df, y_train, seed, n_estimators)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    X_train, _, num_cols = build_combined_features(train_df, train_df)

    model = LGBMClassifier(
        n_estimators=n_estimators, max_depth=8, learning_rate=0.1,
        random_state=seed, verbose=-1, num_leaves=31,
        class_weight="balanced",
    )
    model.fit(X_train, y_enc)
    return model, le, num_cols, train_df


def predict_lgbm_cat(model_tuple, test_df: pd.DataFrame):
    """Predict with LightGBM."""
    if len(model_tuple) == 3:  # RF fallback
        return predict_rf_cat(model_tuple, test_df)
    model, le, num_cols, ref_train = model_tuple
    _, X_test, _ = build_combined_features(ref_train, test_df)
    y_enc = model.predict(X_test)
    return le.inverse_transform(y_enc)


def predict_proba_lgbm_cat(model_tuple, test_df: pd.DataFrame):
    """Predict probabilities with LightGBM."""
    if len(model_tuple) == 3:  # RF fallback
        return predict_proba_rf_cat(model_tuple, test_df)
    model, le, num_cols, ref_train = model_tuple
    _, X_test, _ = build_combined_features(ref_train, test_df)
    return model.predict_proba(X_test)


# ============================================================================
# CATEGORIZATION MODEL REGISTRY
# ============================================================================

CAT_MODEL_REGISTRY = {
    "majority": {
        "train": lambda df, y, seed=42, **kw: (
            train_majority(
                build_text_features(
                    df["description" if "description" in df.columns else "merchant_raw"],
                    df["description" if "description" in df.columns else "merchant_raw"],
                )[0], y, seed
            ),
            None, "description" if "description" in df.columns else "merchant_raw"
        ),
        "predict": lambda mt, df: mt[0].predict(
            build_text_features(
                df["description" if "description" in df.columns else "merchant_raw"],
                df["description" if "description" in df.columns else "merchant_raw"],
            )[0]
        ),
        "has_proba": False,
    },
    "tfidf_lr": {
        "train": train_tfidf_lr,
        "predict": predict_tfidf_lr,
        "predict_proba": predict_proba_tfidf_lr,
        "has_proba": True,
    },
    "tfidf_svm": {
        "train": train_tfidf_svm,
        "predict": predict_tfidf_svm,
        "has_proba": False,
    },
    "random_forest": {
        "train": train_random_forest_cat,
        "predict": predict_rf_cat,
        "predict_proba": predict_proba_rf_cat,
        "has_proba": True,
    },
    "xgboost": {
        "train": train_xgboost_cat,
        "predict": predict_xgb_cat,
        "predict_proba": predict_proba_xgb_cat,
        "has_proba": True,
    },
    "lightgbm": {
        "train": train_lightgbm_cat,
        "predict": predict_lgbm_cat,
        "predict_proba": predict_proba_lgbm_cat,
        "has_proba": True,
    },
}


# ============================================================================
# FORECASTING MODELS
# ============================================================================

def _get_forecast_features(df: pd.DataFrame) -> list:
    """Get numeric feature columns for forecasting."""
    drop = {"user_id", "month", "category", "target"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def train_naive_previous(train_df: pd.DataFrame, seed=42, **kw):
    """F0a: Naive previous value."""
    return {"type": "naive_previous"}


def predict_naive_previous(model, test_df: pd.DataFrame):
    """Predict = lag_1."""
    if "lag_1" in test_df.columns:
        return np.clip(test_df["lag_1"].fillna(0).values, 0, None)
    return np.zeros(len(test_df))


def train_naive_rolling(train_df: pd.DataFrame, seed=42, **kw):
    """F0b: Rolling average."""
    return {"type": "naive_rolling"}


def predict_naive_rolling(model, test_df: pd.DataFrame):
    """Predict = rolling mean of last 3."""
    if "roll_mean_3" in test_df.columns:
        return np.clip(test_df["roll_mean_3"].fillna(0).values, 0, None)
    return np.zeros(len(test_df))


def train_naive_seasonal(train_df: pd.DataFrame, seed=42, **kw):
    """F0c: Seasonal naive (same month last year)."""
    # Build per-user-category-month lookup
    lookup = {}
    if "month" in train_df.columns and "user_id" in train_df.columns:
        for _, row in train_df.iterrows():
            key = (row["user_id"], row["category"], row["month"].month if hasattr(row["month"], "month") else 0)
            lookup[key] = row["target"]
    return {"type": "seasonal_naive", "lookup": lookup}


def predict_naive_seasonal(model, test_df: pd.DataFrame):
    """Predict using seasonal lookup, fallback to lag_1."""
    lookup = model.get("lookup", {})
    preds = []
    for _, row in test_df.iterrows():
        m = row["month"].month if hasattr(row.get("month", 0), "month") else 0
        key = (row["user_id"], row["category"], m)
        val = lookup.get(key, row.get("lag_1", 0))
        preds.append(max(0, val if val == val else 0))  # nan check
    return np.array(preds)


def train_linear_regression(train_df: pd.DataFrame, seed=42, **kw):
    """F1: Linear Regression."""
    feat_cols = _get_forecast_features(train_df)
    X = train_df[feat_cols].fillna(0).values
    y = train_df["target"].values
    model = LinearRegression()
    model.fit(X, y)
    return {"model": model, "feat_cols": feat_cols}


def predict_linear_regression(model_dict, test_df: pd.DataFrame):
    feat_cols = model_dict["feat_cols"]
    X = test_df[feat_cols].fillna(0).values
    return np.clip(model_dict["model"].predict(X), 0, None)


def train_rf_regressor(train_df: pd.DataFrame, seed=42, n_estimators=100, **kw):
    """F2: Random Forest Regressor."""
    feat_cols = _get_forecast_features(train_df)
    X = train_df[feat_cols].fillna(0).values
    y = train_df["target"].values
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=8,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X, y)
    importances = dict(zip(feat_cols, model.feature_importances_))
    return {"model": model, "feat_cols": feat_cols, "importances": importances}


def predict_rf_regressor(model_dict, test_df: pd.DataFrame):
    feat_cols = model_dict["feat_cols"]
    X = test_df[feat_cols].fillna(0).values
    return np.clip(model_dict["model"].predict(X), 0, None)


def train_xgb_regressor(train_df: pd.DataFrame, seed=42, n_estimators=100, **kw):
    """F3: XGBoost Regressor."""
    feat_cols = _get_forecast_features(train_df)
    X = train_df[feat_cols].fillna(0).values
    y = train_df["target"].values

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=6,
            learning_rate=0.1, random_state=seed,
            tree_method="hist",
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators, max_depth=6,
            learning_rate=0.1, random_state=seed,
        )

    model.fit(X, y)
    importances = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feat_cols, model.feature_importances_))
    return {"model": model, "feat_cols": feat_cols, "importances": importances}


def predict_xgb_regressor(model_dict, test_df: pd.DataFrame):
    feat_cols = model_dict["feat_cols"]
    X = test_df[feat_cols].fillna(0).values
    return np.clip(model_dict["model"].predict(X), 0, None)


def train_lgbm_regressor(train_df: pd.DataFrame, seed=42, n_estimators=100, **kw):
    """F4: LightGBM Regressor."""
    feat_cols = _get_forecast_features(train_df)
    X = train_df[feat_cols].fillna(0).values
    y = train_df["target"].values

    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=n_estimators, max_depth=8,
            learning_rate=0.1, random_state=seed,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators, max_depth=8,
            learning_rate=0.1, random_state=seed,
        )

    model.fit(X, y)
    importances = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feat_cols, model.feature_importances_))
    return {"model": model, "feat_cols": feat_cols, "importances": importances}


def predict_lgbm_regressor(model_dict, test_df: pd.DataFrame):
    feat_cols = model_dict["feat_cols"]
    X = test_df[feat_cols].fillna(0).values
    return np.clip(model_dict["model"].predict(X), 0, None)


# ============================================================================
# FORECASTING MODEL REGISTRY
# ============================================================================

FORECAST_MODEL_REGISTRY = {
    "naive_previous": {
        "train": train_naive_previous,
        "predict": predict_naive_previous,
    },
    "naive_rolling": {
        "train": train_naive_rolling,
        "predict": predict_naive_rolling,
    },
    "naive_seasonal": {
        "train": train_naive_seasonal,
        "predict": predict_naive_seasonal,
    },
    "linear_regression": {
        "train": train_linear_regression,
        "predict": predict_linear_regression,
    },
    "random_forest": {
        "train": train_rf_regressor,
        "predict": predict_rf_regressor,
    },
    "xgboost": {
        "train": train_xgb_regressor,
        "predict": predict_xgb_regressor,
    },
    "lightgbm": {
        "train": train_lgbm_regressor,
        "predict": predict_lgbm_regressor,
    },
}
