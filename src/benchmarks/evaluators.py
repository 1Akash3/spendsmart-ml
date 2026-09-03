"""Benchmark evaluators for all experimental regimes.

Implements evaluation protocols for:
- Standard splits (Random, Temporal, Merchant-Disjoint, Novel-Merchant, Cross-Source, Noisy-Description)
- Robustness (5%, 10%, 20% corruption)
- Cold-Start (0-5, 6-20, 21-50, 51-100, 100+ txs)
- Long-Tail Merchant Generalization (Head, Torso, Tail)
- Distribution Drift (PSI, Wasserstein, KL Divergence)
- Calibration (ECE, Brier)
- Forecasting (F0-F4 / ARIMA)
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import wasserstein_distance, entropy

from src.benchmarks import (
    BenchmarkResult, ExperimentMeta, RuntimeInfo, MemoryTracker, Timer,
    make_experiment_id, get_device_str, get_git_commit, load_dataset_hash,
    hash_config, log, COLD_START_BUCKETS, NOISE_LEVELS,
)
from src.benchmarks.metrics import (
    compute_categorization_metrics, compute_forecast_metrics,
    compute_calibration_metrics,
)
from src.benchmarks.models import (
    CAT_MODEL_REGISTRY, FORECAST_MODEL_REGISTRY,
    build_text_features, build_behavioral_features, build_combined_features,
)
from src.evaluation.splits import (
    create_temporal_split, create_merchant_disjoint_split,
    create_novel_merchant_split, create_noisy_description_split,
)


# ============================================================================
# 1. Categorization Evaluator Across Splits
# ============================================================================

def evaluate_categorization_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_name: str,
    seed: int = 42,
    mode: str = "smoke",
) -> BenchmarkResult:
    """Train and evaluate a categorization model on a specific split."""
    exp_id = make_experiment_id("categorization", model_name, split_name, seed)
    log(f"  Running Categorization: {exp_id} ({model_name} on {split_name})")

    spec = CAT_MODEL_REGISTRY[model_name]
    train_func = spec["train"]
    predict_func = spec["predict"]
    has_proba = spec.get("has_proba", False)

    y_train = train_df["category"].values
    y_test = test_df["category"].values

    runtime = RuntimeInfo()

    # Measure training time and memory
    with MemoryTracker() as mem_track:
        with Timer() as t_train:
            model_obj = train_func(train_df, y_train, seed=seed)
        runtime.training_seconds = round(t_train.elapsed, 4)
        runtime.peak_ram_mb = round(mem_track.peak_mb, 2)

    # Measure inference latency
    n_test = len(test_df)
    with Timer() as t_inf:
        y_pred = predict_func(model_obj, test_df)
    runtime.inference_ms = round((t_inf.elapsed / max(1, n_test)) * 1000.0, 4)

    # Probabilities if supported
    y_proba = None
    if has_proba and "predict_proba" in spec:
        try:
            y_proba = spec["predict_proba"](model_obj, test_df)
        except Exception:
            y_proba = None

    metrics = compute_categorization_metrics(y_test, y_pred, y_proba)
    cm = metrics.pop("confusion_matrix")
    report = metrics.pop("classification_report")

    # Feature importances if available
    fi = None
    if hasattr(model_obj, "feature_importances_"):
        fi = dict(zip(range(len(model_obj.feature_importances_)), model_obj.feature_importances_))

    meta = ExperimentMeta(
        experiment_id=exp_id,
        model_name=model_name,
        task="categorization",
        split_name=split_name,
        seed=seed,
        mode=mode,
        git_commit=get_git_commit(),
        dataset_hash=load_dataset_hash(mode),
        config_hash=hash_config({"model": model_name, "split": split_name, "seed": seed}),
        device=get_device_str(),
        runtime_seconds=round(runtime.training_seconds, 4),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status="COMPLETED",
    )

    return BenchmarkResult(
        meta=meta,
        metrics=metrics,
        confusion_matrix=cm,
        classification_report=report,
        feature_importance=fi,
        runtime=runtime,
        predictions=y_pred,
        probabilities=y_proba,
    )


# ============================================================================
# 2. Forecasting Evaluator
# ============================================================================

def evaluate_forecasting_model(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_name: str = "temporal",
    seed: int = 42,
    mode: str = "smoke",
) -> BenchmarkResult:
    """Train and evaluate a forecasting model."""
    exp_id = make_experiment_id("forecasting", model_name, split_name, seed)
    log(f"  Running Forecasting: {exp_id} ({model_name} on {split_name})")

    spec = FORECAST_MODEL_REGISTRY[model_name]
    train_func = spec["train"]
    predict_func = spec["predict"]

    y_test = test_frame["target"].values if "target" in test_frame.columns else test_frame["amount"].values

    runtime = RuntimeInfo()

    with MemoryTracker() as mem_track:
        with Timer() as t_train:
            model_obj = train_func(train_frame, seed=seed)
        runtime.training_seconds = round(t_train.elapsed, 4)
        runtime.peak_ram_mb = round(mem_track.peak_mb, 2)

    n_test = len(test_frame)
    with Timer() as t_inf:
        y_pred = predict_func(model_obj, test_frame)
    runtime.inference_ms = round((t_inf.elapsed / max(1, n_test)) * 1000.0, 4)

    metrics = compute_forecast_metrics(y_test, y_pred)

    fi = model_obj.get("importances") if isinstance(model_obj, dict) else None

    meta = ExperimentMeta(
        experiment_id=exp_id,
        model_name=model_name,
        task="forecasting",
        split_name=split_name,
        seed=seed,
        mode=mode,
        git_commit=get_git_commit(),
        dataset_hash=load_dataset_hash(mode),
        config_hash=hash_config({"model": model_name, "split": split_name, "seed": seed}),
        device=get_device_str(),
        runtime_seconds=round(runtime.training_seconds, 4),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status="COMPLETED",
    )

    return BenchmarkResult(
        meta=meta,
        metrics=metrics,
        feature_importance=fi,
        runtime=runtime,
        predictions=y_pred,
    )


# ============================================================================
# 3. Robustness Benchmark (Noise Injection)
# ============================================================================

def evaluate_robustness(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "tfidf_lr",
    noise_levels: List[float] = NOISE_LEVELS,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate performance degradation under 5%, 10%, 20% description/merchant noise."""
    log(f"  Running Robustness Benchmark for {model_name}...")
    results = []

    # Clean baseline
    clean_res = evaluate_categorization_model(model_name, train_df, test_df, "clean", seed=seed)
    results.append({
        "noise_level": 0.0,
        "macro_f1": clean_res.metrics["macro_f1"],
        "accuracy": clean_res.metrics["accuracy"],
        "degradation_f1": 0.0,
    })

    base_f1 = clean_res.metrics["macro_f1"]

    for noise_pct in noise_levels:
        corrupted_test = test_df.copy()
        desc_col = "description" if "description" in corrupted_test.columns else "merchant_raw"

        np.random.seed(seed)
        mask = np.random.random(len(corrupted_test)) < noise_pct

        def _corrupt(text):
            if not isinstance(text, str):
                return ""
            r = np.random.random()
            if r < 0.3:
                return text.lower()
            elif r < 0.6:
                return text.upper()
            else:
                return text.replace(" ", "") + " TX"

        corrupted_test.loc[mask, desc_col] = corrupted_test.loc[mask, desc_col].apply(_corrupt)

        noisy_res = evaluate_categorization_model(
            model_name, train_df, corrupted_test, f"noise_{int(noise_pct*100)}", seed=seed
        )
        current_f1 = noisy_res.metrics["macro_f1"]

        results.append({
            "noise_level": noise_pct,
            "macro_f1": current_f1,
            "accuracy": noisy_res.metrics["accuracy"],
            "degradation_f1": round(base_f1 - current_f1, 4),
        })

    return pd.DataFrame(results)


# ============================================================================
# 4. Cold-Start Benchmark
# ============================================================================

def evaluate_cold_start(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "tfidf_lr",
    buckets: List[Tuple[int, int]] = COLD_START_BUCKETS,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate categorization Macro F1 across user history length buckets."""
    log(f"  Running Cold-Start Benchmark for {model_name}...")

    # Calculate user tx count in training set
    train_user_counts = train_df["user_id"].value_counts().to_dict()

    results = []
    for low, high in buckets:
        # Filter test users whose training history length falls into [low, high]
        target_users = {
            u for u, cnt in train_user_counts.items() if low <= cnt <= high
        }
        # Also include unseen users if low == 0
        if low == 0:
            test_users = set(test_df["user_id"].unique())
            unseen = test_users - set(train_user_counts.keys())
            target_users.update(unseen)

        sub_test = test_df[test_df["user_id"].isin(target_users)]

        if len(sub_test) > 5:
            res = evaluate_categorization_model(
                model_name, train_df, sub_test, f"cold_{low}_{high}", seed=seed
            )
            macro_f1 = res.metrics["macro_f1"]
            acc = res.metrics["accuracy"]
            count = len(sub_test)
        else:
            macro_f1 = 0.0
            acc = 0.0
            count = len(sub_test)

        bucket_label = f"{low}-{high}" if high < 99999 else "100+"
        results.append({
            "history_bucket": bucket_label,
            "user_count": len(target_users),
            "test_sample_count": count,
            "macro_f1": macro_f1,
            "accuracy": acc,
        })

    return pd.DataFrame(results)


# ============================================================================
# 5. Long-Tail Merchant Benchmark
# ============================================================================

def evaluate_merchant_generalization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "tfidf_lr",
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate performance across Head, Torso, and Tail merchants."""
    log(f"  Running Long-Tail Merchant Benchmark for {model_name}...")

    merch_counts = train_df["merchant_key"].value_counts()
    q75 = merch_counts.quantile(0.75)
    q25 = merch_counts.quantile(0.25)

    head_merchants = set(merch_counts[merch_counts >= q75].index)
    torso_merchants = set(merch_counts[(merch_counts >= q25) & (merch_counts < q75)].index)
    tail_merchants = set(merch_counts[merch_counts < q25].index)

    results = []
    for group_name, m_set in [("Head", head_merchants), ("Torso", torso_merchants), ("Tail", tail_merchants)]:
        sub_test = test_df[test_df["merchant_key"].isin(m_set)]
        if len(sub_test) > 5:
            res = evaluate_categorization_model(
                model_name, train_df, sub_test, f"merchant_{group_name.lower()}", seed=seed
            )
            macro_f1 = res.metrics["macro_f1"]
            acc = res.metrics["accuracy"]
        else:
            macro_f1 = 0.0
            acc = 0.0

        results.append({
            "merchant_tier": group_name,
            "test_samples": len(sub_test),
            "macro_f1": macro_f1,
            "accuracy": acc,
        })

    return pd.DataFrame(results)


# ============================================================================
# 6. Distribution Drift Benchmark
# ============================================================================

def evaluate_distribution_drift(
    df: pd.DataFrame,
    mode: str = "smoke",
) -> pd.DataFrame:
    """Compute temporal drift metrics: Population Stability Index (PSI), Wasserstein Distance, KL Divergence."""
    from src.drift import DriftEngine
    engine = DriftEngine(mode=mode)
    drift_df, _ = engine.run_drift_analysis(df)
    return drift_df


# ============================================================================
# 7. Calibration Benchmark
# ============================================================================

def evaluate_calibration(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "tfidf_lr",
    seed: int = 42,
) -> Tuple[Dict[str, Any], BenchmarkResult]:
    """Evaluate ECE and Brier score for probabilistic classifiers."""
    log(f"  Running Calibration Benchmark for {model_name}...")
    res = evaluate_categorization_model(model_name, train_df, test_df, "calibration", seed=seed)

    if res.probabilities is not None:
        cal_metrics = compute_calibration_metrics(
            test_df["category"].values, res.probabilities
        )
    else:
        cal_metrics = {
            "ece": 0.0,
            "brier_score": 0.0,
            "reliability_curve": {"bin_accuracies": [], "bin_confidences": [], "bin_counts": []},
        }

    return cal_metrics, res
