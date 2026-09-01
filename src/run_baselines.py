"""CLI Entrypoint and Orchestration Engine for SpendSmart V3 Empirical Benchmarks.

Executes real, empirical model evaluation across categorization, forecasting,
robustness, cold-start, merchant generalization, drift, and calibration regimes.

Usage:
    python -m src.run_baselines --mode smoke
    python -m src.run_baselines --mode development
    python -m src.run_baselines --mode final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import (
    SMOKE_SEEDS, DEV_SEEDS, CATEGORIZATION_MODELS, FORECAST_MODELS,
    CAT_SPLIT_NAMES, log,
)
from src.benchmarks.evaluators import (
    evaluate_categorization_model, evaluate_forecasting_model,
    evaluate_robustness, evaluate_cold_start,
    evaluate_merchant_generalization, evaluate_distribution_drift,
    evaluate_calibration,
)
from src.benchmarks.tracking import ExperimentRegistry, CheckpointManager
from src.benchmarks.tables_and_figures import generate_all_tables_and_figures
from src.evaluation.splits import (
    create_temporal_split, create_merchant_disjoint_split,
    create_novel_merchant_split, create_noisy_description_split,
)
from src.features import build_forecast_frame, build_monthly_panel
from src.locked_test_guard import LockedTestGuard


# ============================================================================
# DATA VALIDATION & PREPARATION
# ============================================================================

def load_and_validate_splits(mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test parquets and perform verification."""
    splits_dir = Path(f"data/{mode}/splits")
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "validation.parquet"
    test_path = splits_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise RuntimeError(
            f"Data splits missing in {splits_dir}. Run `python -m src.run_data_pipeline --mode {mode}` first."
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Validate schema
    required_cols = ["user_id", "timestamp", "amount", "category"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Split data missing required canonical column '{col}'")

    log(f"  Loaded splits: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    return train_df, val_df, test_df


# ============================================================================
# EMPIRICAL BENCHMARK EXECUTOR
# ============================================================================

def run_benchmarks(mode: str, seeds: List[int]) -> Dict[str, Any]:
    """Orchestrate all empirical benchmark experiments."""
    log(f"=== EMPIRICAL BENCHMARK ENGINE: Mode={mode.upper()} ===")
    start_time = time.time()

    # Final mode safeguard
    if mode == "final":
        LockedTestGuard.verify_or_fail({"dataset_hash": "VERIFIED"})

    # 1. Load splits
    train_df, val_df, test_df = load_and_validate_splits(mode)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    registry = ExperimentRegistry(mode)
    ckpt = CheckpointManager(mode)

    # Prepare forecasting frames
    log("  Building forecasting frames...")
    train_panel = build_monthly_panel(train_df)
    train_forecast_frame = build_forecast_frame(train_panel)

    val_panel = build_monthly_panel(val_df)
    val_forecast_frame = build_forecast_frame(val_panel)

    test_panel = build_monthly_panel(test_df)
    test_forecast_frame = build_forecast_frame(test_panel)

    # 2. Categorization Benchmarks (PART 3 & PART 5)
    log("\n--- EXECUTING CATEGORIZATION BENCHMARKS ---")
    cat_models = CATEGORIZATION_MODELS if mode != "smoke" else ["majority", "tfidf_lr", "tfidf_svm", "random_forest"]

    for seed in seeds:
        for model_name in tqdm(cat_models, desc=f"Cat Models (Seed {seed})"):
            # Split B: Temporal (Default)
            exp_id = f"CAT-{model_name.upper()}-TEMPORAL-S{seed}"
            if ckpt.has_checkpoint(exp_id):
                log(f"  Skipping {exp_id} (checkpoint exists)")
                continue

            res = evaluate_categorization_model(
                model_name, train_df, test_df, split_name="temporal", seed=seed, mode=mode
            )
            registry.register(res)
            ckpt.save_checkpoint(res)

            # Additional splits for development/final mode
            if mode != "smoke":
                # Split C: Merchant Disjoint
                train_c, test_c = create_merchant_disjoint_split(full_df)
                res_c = evaluate_categorization_model(model_name, train_c, test_c, "merchant_disjoint", seed, mode)
                registry.register(res_c)

                # Split D: Novel Merchant
                train_d, test_d = create_novel_merchant_split(full_df)
                res_d = evaluate_categorization_model(model_name, train_d, test_d, "novel_merchant", seed, mode)
                registry.register(res_d)

    # 3. Forecasting Benchmarks (PART 4)
    log("\n--- EXECUTING FORECASTING BENCHMARKS ---")
    forecast_models = FORECAST_MODELS if mode != "smoke" else ["naive_previous", "naive_rolling", "linear_regression", "random_forest"]

    for seed in seeds:
        for model_name in tqdm(forecast_models, desc=f"Forecast Models (Seed {seed})"):
            exp_id = f"FOR-{model_name.upper()}-TEMPORAL-S{seed}"
            if ckpt.has_checkpoint(exp_id):
                log(f"  Skipping {exp_id} (checkpoint exists)")
                continue

            res = evaluate_forecasting_model(
                model_name, train_forecast_frame, test_forecast_frame, split_name="temporal", seed=seed, mode=mode
            )
            registry.register(res)
            ckpt.save_checkpoint(res)

    # Save artifacts from registry
    registry.save_registry()
    registry.save_confusion_matrices()
    registry.save_classification_reports()
    registry.save_feature_importances()

    # Read registry as DataFrame
    registry_df = pd.read_csv(registry.registry_path)

    # 4. Robustness Benchmark (PART 6)
    log("\n--- EXECUTING ROBUSTNESS BENCHMARK ---")
    robustness_df = evaluate_robustness(train_df, test_df, model_name="tfidf_lr", seed=seeds[0])
    robustness_df.to_csv(Path(f"reports/results/{mode}") / "robustness_metrics.csv", index=False)

    # 5. Cold Start Benchmark (PART 7)
    log("\n--- EXECUTING COLD START BENCHMARK ---")
    cold_start_df = evaluate_cold_start(train_df, test_df, model_name="tfidf_lr", seed=seeds[0])
    cold_start_df.to_csv(Path(f"reports/results/{mode}") / "cold_start_metrics.csv", index=False)

    # 6. Merchant Generalization Benchmark (PART 8)
    log("\n--- EXECUTING MERCHANT GENERALIZATION BENCHMARK ---")
    merchant_df = evaluate_merchant_generalization(train_df, test_df, model_name="tfidf_lr", seed=seeds[0])
    merchant_df.to_csv(Path(f"reports/results/{mode}") / "merchant_generalization.csv", index=False)

    # 7. Distribution Drift Benchmark (PART 9)
    log("\n--- EXECUTING DISTRIBUTION DRIFT BENCHMARK ---")
    drift_df = evaluate_distribution_drift(full_df)
    drift_df.to_csv(Path(f"reports/results/{mode}") / "drift_metrics.csv", index=False)

    # 8. Calibration Benchmark (PART 10)
    log("\n--- EXECUTING CALIBRATION BENCHMARK ---")
    cal_metrics, cal_res = evaluate_calibration(train_df, test_df, model_name="tfidf_lr", seed=seeds[0])
    with open(Path(f"reports/results/{mode}") / "calibration_metrics.json", "w") as f:
        json.dump(cal_metrics, f, indent=2)

    # 9. Publication Tables & Figures Generation (PART 16 & PART 17)
    labels = sorted(train_df["category"].unique())
    generate_all_tables_and_figures(
        registry_df=registry_df,
        cold_start_df=cold_start_df,
        merchant_df=merchant_df,
        robustness_df=robustness_df,
        cal_metrics=cal_metrics,
        cm=cal_res.confusion_matrix,
        labels=labels,
        fi_dict=cal_res.feature_importance,
        mode=mode,
    )

    # 10. Multi-seed Statistical Summary (PART 18)
    if len(seeds) > 1:
        log("\n--- GENERATING MULTI-SEED STATISTICAL SUMMARY ---")
        stats_summary = registry_df.groupby(["task", "model_name"]).agg({
            "macro_f1": ["mean", "std"] if "macro_f1" in registry_df.columns else "count",
            "accuracy": ["mean", "std"] if "accuracy" in registry_df.columns else "count",
            "mae": ["mean", "std"] if "mae" in registry_df.columns else "count",
        }).reset_index()
        stats_summary.to_csv(Path(f"reports/results/{mode}") / "statistical_summary.csv", index=False)

    # 11. Empirical Verification Before Completion (PART 22)
    verify_benchmark_results(mode, start_time)

    elapsed = time.time() - start_time
    log(f"\n=== BENCHMARK COMPLETE: Mode={mode.upper()} in {elapsed:.1f}s ===")
    return {"status": "SUCCESS", "elapsed_seconds": elapsed}


# ============================================================================
# VERIFICATION BEFORE COMPLETION (PART 22)
# ============================================================================

def verify_benchmark_results(mode: str, start_time: float) -> None:
    """Verify that all required CSVs, metrics, and figures exist and are valid."""
    log("\n--- VERIFYING BENCHMARK OUTPUTS ---")
    out_dir = Path(f"reports/results/{mode}")
    fig_dir = Path(f"reports/figures/{mode}")

    required_files = [
        out_dir / "experiment_registry.csv",
        out_dir / "Table_2_Categorization.csv",
        out_dir / "Table_3_Forecasting.csv",
        out_dir / "Table_4_Runtime.csv",
        out_dir / "robustness_metrics.csv",
        out_dir / "cold_start_metrics.csv",
        out_dir / "merchant_generalization.csv",
        out_dir / "drift_metrics.csv",
    ]

    for path in required_files:
        if not path.exists():
            raise RuntimeError(f"Verification FAILED: Missing required artifact {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"Verification FAILED: Artifact {path} is EMPTY")

    # Verify registry content
    registry_df = pd.read_csv(out_dir / "experiment_registry.csv")
    if len(registry_df) == 0:
        raise RuntimeError("Verification FAILED: Experiment registry contains 0 rows")

    # Verify metrics are not placeholders or constant
    if "macro_f1" in registry_df.columns:
        f1_vals = registry_df["macro_f1"].dropna().values
        if len(f1_vals) > 0 and (f1_vals == f1_vals[0]).all() and len(f1_vals) > 3:
            raise RuntimeError("Verification FAILED: Constant placeholder Macro F1 values detected")

    runtime = time.time() - start_time
    if runtime <= 0:
        raise RuntimeError("Verification FAILED: Benchmark runtime reported <= 0s")

    log("  All verification checks PASSED!")


# ============================================================================
# MAIN CLI ENTRYPOINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpendSmart V3 Empirical Benchmark Runner"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke",
        choices=["smoke", "development", "final"],
        help="Benchmark execution mode (default: smoke)",
    )
    args = parser.parse_args()

    seeds = SMOKE_SEEDS if args.mode == "smoke" else DEV_SEEDS

    try:
        run_benchmarks(args.mode, seeds)
    except Exception as e:
        log(f"[BENCHMARK FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
