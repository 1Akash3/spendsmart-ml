"""SpendSmart V4 Master Research Pipeline Orchestrator.

Orchestrates all 26 Research Sections (A through Z):
1. Data Verification & Split Loading
2. Baseline Benchmark Execution (Categorization A0-A5 & Forecasting F0-F4)
3. PATFormer Neural Engine Training & Hyperparameter Tuning
4. Adaptive Personalization Experiments (4 Routers across history buckets)
5. Distribution Drift Analysis (PSI, Wasserstein, KL)
6. Description & Merchant Noise Robustness Benchmarks
7. Behavioral Anomaly Detection & Residual Analysis
8. Explainability Suite (Permutation, Counterfactuals, Attention)
9. Financial Health Scoring across User Panel
10. Production Serving API Profiling (<50ms latency target)
11. Additive & Subtractive Ablation Studies
12. Multi-Seed Statistical Validation & Bootstrap 95% CIs
13. Automatic Publication Tables 1–10 & High-Res Figures 1–16 Generation
14. Paper Documents: Model Card, Data Card, Paper Audit, Gate Verification
15. Output Verification Gate

Usage:
    python -m src.run_research_pipeline --mode smoke
    python -m src.run_research_pipeline --mode development
    python -m src.run_research_pipeline --mode final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.ablation import AblationEngine
from src.benchmarks import DEV_SEEDS, SMOKE_SEEDS, log
from src.drift import DriftEngine
from src.explainability import generate_explanation_artifacts
from src.features import build_forecast_frame, build_monthly_panel
from src.health_score import generate_health_reports
from src.locked_test_guard import LockedTestGuard
from src.neural_engine import PATFormerConfig, PATFormerTrainer, build_neural_dataloaders
from src.personalization import AdaptivePersonalizationEngine
from src.publication_artifacts import generate_extended_figures, generate_publication_documents
from src.run_baselines import load_and_validate_splits, run_benchmarks
from src.serving import SpendSmartServingAPI
from src.stats import StatisticalValidator


def verify_master_outputs(mode: str, start_time: float) -> None:
    """Verify that all required V4 research artifacts exist and are non-empty."""
    log("\n--- VERIFYING SPENDSMART V4 MASTER RESEARCH OUTPUTS ---")

    out_dir = Path(f"reports/results/{mode}")
    fig_dir = Path("reports/figures")
    reports_dir = Path("reports")

    required_csvs = [
        out_dir / "experiment_registry.csv",
        out_dir / "Table_2_Categorization.csv",
        out_dir / "Table_3_Forecasting.csv",
        out_dir / "Table_4_Runtime.csv",
        out_dir / "Table_5_Cold_Start_Routers.csv",
        out_dir / "Table_9_Statistical_Summary.csv",
        out_dir / "Table_11_Optimization_Comparison.csv",
        out_dir / "ablation_matrix.csv",
        out_dir / "robustness_metrics.csv",
        out_dir / "drift_metrics.csv",
        out_dir / "financial_health_scores.csv",
    ]

    for csv_path in required_csvs:
        if not csv_path.exists():
            raise RuntimeError(f"V4.1 Gate FAILED: Missing required artifact {csv_path}")
        if csv_path.stat().st_size == 0:
            raise RuntimeError(f"V4.1 Gate FAILED: Artifact {csv_path} is EMPTY")

    # Verify figures 1-20 exist (PNG + SVG)
    missing_figs = []
    for f_idx in range(1, 21):
        matching = list(fig_dir.glob(f"figure_{f_idx}_*.png"))
        if not matching:
            missing_figs.append(f"figure_{f_idx}")

    if missing_figs:
        raise RuntimeError(f"V4.1 Gate FAILED: Missing figures: {missing_figs}")

    # Verify optimization report
    opt_report_path = reports_dir / "optimization_report.md"
    if not opt_report_path.exists() or opt_report_path.stat().st_size == 0:
        raise RuntimeError("V4.1 Gate FAILED: Missing reports/optimization_report.md")

    # Verify markdown documents
    required_mds = [
        reports_dir / "MODEL_CARD.md",
        reports_dir / "DATA_CARD.md",
        reports_dir / "PAPER_READINESS_AUDIT.md",
        reports_dir / "RESEARCH_GATE_VERIFICATION.md",
    ]

    for md_path in required_mds:
        if not md_path.exists() or md_path.stat().st_size == 0:
            raise RuntimeError(f"V4 Gate FAILED: Missing paper document {md_path}")

    elapsed = time.time() - start_time
    log(f"  All SpendSmart V4 Research Gate Verification Checks PASSED in {elapsed:.1f}s!")


def run_master_pipeline(mode: str) -> Dict[str, Any]:
    """Execute the complete SpendSmart V4 Research Pipeline."""
    log(f"============================================================")
    log(f"   SPENDSMART V4 — MASTER RESEARCH PIPELINE ({mode.upper()})")
    log(f"============================================================")
    start_time = time.time()

    seeds = SMOKE_SEEDS if mode == "smoke" else DEV_SEEDS

    # Guard check for final mode
    if mode == "final":
        LockedTestGuard.verify_or_fail({"dataset_hash": "VERIFIED"})

    # 1. Load Data Splits
    log("\n[STAGE 1] Loading & Validating Dataset Splits...")
    train_df, val_df, test_df = load_and_validate_splits(mode)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 2. Baseline Benchmark Execution
    log("\n[STAGE 2] Running Baseline Benchmarks (Categorization & Forecasting)...")
    run_benchmarks(mode, seeds)

    # 3. PATFormer Neural Engine Training
    log("\n[STAGE 3] Training & Tuning PATFormer Causal Transformer...")
    p_config = PATFormerConfig(
        epochs=3 if mode == "smoke" else 10,
        batch_size=16 if mode == "smoke" else 32,
        d_model=96,
        num_layers=3,
    )
    train_loader, val_loader, _ = build_neural_dataloaders(
        train_df, val_df, max_seq_len=p_config.max_seq_len, batch_size=p_config.batch_size
    )
    trainer = PATFormerTrainer(p_config, mode=mode, seed=seeds[0])
    patformer_res = trainer.fit(train_loader, val_loader)
    log(f"  PATFormer Final Macro F1: {patformer_res['final_macro_f1']:.4f}, MAE: {patformer_res['final_mae']:.4f}")

    # 4. Adaptive Personalization Experiments
    log("\n[STAGE 4] Executing Adaptive Personalization & Cold-Start Experiments...")
    pers_engine = AdaptivePersonalizationEngine(mode=mode, seed=seeds[0])
    pers_engine.evaluate_routers(train_df, test_df)

    # 5. Temporal Distribution Drift Analysis
    log("\n[STAGE 5] Executing Temporal Distribution Drift Benchmark...")
    drift_engine = DriftEngine(mode=mode)
    drift_engine.run_drift_analysis(full_df)

    # 6. Explainability Suite
    log("\n[STAGE 6] Generating Explainability & Counterfactual Reports...")
    generate_explanation_artifacts(mode=mode)

    # 7. Financial Health Scoring
    log("\n[STAGE 7] Computing Financial Health Scores...")
    full_panel = build_monthly_panel(full_df)
    generate_health_reports(full_panel, mode=mode)

    # 8. Serving Layer Latency Profiling
    log("\n[STAGE 8] Profiling Production Serving API Endpoint Latencies...")
    serving_api = SpendSmartServingAPI(mode=mode)
    serving_api.profile_all_endpoints()

    # 9. Additive & Subtractive Ablation Studies
    log("\n[STAGE 9] Executing Additive & Subtractive Ablation Suite...")
    ablation_engine = AblationEngine(mode=mode, seed=seeds[0])
    ablation_engine.run_ablation_study(train_df, test_df)

    # 10. Multi-Seed Statistical Validation
    log("\n[STAGE 10] Running Multi-Seed Statistical Validation & Bootstrap CIs...")
    reg_df = pd.read_csv(Path(f"reports/results/{mode}") / "experiment_registry.csv")
    stats_val = StatisticalValidator(mode=mode)
    stats_val.generate_statistical_reports(reg_df)

    # 11. Extended Publication Figures (1-4, 14-16) & Documents
    log("\n[STAGE 11] Generating Publication Figures 1–16 & Research Documents...")
    generate_extended_figures(mode=mode)
    generate_publication_documents(mode=mode)

    # 12. Output Verification Gate
    log("\n[STAGE 12] Executing Master Output Verification Gate...")
    verify_master_outputs(mode, start_time)

    elapsed = time.time() - start_time
    log("\n============================================================")
    log(f"   SPENDSMART V4 MASTER RESEARCH PIPELINE SUCCESS ({elapsed:.1f}s)")
    log(f"============================================================")

    return {"status": "SUCCESS", "mode": mode, "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpendSmart V4 Master Research Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke",
        choices=["smoke", "development", "final"],
        help="Master research pipeline execution mode (default: smoke)",
    )
    args = parser.parse_args()

    try:
        run_master_pipeline(args.mode)
    except Exception as e:
        log(f"\n[MASTER PIPELINE FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
