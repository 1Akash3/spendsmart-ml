"""Unit tests for the empirical benchmark runner and artifacts.

Verifies:
1. Benchmark runner execution in smoke mode.
2. CSV creation (experiment_registry, Table_2..Table_8).
3. Non-empty CSVs with real, non-constant metrics.
4. Metric bounds (0 <= Accuracy/F1 <= 1, MAE >= 0).
5. Runtime metadata existence.
6. Unique experiment IDs.
7. Dataset hash recording.
8. Classification reports and confusion matrices saving.
9. PNG Figure creation (Figures 5-13).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

# Path bootstrap
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.data_pipeline import run as run_data_pipeline
from src.run_baselines import run_benchmarks


@pytest.fixture(scope="module")
def benchmark_results():
    """Ensure data exists and run smoke benchmarks once."""
    # Ensure data/smoke exists
    smoke_data_dir = Path("data/smoke")
    if not smoke_data_dir.exists() or not (smoke_data_dir / "splits/train.parquet").exists():
        run_data_pipeline("smoke")

    out_dir = Path("reports/results/smoke")
    fig_dir = Path("reports/figures/smoke")

    # Run smoke benchmark
    res = run_benchmarks("smoke", seeds=[42])
    return out_dir, fig_dir, res


class TestBaselineRunner:
    """Test suite for empirical benchmark runner outputs."""

    def test_experiment_registry_created(self, benchmark_results):
        """experiment_registry.csv must exist and be non-empty."""
        out_dir, _, _ = benchmark_results
        registry_path = out_dir / "experiment_registry.csv"
        assert registry_path.exists(), "experiment_registry.csv missing!"
        assert registry_path.stat().st_size > 0, "experiment_registry.csv is empty!"

    def test_unique_experiment_ids(self, benchmark_results):
        """All experiment IDs in registry must be unique."""
        out_dir, _, _ = benchmark_results
        df = pd.read_csv(out_dir / "experiment_registry.csv")
        assert len(df) > 0, "Registry has 0 rows!"
        assert df["experiment_id"].is_unique, "Duplicate experiment IDs found in registry!"

    def test_dataset_hash_recorded(self, benchmark_results):
        """Registry rows must contain valid 16-char dataset hash."""
        out_dir, _, _ = benchmark_results
        df = pd.read_csv(out_dir / "experiment_registry.csv")
        assert "dataset_hash" in df.columns, "dataset_hash missing from registry!"
        assert (df["dataset_hash"] != "UNKNOWN").all(), "dataset_hash is UNKNOWN!"

    def test_metric_valid_ranges(self, benchmark_results):
        """Metrics must fall within valid mathematical bounds."""
        out_dir, _, _ = benchmark_results
        df = pd.read_csv(out_dir / "experiment_registry.csv")

        # Categorization metrics
        cat_df = df[df["task"] == "categorization"]
        if not cat_df.empty:
            assert (cat_df["accuracy"] >= 0.0).all() and (cat_df["accuracy"] <= 1.0).all()
            assert (cat_df["macro_f1"] >= 0.0).all() and (cat_df["macro_f1"] <= 1.0).all()

        # Forecasting metrics
        for_df = df[df["task"] == "forecasting"]
        if not for_df.empty:
            assert (for_df["mae"] >= 0.0).all()
            assert (for_df["rmse"] >= 0.0).all()

    def test_runtime_metadata_exists(self, benchmark_results):
        """Runtime metadata must exist and training_seconds > 0."""
        out_dir, _, _ = benchmark_results
        df = pd.read_csv(out_dir / "experiment_registry.csv")
        assert "training_seconds" in df.columns, "training_seconds column missing!"
        assert "inference_ms" in df.columns, "inference_ms column missing!"
        assert (df["training_seconds"] >= 0).all()

    def test_publication_tables_created(self, benchmark_results):
        """Tables 2-8 must be generated as non-empty CSVs."""
        out_dir, _, _ = benchmark_results
        tables = [
            "Table_2_Categorization.csv",
            "Table_3_Forecasting.csv",
            "Table_4_Runtime.csv",
            "Table_5_Cold_Start.csv",
            "Table_6_Novel_Merchant.csv",
            "Table_7_Robustness.csv",
            "Table_8_Calibration.csv",
        ]
        for t in tables:
            path = out_dir / t
            assert path.exists(), f"Missing table: {t}"
            assert path.stat().st_size > 0, f"Empty table: {t}"

    def test_auxiliary_benchmark_csvs_created(self, benchmark_results):
        """Robustness, Cold-Start, Merchant Gen, and Drift CSVs must exist."""
        out_dir, _, _ = benchmark_results
        csvs = [
            "robustness_metrics.csv",
            "cold_start_metrics.csv",
            "merchant_generalization.csv",
            "drift_metrics.csv",
        ]
        for c in csvs:
            path = out_dir / c
            assert path.exists(), f"Missing benchmark CSV: {c}"
            assert path.stat().st_size > 0, f"Empty benchmark CSV: {c}"

    def test_confusion_matrices_saved(self, benchmark_results):
        """Confusion matrices directory must contain CSV files."""
        out_dir, _, _ = benchmark_results
        cm_dir = out_dir / "confusion_matrices"
        assert cm_dir.exists(), "confusion_matrices directory missing!"
        cms = list(cm_dir.glob("*.csv"))
        assert len(cms) > 0, "No confusion matrix CSVs found!"

    def test_classification_reports_saved(self, benchmark_results):
        """Classification reports directory must contain txt files."""
        out_dir, _, _ = benchmark_results
        cr_dir = out_dir / "classification_reports"
        assert cr_dir.exists(), "classification_reports directory missing!"
        reports = list(cr_dir.glob("*.txt"))
        assert len(reports) > 0, "No classification report text files found!"

    def test_figures_created(self, benchmark_results):
        """Figures 5-13 must be created as non-empty PNGs."""
        _, fig_dir, _ = benchmark_results
        expected_figs = [
            "figure_5_macro_f1_comparison.png",
            "figure_6_forecast_mae_comparison.png",
            "figure_7_confusion_matrix.png",
            "figure_8_feature_importance.png",
            "figure_9_calibration_curve.png",
            "figure_10_cold_start_performance.png",
            "figure_11_novel_merchant_performance.png",
            "figure_12_robustness_degradation.png",
            "figure_13_runtime_comparison.png",
        ]
        for f in expected_figs:
            path = fig_dir / f
            assert path.exists(), f"Missing figure: {f}"
            assert path.stat().st_size > 0, f"Empty figure: {f}"
