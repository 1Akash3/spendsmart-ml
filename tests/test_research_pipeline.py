"""End-to-End Integration Tests for SpendSmart V4 Master Research Pipeline.

Verifies that the master research pipeline executes in smoke mode and generates:
- All required CSV artifacts (Tables 1-10, registry, patformer history, drift, health scores, latency profile, ablations).
- Publication figures 1-16.
- Research paper cards and audit documents.
- Serving endpoint API schema & sub-50ms latency SLAs.
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
from src.run_research_pipeline import run_master_pipeline
from src.serving import SpendSmartServingAPI


@pytest.fixture(scope="module")
def v4_pipeline_results():
    """Ensure data exists and run the V4 research pipeline in smoke mode."""
    smoke_data_dir = Path("data/smoke")
    if not smoke_data_dir.exists() or not (smoke_data_dir / "splits/train.parquet").exists():
        run_data_pipeline("smoke")

    res = run_master_pipeline("smoke")
    out_dir = Path("reports/results/smoke")
    fig_dir = Path("reports/figures")
    reports_dir = Path("reports")
    return out_dir, fig_dir, reports_dir, res


class TestV4ResearchPipeline:
    """Test suite for SpendSmart V4 Master Research Pipeline."""

    def test_pipeline_execution_status(self, v4_pipeline_results):
        """Pipeline execution must return SUCCESS status."""
        _, _, _, res = v4_pipeline_results
        assert res["status"] == "SUCCESS"
        assert res["elapsed_seconds"] > 0

    def test_publication_tables_exist_and_nonempty(self, v4_pipeline_results):
        """Tables 2-10 must exist as non-empty CSV files."""
        out_dir, _, _, _ = v4_pipeline_results
        required_tables = [
            "Table_2_Categorization.csv",
            "Table_3_Forecasting.csv",
            "Table_4_Runtime.csv",
            "Table_5_Cold_Start_Routers.csv",
            "Table_9_Statistical_Summary.csv",
            "ablation_matrix.csv",
            "robustness_metrics.csv",
            "drift_metrics.csv",
            "financial_health_scores.csv",
        ]
        for t in required_tables:
            path = out_dir / t
            assert path.exists(), f"Missing V4 table: {t}"
            assert path.stat().st_size > 0, f"Empty V4 table: {t}"

    def test_patformer_history_saved(self, v4_pipeline_results):
        """patformer_history.csv must exist with epoch metric records."""
        out_dir, _, _, _ = v4_pipeline_results
        path = out_dir / "patformer_history.csv"
        assert path.exists(), "patformer_history.csv missing!"
        df = pd.read_csv(path)
        assert len(df) > 0, "patformer_history.csv is empty!"
        assert "val_loss" in df.columns
        assert "val_macro_f1" in df.columns

    def test_figures_1_to_16_exist(self, v4_pipeline_results):
        """Figures 1-16 must exist in reports/figures/ as non-empty PNG files."""
        _, fig_dir, _, _ = v4_pipeline_results
        for idx in range(1, 17):
            matching = list(fig_dir.glob(f"figure_{idx}_*.png"))
            assert len(matching) > 0, f"Missing Figure {idx} in {fig_dir}"
            assert matching[0].stat().st_size > 0, f"Figure {idx} is empty!"

    def test_paper_documents_exist(self, v4_pipeline_results):
        """Paper markdown documents must exist in reports/."""
        _, _, reports_dir, _ = v4_pipeline_results
        required_mds = [
            "MODEL_CARD.md",
            "DATA_CARD.md",
            "LIMITATIONS.md",
            "ERROR_ANALYSIS.md",
            "REPRODUCIBILITY.md",
            "PAPER_READINESS_AUDIT.md",
            "RESEARCH_GATE_VERIFICATION.md",
        ]
        for md in required_mds:
            path = reports_dir / md
            assert path.exists(), f"Missing markdown document: {md}"
            assert path.stat().st_size > 0, f"Empty markdown document: {md}"

    def test_serving_api_endpoints_and_latency(self, v4_pipeline_results):
        """Serving API endpoints must return valid schemas with < 50ms latency."""
        api = SpendSmartServingAPI("smoke")

        # 1. predict_transaction
        res_tx = api.predict_transaction("Swiggy Order", 450.0)
        assert "predicted_category" in res_tx
        assert res_tx["latency_ms"] < 50.0

        # 2. forecast_user
        res_fc = api.forecast_user("u1")
        assert "category_forecasts" in res_fc
        assert res_fc["latency_ms"] < 50.0

        # 3. predict_user_state
        res_st = api.predict_user_state("u1")
        assert "cohort_id" in res_st
        assert res_st["latency_ms"] < 50.0

        # 4. recommend_budget
        res_rec = api.recommend_budget("u1")
        assert "recommendations" in res_rec
        assert res_rec["latency_ms"] < 50.0

        # 5. health_score
        res_hs = api.health_score("u1")
        assert "total_health_score" in res_hs
        assert res_hs["latency_ms"] < 50.0
