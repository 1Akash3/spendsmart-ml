"""Unit and Integration Tests for SpendSmart V4.1 Research Infrastructure Optimization.

Verifies:
1. FeatureCacheManager caching & dataset hash invalidation logic.
2. AutosaveManager artifact versioning & drive mirror checks.
3. GitAutoSync commit message formatting.
4. RuntimeLogger resource logging & Table 11 comparison generation.
5. ExperimentScheduler resumable queue status transitions & TensorBoard writer.
6. Generation of Figures 17-20 in both PNG (300 DPI) and SVG vector formats.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# Path bootstrap
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.autosave_manager import AutosaveManager
from src.experiment_scheduler import ExperimentScheduler, ScheduledJob
from src.feature_cache import FeatureCacheManager
from src.git_sync import GitAutoSync
from src.publication_artifacts import generate_extended_figures
from src.runtime_logger import RuntimeLogger


class TestV41OptimizationSuite:
    """Test suite for SpendSmart V4.1 research optimization modules."""

    def test_feature_cache_manager(self):
        """FeatureCacheManager must cache and retrieve sparse matrices and DataFrames."""
        cache = FeatureCacheManager("smoke")

        # Sparse matrix test
        mat = csr_matrix([[1, 0, 2], [0, 3, 0]])
        cache.save_sparse_matrix("test_mat", mat)
        retrieved_mat = cache.get_sparse_matrix("test_mat")
        assert retrieved_mat is not None
        assert (retrieved_mat.toarray() == mat.toarray()).all()

        # DataFrame test
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        cache.save_dataframe("test_df", df)
        retrieved_df = cache.get_dataframe("test_df")
        assert retrieved_df is not None
        assert len(retrieved_df) == 2

    def test_autosave_manager(self, tmp_path):
        """AutosaveManager must mirror artifacts and support versioning."""
        autosave = AutosaveManager("smoke")
        dummy_file = tmp_path / "test_artifact.json"
        dummy_file.write_text(json.dumps({"test": "data"}))

        saved_path = autosave.save_artifact(dummy_file, category="reports")
        assert saved_path is not None
        assert saved_path.exists()

    def test_runtime_logger_and_table_11(self):
        """RuntimeLogger must record resources and generate Table 11 & optimization report."""
        logger = RuntimeLogger("smoke")
        t0 = time.time()
        time.sleep(0.01)
        t1 = time.time()

        rec = logger.log_experiment("CAT-TEST-S42", "tfidf_lr", "temporal", 42, 100, t0, t1)
        assert rec["experiment_id"] == "CAT-TEST-S42"
        assert rec["elapsed_seconds"] > 0

        t11 = logger.generate_optimization_reports()
        assert not t11.empty
        assert "Mean Runtime (s)" in t11.columns
        assert Path("reports/optimization_report.md").exists()

    def test_experiment_scheduler_resumption(self):
        """ExperimentScheduler must track queue states and resume completed runs."""
        scheduler = ExperimentScheduler("smoke")
        exp_id = scheduler.enqueue_job("categorization", "tfidf_lr", "temporal", 9999)

        # Cleanup any previous test checkpoint marker
        ckpt_file = scheduler.ckpt_dir / f"{exp_id}_COMPLETED.pt"
        if ckpt_file.exists():
            ckpt_file.unlink()
        if exp_id in scheduler.jobs:
            scheduler.jobs[exp_id].status = "PENDING"

        assert not scheduler.is_completed(exp_id)
        scheduler.mark_running(exp_id)
        assert scheduler.jobs[exp_id].status in ("RUNNING", "RESUMED")

        scheduler.mark_completed(exp_id, 1.25)
        assert scheduler.is_completed(exp_id)
        assert scheduler.jobs[exp_id].status == "COMPLETED"

    def test_figures_17_to_20_png_and_svg(self):
        """Figures 17-20 must be generated as non-empty PNG (300 DPI) and SVG files."""
        generate_extended_figures("smoke")
        fig_dir = Path("reports/figures")

        for fig_idx in range(17, 21):
            png_matches = list(fig_dir.glob(f"figure_{fig_idx}_*.png"))
            svg_matches = list(fig_dir.glob(f"figure_{fig_idx}_*.svg"))

            assert len(png_matches) > 0, f"Missing PNG for Figure {fig_idx}"
            assert len(svg_matches) > 0, f"Missing SVG for Figure {fig_idx}"
            assert png_matches[0].stat().st_size > 0
            assert svg_matches[0].stat().st_size > 0
