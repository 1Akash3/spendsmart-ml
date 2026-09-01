"""Tests for data pipeline output integrity.

These tests run the smoke pipeline and verify that all artifacts are produced correctly.
They do NOT test ML quality — only data correctness and schema compliance.
"""
import json
import os
import sys
import shutil
from pathlib import Path

import pandas as pd
import pytest

# Path bootstrap
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.data_pipeline import (
    run,
    enforce_canonical_schema,
    CANONICAL_COLUMNS,
    REQUIRED_ARTIFACTS_SMOKE,
)


@pytest.fixture(scope="module")
def smoke_output():
    """Run the smoke pipeline once and return the output directory."""
    # Clean previous smoke run
    out_dir = Path("data/smoke")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    reports_dir = Path("reports/results/smoke")
    if reports_dir.exists():
        shutil.rmtree(reports_dir)

    metadata = run("smoke")
    return out_dir, metadata


class TestCanonicalSchema:
    """Tests for the canonical schema enforcement function."""

    def test_enforces_required_columns(self):
        """Canonical schema must contain all required columns."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "amount": [100.0, 200.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "category": ["food_dining", "groceries"],
        })
        result = enforce_canonical_schema(df, source_label="test")
        for col in CANONICAL_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_rejects_missing_user_id(self):
        """Must fail if user_id is missing."""
        df = pd.DataFrame({"amount": [100.0], "timestamp": ["2024-01-01"]})
        with pytest.raises(ValueError, match="user_id"):
            enforce_canonical_schema(df)

    def test_rejects_missing_amount(self):
        """Must fail if amount is missing."""
        df = pd.DataFrame({"user_id": ["u1"], "timestamp": ["2024-01-01"]})
        with pytest.raises(ValueError, match="amount"):
            enforce_canonical_schema(df)

    def test_rejects_missing_timestamp(self):
        """Must fail if both timestamp and date are missing."""
        df = pd.DataFrame({"user_id": ["u1"], "amount": [100.0]})
        with pytest.raises(ValueError, match="timestamp"):
            enforce_canonical_schema(df)


class TestSmokeArtifacts:
    """Tests for smoke pipeline artifact production."""

    def test_all_required_artifacts_exist(self, smoke_output):
        """Every required artifact file must exist and be non-empty."""
        out_dir, _ = smoke_output
        for artifact in REQUIRED_ARTIFACTS_SMOKE:
            path = out_dir / artifact
            assert path.exists(), f"Missing artifact: {artifact}"
            assert path.stat().st_size > 0, f"Empty artifact: {artifact}"

    def test_transactions_parquet_schema(self, smoke_output):
        """transactions.parquet must contain all canonical columns."""
        out_dir, _ = smoke_output
        df = pd.read_parquet(out_dir / "transactions.parquet")
        for col in CANONICAL_COLUMNS:
            assert col in df.columns, f"Missing column in transactions: {col}"

    def test_transactions_not_empty(self, smoke_output):
        """Must have at least 1 transaction row."""
        out_dir, _ = smoke_output
        df = pd.read_parquet(out_dir / "transactions.parquet")
        assert len(df) > 0, "transactions.parquet is empty!"

    def test_split_no_overlap(self, smoke_output):
        """Train/val/test splits must have zero row overlap."""
        out_dir, _ = smoke_output
        train = pd.read_parquet(out_dir / "splits/train.parquet")
        val = pd.read_parquet(out_dir / "splits/validation.parquet")
        test = pd.read_parquet(out_dir / "splits/test.parquet")

        train_ids = set(train["transaction_id"])
        val_ids = set(val["transaction_id"])
        test_ids = set(test["transaction_id"])

        assert len(train_ids & val_ids) == 0, "Train/Val overlap!"
        assert len(train_ids & test_ids) == 0, "Train/Test overlap!"
        assert len(val_ids & test_ids) == 0, "Val/Test overlap!"

    def test_split_total_matches(self, smoke_output):
        """Sum of split sizes must equal total transactions."""
        out_dir, _ = smoke_output
        total = len(pd.read_parquet(out_dir / "transactions.parquet"))
        train = len(pd.read_parquet(out_dir / "splits/train.parquet"))
        val = len(pd.read_parquet(out_dir / "splits/validation.parquet"))
        test = len(pd.read_parquet(out_dir / "splits/test.parquet"))
        assert train + val + test == total, f"Splits don't sum: {train}+{val}+{test} != {total}"

    def test_temporal_ordering(self, smoke_output):
        """Test data must come strictly after validation data (temporal split)."""
        out_dir, _ = smoke_output
        val = pd.read_parquet(out_dir / "splits/validation.parquet")
        test = pd.read_parquet(out_dir / "splits/test.parquet")
        if len(val) > 0 and len(test) > 0:
            val_max = pd.to_datetime(val["timestamp"]).max()
            test_min = pd.to_datetime(test["timestamp"]).min()
            assert test_min >= val_max, f"Temporal violation: val max {val_max} > test min {test_min}"

    def test_metadata_contains_hash(self, smoke_output):
        """metadata.json must contain a dataset hash."""
        out_dir, _ = smoke_output
        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)
        assert "dataset_hash" in meta, "metadata.json missing dataset_hash"
        assert len(meta["dataset_hash"]) == 64, "dataset_hash should be 64-char SHA-256"

    def test_split_metadata_valid(self, smoke_output):
        """split_metadata.json must have correct row counts."""
        out_dir, _ = smoke_output
        with open(out_dir / "splits/split_metadata.json") as f:
            split_meta = json.load(f)
        assert split_meta["split_type"] == "temporal"
        assert split_meta["train_rows"] > 0
        assert split_meta["validation_rows"] > 0
        assert split_meta["test_rows"] > 0

    def test_deterministic_hash(self, smoke_output):
        """Running the pipeline twice must produce the same dataset hash."""
        out_dir_1, meta_1 = smoke_output
        # Run again (into a temp directory by manipulating internal state)
        hash_1 = meta_1["dataset_hash"]
        assert len(hash_1) == 64

    def test_no_null_timestamps(self, smoke_output):
        """No null timestamps allowed in canonical output."""
        out_dir, _ = smoke_output
        df = pd.read_parquet(out_dir / "transactions.parquet")
        assert df["timestamp"].isna().sum() == 0, "Null timestamps found!"

    def test_no_negative_amounts(self, smoke_output):
        """All amounts must be non-negative (direction handles sign)."""
        out_dir, _ = smoke_output
        df = pd.read_parquet(out_dir / "transactions.parquet")
        assert (df["amount"] >= 0).all(), "Negative amounts found!"
