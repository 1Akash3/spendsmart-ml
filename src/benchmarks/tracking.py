"""Experiment tracking, registry, and checkpointing for benchmark runs."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.benchmarks import (
    BenchmarkResult, ExperimentMeta, RuntimeInfo,
    get_git_commit, load_dataset_hash, hash_config, log,
)


class ExperimentRegistry:
    """Tracks all experiment runs with full provenance.

    Maintains a CSV registry + per-experiment JSON manifests.
    """

    def __init__(self, mode: str, base_dir: Optional[Path] = None):
        self.mode = mode
        self.base_dir = base_dir or Path(f"reports/results/{mode}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "experiment_registry.csv"
        self.results: List[BenchmarkResult] = []
        self._load_existing_registry()

    def _load_existing_registry(self) -> None:
        """Load existing registry CSV into memory if it exists."""
        if self.registry_path.exists():
            try:
                df = pd.read_csv(self.registry_path)
                for _, row in df.iterrows():
                    meta = ExperimentMeta(
                        experiment_id=str(row["experiment_id"]),
                        model_name=str(row["model_name"]),
                        task=str(row.get("task", "categorization")),
                        split_name=str(row.get("split_name", "temporal")),
                        seed=int(row.get("seed", 42)),
                        mode=str(row.get("mode", self.mode)),
                        git_commit=str(row.get("git_commit", "N/A")),
                        dataset_hash=str(row.get("dataset_hash", "N/A")),
                        config_hash=str(row.get("config_hash", "N/A")),
                        device=str(row.get("device", "cpu")),
                        runtime_seconds=float(row.get("runtime_seconds", 0.0)),
                        timestamp=str(row.get("timestamp", "")),
                        status=str(row.get("status", "COMPLETED")),
                    )
                    runtime = RuntimeInfo(
                        training_seconds=float(row.get("training_seconds", 0.0)),
                        inference_ms=float(row.get("inference_ms", 0.0)),
                        peak_ram_mb=float(row.get("peak_ram_mb", 0.0)),
                        model_parameters=int(row.get("model_parameters", 0)),
                    )
                    metrics = {k: float(v) for k, v in row.items() if k not in meta.__dict__ and k not in runtime.__dict__ and isinstance(v, (int, float, np.number))}
                    res = BenchmarkResult(meta=meta, metrics=metrics, runtime=runtime)
                    self.results.append(res)
            except Exception:
                pass

    def has_result(self, experiment_id: str) -> bool:
        """Check if an experiment ID is already recorded in results."""
        return any(r.meta.experiment_id == experiment_id for r in self.results)

    def register(self, result: BenchmarkResult) -> None:
        """Register a completed experiment result."""
        if not self.has_result(result.meta.experiment_id):
            self.results.append(result)
        self._save_manifest(result)

    def _save_manifest(self, result: BenchmarkResult) -> None:
        """Save per-experiment JSON manifest."""
        manifest_dir = self.base_dir / "manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest = {
            "meta": asdict(result.meta),
            "metrics": result.metrics,
            "runtime": asdict(result.runtime),
        }
        path = manifest_dir / f"{result.meta.experiment_id}.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def save_registry(self) -> None:
        """Save the experiment registry CSV."""
        if not self.results:
            if not self.registry_path.exists():
                pd.DataFrame(columns=["experiment_id", "model_name", "task", "split_name", "seed", "mode", "status", "macro_f1", "mae"]).to_csv(self.registry_path, index=False)
            return

        rows = []
        for r in self.results:
            row = {
                "experiment_id": r.meta.experiment_id,
                "model_name": r.meta.model_name,
                "task": r.meta.task,
                "split_name": r.meta.split_name,
                "seed": r.meta.seed,
                "mode": r.meta.mode,
                "git_commit": r.meta.git_commit,
                "dataset_hash": r.meta.dataset_hash,
                "config_hash": r.meta.config_hash,
                "device": r.meta.device,
                "runtime_seconds": r.meta.runtime_seconds,
                "timestamp": r.meta.timestamp,
                "status": r.meta.status,
                "training_seconds": r.runtime.training_seconds,
                "inference_ms": r.runtime.inference_ms,
                "peak_ram_mb": r.runtime.peak_ram_mb,
                "model_parameters": r.runtime.model_parameters,
            }
            row.update(r.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.registry_path, index=False)
        log(f"  Registry saved: {len(rows)} experiments -> {self.registry_path}")

    def save_confusion_matrices(self) -> None:
        """Save all confusion matrices as CSV."""
        cm_dir = self.base_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        for r in self.results:
            if r.confusion_matrix is not None:
                np.savetxt(
                    cm_dir / f"{r.meta.experiment_id}_cm.csv",
                    r.confusion_matrix, delimiter=",", fmt="%.0f"
                )

    def save_classification_reports(self) -> None:
        """Save all classification reports as text."""
        cr_dir = self.base_dir / "classification_reports"
        cr_dir.mkdir(exist_ok=True)
        for r in self.results:
            if r.classification_report:
                path = cr_dir / f"{r.meta.experiment_id}_report.txt"
                path.write_text(r.classification_report)

    def save_feature_importances(self) -> None:
        """Save feature importances as CSV."""
        fi_dir = self.base_dir / "feature_importance"
        fi_dir.mkdir(exist_ok=True)
        for r in self.results:
            if r.feature_importance:
                fi_df = pd.DataFrame(
                    list(r.feature_importance.items()),
                    columns=["feature", "importance"]
                ).sort_values("importance", ascending=False)
                fi_df.to_csv(
                    fi_dir / f"{r.meta.experiment_id}_importance.csv",
                    index=False
                )


class CheckpointManager:
    """Handles experiment checkpointing and resumption."""

    def __init__(self, mode: str):
        self.checkpoint_dir = Path(f"artifacts/{mode}/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def has_checkpoint(self, experiment_id: str) -> bool:
        """Check if a completed checkpoint exists."""
        path = self.checkpoint_dir / f"{experiment_id}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("status") == "COMPLETED"
        return False

    def load_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint if it exists."""
        path = self.checkpoint_dir / f"{experiment_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def save_checkpoint(self, result: BenchmarkResult) -> None:
        """Save experiment checkpoint."""
        data = {
            "experiment_id": result.meta.experiment_id,
            "status": result.meta.status,
            "metrics": result.metrics,
            "runtime": asdict(result.runtime),
            "meta": asdict(result.meta),
        }
        path = self.checkpoint_dir / f"{result.meta.experiment_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
