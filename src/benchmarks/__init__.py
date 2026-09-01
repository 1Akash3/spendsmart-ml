"""SpendSmart V3 Benchmark Engine Package.

Core data structures and shared utilities for the empirical benchmark system.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_SRC)
_ROOT = os.path.dirname(_PKG)
for _p in (_ROOT, _PKG, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RuntimeInfo:
    """Runtime measurements for a single experiment."""
    training_seconds: float = 0.0
    inference_ms: float = 0.0
    peak_ram_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    model_parameters: int = 0
    model_size_mb: float = 0.0


@dataclass
class ExperimentMeta:
    """Full provenance for a single experiment run."""
    experiment_id: str = ""
    model_name: str = ""
    task: str = ""          # "categorization" or "forecasting"
    split_name: str = ""
    seed: int = 42
    mode: str = "smoke"
    git_commit: str = ""
    dataset_hash: str = ""
    config_hash: str = ""
    device: str = "cpu"
    runtime_seconds: float = 0.0
    timestamp: str = ""
    status: str = "PLANNED"


@dataclass
class BenchmarkResult:
    """Complete result from a single benchmark experiment."""
    meta: ExperimentMeta
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[Any] = None
    classification_report: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)
    predictions: Optional[Any] = None
    probabilities: Optional[Any] = None


# ============================================================================
# Mode Configuration
# ============================================================================

SMOKE_SEEDS = [42]
DEV_SEEDS = [42, 43, 44, 45, 46]

CATEGORIZATION_MODELS = [
    "majority", "tfidf_lr", "tfidf_svm", "random_forest",
    "xgboost", "lightgbm",
]

FORECAST_MODELS = [
    "naive_previous", "naive_rolling", "naive_seasonal",
    "linear_regression", "random_forest", "xgboost", "lightgbm",
]

CAT_SPLIT_NAMES = ["random", "temporal", "merchant_disjoint",
                   "novel_merchant", "cross_source", "noisy_description"]
FORECAST_SPLIT_NAMES = ["temporal"]

COLD_START_BUCKETS = [(0, 5), (6, 20), (21, 50), (51, 100), (101, 999999)]
NOISE_LEVELS = [0.05, 0.10, 0.20]


# ============================================================================
# Utilities
# ============================================================================

def make_experiment_id(task: str, model: str, split: str, seed: int) -> str:
    """Generate deterministic experiment ID."""
    task_prefix = "CAT" if task == "categorization" else "FOR"
    model_tag = model.upper().replace("_", "-")
    split_tag = "SPLIT" + split[0].upper()
    return f"{task_prefix}-{model_tag}-{split_tag}-S{seed}"


def get_device_str() -> str:
    """Get device string."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def hash_config(config: dict) -> str:
    """SHA-256 hash of a config dict."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        return subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "UNKNOWN"


def load_dataset_hash(mode: str) -> str:
    """Load the dataset hash from pipeline metadata."""
    hash_path = Path(f"data/{mode}/dataset_hash.json")
    if hash_path.exists():
        with open(hash_path) as f:
            return json.load(f).get("dataset_sha256", "UNKNOWN")[:16]
    return "UNKNOWN"


class MemoryTracker:
    """Context manager for peak RAM tracking."""

    def __init__(self):
        self.peak_mb = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        self.elapsed = time.perf_counter() - self._start


class Timer:
    """Simple wall-clock timer."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
