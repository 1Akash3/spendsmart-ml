"""Feature Cache Engine for SpendSmart V4.1.

Caches expensive preprocessed data structures on disk and in memory:
- TF-IDF vocabulary and sparse matrices (CSR format)
- Supervised forecasting panel frames (parquet)
- Sequence tensors for PATFormer DataLoaders
- Merchant embeddings

All cache items are validated against `data/{mode}/dataset_hash.json`.
Cache is invalidated automatically if dataset hash changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import load_dataset_hash, log


class FeatureCacheManager:
    """Manages disk and memory caching of preprocessed features."""

    def __init__(self, mode: str = "smoke"):
        self.mode = mode
        self.cache_dir = Path(f"artifacts/cache/{mode}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}
        self.current_dataset_hash = load_dataset_hash(mode)
        self._validate_cache_hash()

    def _validate_cache_hash(self) -> None:
        """Invalidate cache directory if dataset hash has changed."""
        hash_file = self.cache_dir / "cache_dataset_hash.json"
        if hash_file.exists():
            try:
                stored = json.loads(hash_file.read_text()).get("dataset_hash")
                if stored != self.current_dataset_hash:
                    log(f"  [FeatureCache] Dataset hash changed ({stored} -> {self.current_dataset_hash}). Invalidating cache.")
                    self.clear_cache()
            except Exception:
                self.clear_cache()

        # Save current hash
        hash_file.write_text(json.dumps({"dataset_hash": self.current_dataset_hash}, indent=2))

    def clear_cache(self) -> None:
        """Clear disk cache files."""
        self.memory_cache.clear()
        for f in self.cache_dir.glob("*"):
            if f.is_file() and f.name != "cache_dataset_hash.json":
                try:
                    f.unlink()
                except Exception:
                    pass

    def get_sparse_matrix(self, key: str) -> Optional[csr_matrix]:
        """Load sparse matrix from memory or disk cache."""
        if key in self.memory_cache:
            return self.memory_cache[key]

        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            try:
                mat = load_npz(path)
                self.memory_cache[key] = mat
                return mat
            except Exception:
                return None
        return None

    def save_sparse_matrix(self, key: str, matrix: csr_matrix) -> None:
        """Save sparse matrix to memory and disk cache."""
        self.memory_cache[key] = matrix
        path = self.cache_dir / f"{key}.npz"
        save_npz(path, matrix)

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from memory or disk cache."""
        if key in self.memory_cache:
            return self.memory_cache[key]

        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                self.memory_cache[key] = df
                return df
            except Exception:
                return None
        return None

    def save_dataframe(self, key: str, df: pd.DataFrame) -> None:
        """Save DataFrame to memory and disk cache."""
        self.memory_cache[key] = df
        path = self.cache_dir / f"{key}.parquet"
        df.to_parquet(path, index=False)

    def get_object(self, key: str) -> Optional[Any]:
        """Load arbitrary object (joblib) from memory or disk cache."""
        if key in self.memory_cache:
            return self.memory_cache[key]

        path = self.cache_dir / f"{key}.joblib"
        if path.exists():
            try:
                obj = joblib.load(path)
                self.memory_cache[key] = obj
                return obj
            except Exception:
                return None
        return None

    def save_object(self, key: str, obj: Any) -> None:
        """Save arbitrary object (joblib) to memory and disk cache."""
        self.memory_cache[key] = obj
        path = self.cache_dir / f"{key}.joblib"
        joblib.dump(obj, path)
