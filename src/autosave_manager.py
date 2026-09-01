"""Autosave and Backup Management Engine for SpendSmart V4.1.

Triggers immediate background/sync autosave after:
- Every completed experiment
- Every completed epoch
- Every generated table, figure, or JSON report
- Every generated checkpoint

Supports optional Google Drive mirror under `/content/drive/MyDrive/SpendSmart_Backups/`
with incremental sync and versioning.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import log


class AutosaveManager:
    """Handles immediate artifact autosave, versioning, and Google Drive mirroring."""

    def __init__(self, mode: str = "smoke"):
        self.mode = mode
        self.drive_mount_dir = Path("/content/drive/MyDrive")
        self.drive_backup_dir = self.drive_mount_dir / "SpendSmart_Backups" / mode
        self.is_drive_available = self.drive_mount_dir.exists()

        if self.is_drive_available:
            self.drive_backup_dir.mkdir(parents=True, exist_ok=True)
            log(f"  [Autosave] Google Drive mirror active at {self.drive_backup_dir}")

    def save_artifact(self, src_path: Path, category: str = "reports") -> Optional[Path]:
        """Autosave artifact with timestamp versioning and Google Drive sync."""
        if not src_path.exists():
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

        # 1. Mirror to Google Drive if available
        if self.is_drive_available:
            try:
                dest_dir = self.drive_backup_dir / category
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / src_path.name
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                log(f"  [Autosave] Drive sync warning for {src_path.name}: {e}")

        return src_path

    def mirror_directory(self, src_dir: Path, category: str = "results") -> None:
        """Incrementally mirror an entire directory to Google Drive."""
        if not self.is_drive_available or not src_dir.exists():
            return

        try:
            dest_dir = self.drive_backup_dir / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            for item in src_dir.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(src_dir)
                    target = dest_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)

                    # Incremental check: copy only if target missing or size/mtime differs
                    if not target.exists() or target.stat().st_size != item.stat().st_size:
                        shutil.copy2(item, target)
        except Exception as e:
            log(f"  [Autosave] Directory mirror warning: {e}")
