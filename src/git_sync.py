"""Git Auto Sync Engine for SpendSmart V4.1.

Automatically stages, commits, and optionally pushes completed experiment artifacts:
- Detects git modifications in reports/ and artifacts/
- Formats structured commit messages per completed experiment
- Pushes to remote repository if GIT_TOKEN or write access is configured
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.benchmarks import log


class GitAutoSync:
    """Manages automatic staging, committing, and pushing of completed experiment runs."""

    def __init__(self, enable_push: bool = False):
        self.repo_dir = Path(_ROOT)
        self.enable_push = enable_push or bool(os.environ.get("GITHUB_TOKEN"))

    def _run_git(self, args: list[str]) -> Tuple[int, str]:
        """Execute a git command in the repository directory."""
        try:
            res = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                cwd=self.repo_dir,
            )
            return res.returncode, res.stdout.strip()
        except Exception as e:
            return 1, str(e)

    def sync_experiment(
        self,
        experiment_id: str,
        runtime_seconds: float,
        seed: int,
        split_name: str,
    ) -> bool:
        """Stage, commit, and optionally push completed experiment artifacts."""
        # 1. Stage results and artifacts
        code, _ = self._run_git(["add", "reports/", "artifacts/experiments/"])
        if code != 0:
            return False

        # Check if there are changes to commit
        code, status_out = self._run_git(["status", "--porcelain"])
        if not status_out:
            return True  # Nothing new to commit

        # 2. Format commit message
        mins, secs = divmod(int(runtime_seconds), 60)
        hrs, mins = divmod(mins, 60)
        time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        msg = f"{experiment_id} completed | Runtime: {time_str} | Seed: {seed} | Split: {split_name}"

        # 3. Commit
        code, commit_out = self._run_git(["commit", "-m", msg])
        if code != 0:
            return False

        log(f"  [GitSync] Committed {experiment_id} ({time_str})")

        # 4. Push if enabled
        if self.enable_push:
            code, push_out = self._run_git(["push", "origin", "main"])
            if code == 0:
                log(f"  [GitSync] Pushed {experiment_id} to remote main.")

        return True
