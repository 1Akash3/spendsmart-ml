import os
import sys
import json
import torch
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

def get_git_provenance() -> Dict[str, str]:
    """Retrieves current Git state for experiment immutability."""
    def _run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except:
            return "UNKNOWN"
            
    return {
        "git_commit": _run("git log -1 --format=%H"),
        "git_branch": _run("git branch --show-current"),
        "git_remote": _run("git remote -v | head -n1 | awk '{print $2}'")
    }

@dataclass
class ExperimentManifest:
    experiment_id: str
    experiment_type: str
    dataset: str
    dataset_hash: str
    config: Dict[str, Any]
    seed: int
    git_commit: str
    git_branch: str
    hardware: Dict[str, Any]
    timestamp_utc: str
    status: str = "PLANNED"
    results: Optional[Dict[str, Any]] = None

class ExperimentRunner:
    """Orchestrates Google Colab GPU experiments with strict checkpointing and provenance."""
    
    def __init__(self, exp_id: str, exp_type: str, seed: int, config: Dict[str, Any]):
        self.exp_id = exp_id
        self.seed = seed
        self.config = config
        self.reports_dir = Path("reports/experiments")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(f"artifacts/checkpoints/{exp_id}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.reports_dir / f"{exp_id}.json"
        
        # Load existing if resuming
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self.manifest = ExperimentManifest(**json.load(f))
                if self.manifest.status == "COMPLETED":
                    print(f"Experiment {exp_id} already COMPLETED. Skipping.")
                    sys.exit(0)
        else:
            prov = get_git_provenance()
            hw = {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            }
            self.manifest = ExperimentManifest(
                experiment_id=exp_id,
                experiment_type=exp_type,
                dataset=config.get("dataset", "combined"),
                dataset_hash="cached_sha256_mock", # TODO: hook into data pipeline
                config=config,
                seed=seed,
                git_commit=prov["git_commit"],
                git_branch=prov["git_branch"],
                hardware=hw,
                timestamp_utc=datetime.utcnow().isoformat(),
                status="PLANNED"
            )
            self._save_manifest()
            
    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(asdict(self.manifest), f, indent=2)

    def start(self):
        """Marks experiment as running."""
        self.manifest.status = "RUNNING"
        self._save_manifest()

    def complete(self, results: Dict[str, Any]):
        """Marks experiment as complete and saves final metrics."""
        self.manifest.status = "COMPLETED"
        self.manifest.results = results
        self._save_manifest()
        
    def fail(self, reason: str):
        """Marks experiment as failed."""
        self.manifest.status = "FAILED"
        self.manifest.results = {"error": reason}
        self._save_manifest()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_dataloader(self, dataset, batch_size=64, is_train=True):
        """Creates an efficient GPU-pinned dataloader."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
