import os
import sys
import json
import torch
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

def get_git_provenance() -> Dict[str, str]:
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

def get_hardware_info() -> Dict[str, str]:
    hw = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "os": platform.system()
    }
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
        hw["cuda_version"] = torch.version.cuda
    else:
        hw["gpu"] = "None"
        hw["cuda_version"] = "None"
    return hw

def hash_dict(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

@dataclass
class ExperimentManifest:
    experiment_id: str
    experiment_type: str
    mode: str
    dataset: str
    dataset_hash: str
    config: Dict[str, Any]
    config_hash: str
    seed: int
    git_commit: str
    git_branch: str
    hardware: Dict[str, str]
    timestamp_utc: str
    status: str = "PLANNED"
    results: Optional[Dict[str, Any]] = None

class ExperimentRunner:
    """Orchestrates Google Colab GPU experiments with strict checkpointing and provenance."""
    
    def __init__(self, exp_id: str, exp_type: str, mode: str, seed: int, config: Dict[str, Any]):
        self.exp_id = exp_id
        self.mode = mode.lower() # smoke, development, final
        if self.mode not in ["smoke", "development", "final"]:
            raise ValueError("Mode must be smoke, development, or final")
            
        self.seed = seed
        self.config = config
        
        # Strict directory separation
        self.reports_dir = Path(f"reports/results/{self.mode}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(f"artifacts/{self.mode}/checkpoints/{exp_id}")
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
            hw = get_hardware_info()
            self.manifest = ExperimentManifest(
                experiment_id=exp_id,
                experiment_type=exp_type,
                mode=self.mode,
                dataset=config.get("dataset", "combined"),
                dataset_hash="mocked_data_hash_for_now", # TODO: Implement actual data hashing
                config=config,
                config_hash=hash_dict(config),
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
        self.manifest.status = "RUNNING"
        self._save_manifest()

    def complete(self, results: Dict[str, Any]):
        self.manifest.status = "COMPLETED"
        self.manifest.results = results
        self._save_manifest()
        
    def fail(self, reason: str):
        self.manifest.status = "FAILED"
        self.manifest.results = {"error": reason}
        self._save_manifest()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_dataloader(self, dataset, batch_size=64, is_train=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
    def save_checkpoint(self, state_dict: Dict[str, Any], filename="checkpoint.pt"):
        torch.save(state_dict, self.checkpoint_dir / filename)
        
    def load_checkpoint(self, filename="checkpoint.pt") -> Optional[Dict[str, Any]]:
        path = self.checkpoint_dir / filename
        if path.exists():
            return torch.load(path, weights_only=False)
        return None
