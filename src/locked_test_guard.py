import os
import json
from pathlib import Path
from typing import Dict, Any
from src.experiment_runner import get_git_provenance

class LockedTestGuard:
    """Safeguard to ensure final locked test is only run against the exact frozen architecture."""
    
    MANIFEST_PATH = Path("reports/final_model_manifest.json")
    
    @classmethod
    def create_manifest(cls, config: Dict[str, Any]):
        """Generates the frozen architecture manifest."""
        prov = get_git_provenance()
        manifest = {
            "git_commit": prov["git_commit"],
            "git_branch": prov["git_branch"],
            "architecture": config.get("architecture", "PATFormer"),
            "hyperparameters": {
                "sequence_length": config.get("sequence_length"),
                "embedding_dimension": config.get("embedding_dimension"),
                "layers": config.get("layers"),
                "dropout": config.get("dropout"),
                "learning_rate": config.get("learning_rate"),
                "optimizer": config.get("optimizer"),
                "batch_size": config.get("batch_size")
            },
            "preprocessing_version": config.get("preprocessing_version", "v1.0"),
            "dataset_hash": config.get("dataset_hash")
        }
        
        with open(cls.MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        print("[LOCKED] FINAL MODEL MANIFEST FROZEN.")

    @classmethod
    def verify_or_fail(cls, current_config: Dict[str, Any]):
        """Refuses to run if the current state doesn't match the frozen manifest."""
        if not cls.MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Cannot run FINAL test. {cls.MANIFEST_PATH} missing. You must lock the model first.")
            
        with open(cls.MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
            
        prov = get_git_provenance()
        
        errors = []
        if manifest["git_commit"] != prov["git_commit"]:
            errors.append(f"Git commit mismatch: Manifest={manifest['git_commit']} vs Current={prov['git_commit']}")
            
        manifest_hp = manifest.get("hyperparameters", {})
        for k, v in manifest_hp.items():
            current_v = current_config.get(k)
            if v != current_v:
                errors.append(f"Hyperparameter '{k}' mismatch: Manifest={v} vs Current={current_v}")
                
        if errors:
            print("[ERROR] REFUSING TO RUN FINAL TEST: Architecture/Configuration mismatch detected!")
            for err in errors:
                print(f"  - {err}")
            raise ValueError("Final test aborted due to manifest mismatch. Do not tune after locking.")
        else:
            print("[SUCCESS] Final Test Guard Passed. Configuration matches locked manifest.")
