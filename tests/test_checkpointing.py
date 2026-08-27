import os
import torch
import pytest
from pathlib import Path
from src.experiment_runner import ExperimentRunner

def test_checkpoint_resume():
    exp_id = "TEST-CKPT-001"
    config = {"dataset": "test", "hyper": 42}
    
    # 1. Start experiment
    runner1 = ExperimentRunner(exp_id=exp_id, exp_type="Test", mode="smoke", seed=42, config=config)
    runner1.start()
    
    # 2. Save checkpoint (simulating interruption)
    mock_state = {"epoch": 5, "model_weights": [1, 2, 3]}
    runner1.save_checkpoint(mock_state, filename="interrupt.pt")
    
    # 3/4. Restart and Resume
    runner2 = ExperimentRunner(exp_id=exp_id, exp_type="Test", mode="smoke", seed=42, config=config)
    
    # 5/6. Verify continuation
    loaded_state = runner2.load_checkpoint("interrupt.pt")
    assert loaded_state is not None
    assert loaded_state["epoch"] == 5
    assert loaded_state["model_weights"] == [1, 2, 3]
    
    # Clean up
    if (runner2.checkpoint_dir / "interrupt.pt").exists():
        os.remove(runner2.checkpoint_dir / "interrupt.pt")
    if runner2.manifest_path.exists():
        os.remove(runner2.manifest_path)
