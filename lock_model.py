import sys
from src.locked_test_guard import LockedTestGuard

config = {
    "architecture": "PATFormer",
    "sequence_length": 64,
    "embedding_dimension": 96,
    "layers": 3,
    "dropout": 0.15,
    "learning_rate": 0.001,
    "optimizer": "AdamW",
    "batch_size": 16,
    "preprocessing_version": "v1.0",
    "dataset_hash": "mocked_data_hash_for_now"
}

LockedTestGuard.create_manifest(config)
