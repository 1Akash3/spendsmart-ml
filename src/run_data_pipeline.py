import argparse
import os
from pathlib import Path
import pandas as pd
from src.synth import generate_transactions
from src.locked_test_guard import LockedTestGuard

def run(mode):
    print(f"Executing Job 1: Data Pipeline (Mode: {mode})")
    
    if mode == "smoke":
        n_users, months = 10, 6
    else:
        n_users, months = 500, 24
        
    out_dir = Path(f"data/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reports_dir = Path(f"reports/results/{mode}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = generate_transactions(n_users=n_users, months=months, seed=42)
    df.to_parquet(out_dir / "transactions.parquet")
    
    if mode == "final":
        LockedTestGuard.verify_or_fail({"dataset_hash": "mocked_data_hash_for_now"})
        
    t1 = pd.DataFrame({
        "Metric": ["Users", "Months", "Transactions"],
        "Value": [n_users, months, len(df)],
        "Mode": [mode.upper(), mode.upper(), mode.upper()]
    })
    t1.to_csv(reports_dir / "Table_1_Dataset.csv", index=False)
    print(f"Job 1 Complete. Dataset Cached to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "development", "final"])
    args = parser.parse_args()
    run(args.mode)
