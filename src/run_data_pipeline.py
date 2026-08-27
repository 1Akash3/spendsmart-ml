import argparse
import os
from pathlib import Path
import pandas as pd
from src.synth import generate_transactions

def run(mode):
    print(f"Executing Job 1: Data Pipeline (Mode: {mode})")
    n_users = 10 if mode == "SMOKE_TEST" else 500
    months = 6 if mode == "SMOKE_TEST" else 24
    
    df = generate_transactions(n_users=n_users, months=months, seed=42)
    
    out_dir = Path("artifacts/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split info
    df.to_parquet(out_dir / "transactions.parquet")
    
    # Mock table 1 (Dataset Statistics)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    t1 = pd.DataFrame({
        "Metric": ["Users", "Months", "Transactions"],
        "Value": [n_users, months, len(df)],
        "Mode": [mode, mode, mode]
    })
    t1.to_csv("reports/tables/Table_1_Dataset.csv", index=False)
    print("Job 1 Complete. Dataset Cached.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="SMOKE_TEST")
    args = parser.parse_args()
    run(args.mode)
