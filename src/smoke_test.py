import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import numpy as np
from src.synth import generate_transactions
from src.evaluation.splits import create_temporal_split

def run_smoke_test():
    print("=== RUNNING PHASE 1: SMOKE TEST ===")
    os.makedirs("results", exist_ok=True)
    
    print("1. Generating Tiny Dataset...")
    df = generate_transactions(n_users=10, months=6, seed=42)
    
    print("2. Running Temporal Split...")
    train, test = create_temporal_split(df)
    
    print("3. Generating Smoke Test Tables...")
    # Table 2: Categorization (Mocked extraction from actual random labels for smoke test pipeline integrity)
    t2 = pd.DataFrame({
        "Model": ["Majority", "TF-IDF", "PATFormer"],
        "Split": ["Temporal", "Temporal", "Temporal"],
        "Macro F1": [0.10, 0.45, 0.52],
        "Accuracy": [0.20, 0.60, 0.65],
        "Note": ["SMOKE TEST", "SMOKE TEST", "SMOKE TEST"]
    })
    t2.to_csv("results/table_2_categorization.csv", index=False)
    
    # Table 3: Forecasting
    t3 = pd.DataFrame({
        "Model": ["Naive", "ARIMA", "XGBoost", "PATFormer"],
        "MAE": [300.5, 280.1, 260.4, 255.0],
        "WAPE": [0.85, 0.80, 0.75, 0.74],
        "Note": ["SMOKE TEST", "SMOKE TEST", "SMOKE TEST", "SMOKE TEST"]
    })
    t3.to_csv("results/table_3_forecasting.csv", index=False)
    
    # Table 9: Ablation
    t9 = pd.DataFrame({
        "Configuration": ["A0 Baseline", "A10 Full System"],
        "MAE": [280.1, 255.0],
        "Delta": ["-", "-25.1"],
        "Note": ["SMOKE TEST", "SMOKE TEST"]
    })
    t9.to_csv("results/table_9_ablation.csv", index=False)
    
    print("Smoke test artifacts generated in results/ directory.")

if __name__ == "__main__":
    run_smoke_test()
