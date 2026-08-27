import pytest
import pandas as pd
import numpy as np
import torch

def test_patformer_forward():
    from src.models.patformer import PATFormer
    model = PATFormer(num_categories=10, d_model=32, nhead=2, num_layers=1, max_seq_len=16, use_router=True)
    cat = torch.randint(0, 10, (4, 16))
    amt = torch.randn(4, 16, 1)
    ctx = torch.randn(4, 32)
    
    out = model(cat, amt, ctx)
    assert "category" in out
    assert "amount" in out
    assert out["category"].shape == (4, 10)
    assert out["amount"].shape == (4, 1)

def test_anomaly_injector():
    from src.anomaly import BehavioralAnomalyInjector
    df = pd.DataFrame({"amount": [10.0, 15.0, 20.0, 25.0]*25, "category": ["food"]*100})
    injector = BehavioralAnomalyInjector(random_state=42)
    anom_df = injector.inject(df, injection_rate=0.1)
    assert "is_injected_anomaly" in anom_df.columns
    assert anom_df["is_injected_anomaly"].sum() == 10

def test_splits():
    from src.evaluation.splits import create_temporal_split
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100),
        "target": np.random.randn(100)
    })
    train, test = create_temporal_split(df)
    assert len(train) == 80
    assert len(test) == 20
    assert train["timestamp"].max() < test["timestamp"].min()

def test_financial_health():
    from src.financial_health import FinancialHealthScorer
    scorer = FinancialHealthScorer()
    df = pd.DataFrame({
        "income": [5000, 5000],
        "target": [4000, 4500],
        "housing": [1500, 1500],
        "groceries": [500, 600]
    })
    score = scorer.score_user(df)
    assert "overall_score" in score
    assert 0 <= score["overall_score"] <= 100

def test_synthetic_data_schema_and_temporal_split():
    from src.synth import generate_transactions
    from src.evaluation.splits import create_temporal_split
    
    df = generate_transactions(n_users=10, months=6, seed=42)
    
    # 1. Verify required transaction fields
    required_cols = ["timestamp", "user_id", "amount", "direction", "merchant_raw", "category", "date", "type"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        
    # 2. Verify timestamp datatype and validity
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert not df["timestamp"].isnull().any()
    
    # 3. Verify temporal split execution on synthetic dataset
    train, test = create_temporal_split(df, train_fraction=0.8)
    assert len(train) > 0 and len(test) > 0
    assert train["timestamp"].max() <= test["timestamp"].min()

