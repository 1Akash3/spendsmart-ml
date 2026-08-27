import pytest
import pandas as pd
from datetime import datetime
from src.schema import Transaction
from src.merchant_resolver import MerchantResolver
from src.leakage_audit import LeakageAuditor, FeatureMetadata

def test_canonical_schema():
    """Gate 2: Canonical schema passes."""
    txn = Transaction(
        user_id="u123",
        timestamp=datetime.now(),
        amount=100.50,
        direction="debit",
        merchant_raw="STARBUCKS STORE 12"
    )
    assert txn.user_id == "u123"
    assert txn.direction == "debit"
    
    d = txn.to_dict()
    assert "timestamp" in d
    assert "merchant_raw" in d
    assert d["currency"] == "INR"

def test_merchant_resolver():
    """Gate 2: Merchant resolution behaves correctly."""
    resolver = MerchantResolver()
    
    # Must fail if not fitted
    with pytest.raises(ValueError):
        resolver.resolve("STARBUCKS")
        
    resolver.fit(["Starbucks Store 12", "UBER EATS"])
    
    # Exact match
    key, state = resolver.resolve("Starbucks Store 12")
    assert key == "Starbucks Store 12"
    assert state == "KNOWN_MERCHANT"
    
    # Normalized match
    key, state = resolver.resolve("uber eats txn12345")
    assert key == "UBER EATS"
    assert state == "KNOWN_MERCHANT"
    
    # Unseen
    key, state = resolver.resolve("Amazon India")
    assert state == "UNKNOWN_MERCHANT"

def test_leakage_auditor():
    """Gate 3: Feature leakage audit passes/fails correctly."""
    auditor = LeakageAuditor()
    
    # Valid feature
    auditor.register_feature(FeatureMetadata(
        feature_name="rolling_7d_spend",
        source="transactions",
        calculation_window="7d",
        available_at="transaction_time",
        uses_future_data=False
    ))
    
    assert auditor.audit() is True
    
    # Invalid feature (leakage)
    auditor.register_feature(FeatureMetadata(
        feature_name="total_lifetime_spend",
        source="transactions",
        calculation_window="lifetime",
        available_at="future",
        uses_future_data=True
    ))
    
    with pytest.raises(RuntimeError) as exc:
        auditor.audit()
    assert "Leakage detected" in str(exc.value)
