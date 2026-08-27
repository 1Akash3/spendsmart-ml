from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Transaction:
    """Canonical Transaction Schema.
    
    All inputs (GPay, Kaggle, synthetic, CSV) must be mapped to this schema
    before feature engineering or model inference.
    """
    user_id: str
    timestamp: datetime
    amount: float
    direction: str  # 'debit' or 'credit'
    merchant_raw: str
    
    # These fields can be populated by downstream pipeline stages (e.g., normalization)
    merchant_normalized: Optional[str] = None
    merchant_key: Optional[str] = None
    
    source: str = "unknown"
    currency: str = "INR"
    
    transaction_id: Optional[str] = None
    category: Optional[str] = None
    
    # Model confidences
    category_confidence: Optional[float] = None
    ingestion_confidence: Optional[float] = None
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "amount": self.amount,
            "direction": self.direction,
            "merchant_raw": self.merchant_raw,
            "merchant_normalized": self.merchant_normalized,
            "merchant_key": self.merchant_key,
            "source": self.source,
            "currency": self.currency,
            "transaction_id": self.transaction_id,
            "category": self.category,
            "category_confidence": self.category_confidence,
            "ingestion_confidence": self.ingestion_confidence
        }
