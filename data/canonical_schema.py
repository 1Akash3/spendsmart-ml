# spendsmart_ml/data/canonical_schema.py
"""Canonical transaction schema and validation utilities.

Defines the standard representation for all transaction records across the pipeline.
The schema is deliberately permissive – missing fields are allowed and will be
filled with ``None`` (or pandas ``NaN``) so that downstream models never crash.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, Field, validator


class Transaction(BaseModel):
    """Pydantic model for a single transaction.

    Required fields are those that most downstream components depend on. All
    other fields are optional and default to ``None`` when not present.
    """

    user_id: str = Field(..., description="Unique identifier for the user (anonymized)")
    timestamp: datetime = Field(..., description="Transaction timestamp (timezone‑aware if possible)")
    amount: float = Field(..., description="Signed transaction amount (positive for inflow, negative for outflow)")
    direction: Optional[str] = Field(
        None,
        description="Direction of cash‑flow, e.g. 'debit' or 'credit'. May be inferred from sign of amount.",
    )
    merchant_raw: Optional[str] = Field(None, description="Original merchant description as extracted from source")
    transaction_id: Optional[str] = Field(None, description="Source‑specific transaction identifier if available")
    source: Optional[str] = Field(None, description="Origin of the data (e.g., 'google_pay', 'bank_statement')")
    currency: Optional[str] = Field(None, description="ISO 4217 currency code, e.g. 'INR'")
    account_id: Optional[str] = Field(
        None, description="Identifier for the account or instrument used, if available"
    )
    category: Optional[str] = Field(None, description="Ground‑truth category label (if known)")
    confidence: Optional[float] = Field(
        None, description="Confidence score for inferred category (0.0‑1.0)"
    )

    @validator("direction")
    def _infer_direction(cls, v: Optional[str], values):  # noqa: D401
        """Infer direction from amount when missing.

        If ``direction`` is not provided we derive it from the sign of ``amount``.
        """
        if v is None:
            amt = values.get("amount")
            if amt is not None:
                return "credit" if amt > 0 else "debit"
        return v

    @validator("timestamp", pre=True)
    def _parse_timestamp(cls, v):  # noqa: D401
        """Accept strings or ``datetime`` objects; coerce to ``datetime``.
        """
        if isinstance(v, datetime):
            return v
        try:
            return pd.to_datetime(v, utc=True).to_pydantic()
        except Exception as exc:
            raise ValueError(f"Unable to parse timestamp '{v}': {exc}")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the canonical schema.

    Returns a new DataFrame with an additional ``_validation_errors`` column
    containing any Pydantic validation error messages. Rows without errors have
    ``None`` in that column.
    """
    errors: List[Optional[str]] = []
    for _, row in df.iterrows():
        try:
            Transaction(**row.to_dict())
            errors.append(None)
        except Exception as exc:
            errors.append(str(exc))
    df = df.copy()
    df["_validation_errors"] = errors
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a simple data‑quality summary.

    The report is a dictionary that can be serialized to JSON/YAML and includes:
    - total rows
    - number of rows with validation errors
    - percentage of missing values per column
    """
    total = len(df)
    error_rows = df["_validation_errors"].notna().sum()
    missing_pct = (df.isna().sum() / total * 100).to_dict()
    return {
        "total_rows": total,
        "rows_with_errors": int(error_rows),
        "error_rate_pct": round(error_rows / total * 100, 2) if total else 0,
        "missing_percentage_per_column": {k: round(v, 2) for k, v in missing_pct.items()},
    }

