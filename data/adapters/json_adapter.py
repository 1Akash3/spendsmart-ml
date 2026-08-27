import json
import pandas as pd
from pathlib import Path
from ..canonical_schema import Transaction, validate_dataframe, data_quality_report

def load_json(file_path: str | Path) -> pd.DataFrame:
    """Load a JSON file containing a list of transaction records.
    The JSON should be an array of objects where each object may or may not
    contain all canonical fields.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        # Support a dict with a key like "transactions"
        raw = raw.get("transactions", [])
    df = pd.DataFrame(raw)
    # Ensure required columns
    required = ["user_id", "timestamp", "amount"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"JSON missing required field '{col}'")
    optional = [
        "direction",
        "merchant_raw",
        "transaction_id",
        "source",
        "currency",
        "account_id",
        "category",
        "confidence",
    ]
    for col in optional:
        if col not in df.columns:
            df[col] = None
    df = validate_dataframe(df)
    report = data_quality_report(df)
    df.attrs["data_quality_report"] = report
    return df
