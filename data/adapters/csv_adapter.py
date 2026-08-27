import pandas as pd
from pathlib import Path
from ..canonical_schema import Transaction, validate_dataframe, data_quality_report

def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file and coerce it to the canonical schema.
    Expected columns may include a superset of the canonical fields; missing
    columns are added with ``None`` values.
    """
    df = pd.read_csv(file_path)
    # Ensure required columns exist
    required = ["user_id", "timestamp", "amount"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    # Add optional columns if absent
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
    # Validate and generate quality report (side‑effect stored in df)
    df = validate_dataframe(df)
    report = data_quality_report(df)
    # Attach report as attribute for downstream use
    df.attrs["data_quality_report"] = report
    return df
