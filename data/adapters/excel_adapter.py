import pandas as pd
from pathlib import Path
from ..canonical_schema import Transaction, validate_dataframe, data_quality_report

def load_excel(file_path: str | Path) -> pd.DataFrame:
    """Load an Excel file (.xlsx/.xls) and map it to the canonical schema.
    The sheet name can be provided via the ``sheet_name`` argument; by default the
    first sheet is used.
    """
    df = pd.read_excel(file_path, sheet_name=0)
    # Ensure required columns exist
    required = ["user_id", "timestamp", "amount"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Excel missing required column '{col}'")
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
