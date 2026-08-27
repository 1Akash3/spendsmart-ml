import os
from pathlib import Path
import pandas as pd

from .canonical_schema import validate_dataframe, data_quality_report
from .adapters.csv_adapter import load_csv
from .adapters.excel_adapter import load_excel
from .adapters.json_adapter import load_json
from .adapters.pdf_adapter import load_pdf_statement

SUPPORTED_EXTENSIONS = {
    ".csv": load_csv,
    ".tsv": load_csv,
    ".xls": load_excel,
    ".xlsx": load_excel,
    ".json": load_json,
    ".pdf": load_pdf_statement,
}


def ingest_file(file_path: str | Path) -> pd.DataFrame:
    """Ingest a single source file and return a validated DataFrame.

    The function selects the appropriate adapter based on file extension. If the
    extension is not recognized, a ``ValueError`` is raised.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: {list(SUPPORTED_EXTENSIONS)}")
    loader = SUPPORTED_EXTENSIONS[ext]
    df = loader(path)
    # Ensure the DataFrame has all canonical columns (adds None for any missing)
    canonical_cols = [
        "user_id",
        "timestamp",
        "amount",
        "direction",
        "merchant_raw",
        "transaction_id",
        "source",
        "currency",
        "account_id",
        "category",
        "confidence",
    ]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = None
    # Final validation (already performed in adapters, but re‑run for safety)
    df = validate_dataframe(df)
    # Attach a basic data‑quality report for downstream use
    df.attrs["data_quality_report"] = data_quality_report(df)
    return df


def ingest_directory(dir_path: str | Path, recursive: bool = True) -> pd.DataFrame:
    """Ingest all supported files within a directory (optionally recursively).
    Returns a concatenated DataFrame of all records.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"'{dir_path}' is not a directory")
    pattern = "**/*" if recursive else "*"
    frames = []
    for file in dir_path.glob(pattern):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            frames.append(ingest_file(file))
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        raise ValueError("No supported files found in the directory")
