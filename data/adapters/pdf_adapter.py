import pandas as pd
from pathlib import Path
import pdfplumber
from ..canonical_schema import Transaction, validate_dataframe, data_quality_report

def load_pdf_statement(file_path: str | Path) -> pd.DataFrame:
    """Parse a bank/statement PDF and convert it to the canonical schema.
    This implementation uses ``pdfplumber`` to extract raw text tables. It attempts
    to locate columns containing date, amount, description, and optionally a
    transaction identifier. The function is tolerant to variations in layout –
    rows that cannot be parsed are logged and excluded, while a data‑quality
    report records the number of rejected rows.
    """
    rows = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Try to extract a table; fallback to raw text line parsing
            table = page.extract_table()
            if table:
                header = [h.strip().lower() for h in table[0]]
                for line in table[1:]:
                    if not any(line):
                        continue
                    row = {h: v.strip() if v else None for h, v in zip(header, line)}
                    rows.append(row)
            else:
                # Very simple fallback: split lines on whitespace and look for patterns
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split("\n"):
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    # naive heuristic: date amount description
                    row = {"date": parts[0], "amount": parts[1], "description": " ".join(parts[2:])}
                    rows.append(row)
    if not rows:
        raise ValueError("No parsable transaction rows found in PDF")
    df = pd.DataFrame(rows)
    # Normalise column names to match canonical schema expectations
    rename_map = {
        "date": "timestamp",
        "description": "merchant_raw",
        "amount": "amount",
        "transaction_id": "transaction_id",
        "category": "category",
    }
    df = df.rename(columns=rename_map)
    # Ensure required columns exist
    required = ["user_id", "timestamp", "amount"]
    for col in required:
        if col not in df.columns:
            if col == "user_id":
                df[col] = None  # will be filled later by downstream processing
            else:
                raise ValueError(f"PDF parser could not locate required column '{col}'")
    # Add optional columns if missing
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
