import argparse
import pandas as pd
import sys
import pathlib
# Ensure project modules are importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.categorizer import HybridTransactionCategorizer

# Optional PDF handling – requires `pdfplumber` (install via pip if needed)
try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

def _load_pdf_descriptions(pdf_path: str) -> pd.DataFrame:
    """Extract a DataFrame with a 'description' column from a PDF statement.

    The function attempts to locate a table in the PDF and guess the column that
    best matches a transaction description. It works for typical Google Pay
    statements where the table contains columns like 'Description', 'Merchant',
    or 'Details'. If no table is found, it falls back to extracting all text and
    using each line as a description.
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber is required for PDF processing. Install via `pip install pdfplumber`.")
    pages = pdfplumber.open(pdf_path)
    rows = []
    for page in pages.pages:
        table = page.extract_table()
        if table:
            header = [h.strip().lower() for h in table[0]]
            # Find index of description-like column
            desc_idx = None
            for idx, col in enumerate(header):
                if "description" in col or "merchant" in col or "details" in col:
                    desc_idx = idx
                    break
            if desc_idx is None:
                # Use first column as fallback
                desc_idx = 0
            for row in table[1:]:
                if len(row) > desc_idx:
                    rows.append({"description": row[desc_idx]})
    if not rows:
        # Fallback: treat each line of text from all pages as a description
        for pg in pages.pages:
            txt = pg.extract_text() or ""
            for line in txt.splitlines():
                line = line.strip()
                if line and len(line) > 5:
                    rows.append({"description": line})
        # If still empty, raise a friendly error
        if not rows:
            raise ValueError("No transaction descriptions could be extracted from the PDF.")
    pages.close()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Predict categories for unlabeled transactions using the hybrid model.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file with a 'description' column.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to write predictions.")
    args = parser.parse_args()
    # Determine input type (CSV or PDF)
    if args.input.lower().endswith('.pdf'):
        df = _load_pdf_descriptions(args.input)
    else:
        df = pd.read_csv(args.input)
        if 'description' not in df.columns:
            raise ValueError("Input CSV must contain a 'description' column.")
    # Load the hybrid model (expects it to be saved already)
    cat = HybridTransactionCategorizer.load()
    preds = cat.predict(df['description'].astype(str).tolist())
    df['predicted_category'] = preds
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
