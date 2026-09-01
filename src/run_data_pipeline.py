"""CLI entrypoint for the SpendSmart V3 data pipeline.

Usage:
    python -m src.run_data_pipeline --mode smoke
    python -m src.run_data_pipeline --mode development
    python -m src.run_data_pipeline --mode final
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Path bootstrap
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpendSmart V3 Research Data Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke",
        choices=["smoke", "development", "final"],
        help="Pipeline execution mode (default: smoke)",
    )
    args = parser.parse_args()

    try:
        metadata = run(args.mode)
        print("\n" + "=" * 60)
        print(f"  PIPELINE SUCCESS: {args.mode.upper()}")
        print(f"  Dataset: {metadata['quality_summary']['rows']:,} transactions")
        print(f"  Users:   {metadata['quality_summary']['users']:,}")
        print(f"  Hash:    {metadata['dataset_hash'][:16]}...")
        print("=" * 60)
    except Exception as e:
        print(f"\n[PIPELINE FAILED] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
