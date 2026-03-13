"""Thin orchestrator — validate a raw dataset file.

Usage:
    uv run python pipelines/validate_data.py
    uv run python pipelines/validate_data.py --path data/raw/churn_data.parquet
"""

import argparse
import sys

import pandas as pd

from services.data_generator import validate_or_raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a raw churn dataset.")
    parser.add_argument(
        "--path",
        default="data/raw/churn_data.parquet",
        help="Path to the parquet file to validate.",
    )
    parser.add_argument(
        "--type",
        default="raw",
        choices=["raw", "scored"],
        help="Dataset type: 'raw' or 'scored'.",
    )
    args = parser.parse_args()

    print(f"Loading {args.path} ...")
    df = pd.read_parquet(args.path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    print(f"Running {args.type} validation ...")
    try:
        validate_or_raise(df, dataset_type=args.type)
        print("  PASSED — all validation checks OK.")
    except ValueError as exc:
        print(f"  FAILED\n{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
