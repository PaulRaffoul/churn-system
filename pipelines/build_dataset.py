"""Thin orchestrator — build a processed dataset from raw data.

Reads raw data, validates it, runs feature engineering, and saves
the result to data/processed/.

Usage:
    uv run python -m pipelines.build_dataset
    uv run python -m pipelines.build_dataset --input data/raw/churn_data.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from services.data_generator import validate_or_raise
from services.training import engineer_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed dataset.")
    parser.add_argument(
        "--input",
        default="data/raw/churn_data.parquet",
        help="Path to raw parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed data.",
    )
    args = parser.parse_args()

    # Step 1: Load raw data
    print(f"Loading raw data from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Step 2: Validate
    print("Validating raw data ...")
    validate_or_raise(df, dataset_type="raw")
    print("  Validation PASSED.")

    # Step 3: Feature engineering
    print("Running feature engineering ...")
    df_processed = engineer_features(df)
    print(f"  Added features. Now {len(df_processed.columns)} columns.")

    # Step 4: Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "churn_processed.parquet"
    df_processed.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
