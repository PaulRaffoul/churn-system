"""CLI runner for synthetic data generation.

Usage:
    uv run python -m services.data_generator.run
    uv run python -m services.data_generator.run --n-customers 10000 --seed 123
"""

import argparse
import sys
from pathlib import Path

from services.data_generator import generate_churn_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--n-customers", type=int, default=5000, help="Number of customers")
    parser.add_argument("--snapshot-date", type=str, default="2025-01-01", help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    print(f"Generating {args.n_customers} customers for snapshot {args.snapshot_date} (seed={args.seed})...")

    df = generate_churn_dataset(
        n_customers=args.n_customers,
        snapshot_date=args.snapshot_date,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "churn_data.parquet"

    df.to_parquet(output_path, index=False)

    # Print summary
    churn_rate = df["churned_30d"].mean()
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {churn_rate:.1%}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
