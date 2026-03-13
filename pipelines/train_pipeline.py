"""Thin orchestrator — train the champion model.

Loads processed data, splits by time, trains, evaluates,
and saves the model artifact + metrics.

Usage:
    uv run python -m pipelines.train_pipeline
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from services.training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    MODEL_NAME,
    compute_metrics,
    predict_proba,
    train_champion,
)


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by signup_date (oldest → training, newest → validation).

    This avoids data leakage — the model never sees future customers
    during training.

    Args:
        df: Processed DataFrame with signup_date column.
        train_frac: Fraction of data to use for training.

    Returns:
        (train_df, val_df) tuple.
    """
    df_sorted = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the champion model.")
    parser.add_argument(
        "--input",
        default="data/processed/churn_processed.parquet",
        help="Path to processed parquet file.",
    )
    parser.add_argument(
        "--model-dir",
        default="model_artifacts/champion",
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest is validation).",
    )
    args = parser.parse_args()

    # Step 1: Load processed data
    print(f"Loading processed data from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows.")

    # Step 2: Time-based split
    print(f"Splitting data (train={args.train_frac:.0%}, val={1-args.train_frac:.0%}) ...")
    train_df, val_df = time_based_split(df, train_frac=args.train_frac)
    print(f"  Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].values
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET].values

    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Val churn rate:   {y_val.mean():.1%}")

    # Step 3: Train
    print("Training champion model (Logistic Regression) ...")
    pipeline = train_champion(X_train, y_train)
    print("  Training complete.")

    # Step 4: Evaluate
    print("Evaluating on validation set ...")
    y_prob = predict_proba(pipeline, X_val)
    metrics = compute_metrics(y_val, y_prob)
    print(f"  ROC-AUC:   {metrics['roc_auc']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['f1']}")

    # Step 5: Save artifacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"  Model saved to {model_path}")

    metrics_out = {
        "model_name": MODEL_NAME,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_churn_rate": round(float(y_train.mean()), 4),
        "val_churn_rate": round(float(y_val.mean()), 4),
        **metrics,
    }
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
