"""Thin orchestrator — train champion and challenger models.

Loads processed data, splits by time, trains both models, compares
them using the promotion policy, and saves artifacts.

Usage:
    uv run python -m pipelines.train_pipeline
"""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from services.training import (
    CATEGORICAL_FEATURES,
    CHALLENGER_MODEL_NAME,
    MODEL_NAME,
    NUMERIC_FEATURES,
    TARGET,
    compare_models,
    compute_metrics,
    predict_proba,
    train_challenger,
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


def _save_model(pipeline, metrics: dict, model_name: str, model_dir: Path) -> None:
    """Save a model and its metrics to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    metrics_out = {"model_name": model_name, **metrics}
    (model_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train champion and challenger models.")
    parser.add_argument(
        "--input",
        default="data/processed/churn_processed.parquet",
        help="Path to processed parquet file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="model_artifacts",
        help="Root directory for model artifacts.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest is validation).",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    # Step 1: Load processed data
    print(f"Loading processed data from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows.")

    # Step 2: Time-based split
    print(f"Splitting data (train={args.train_frac:.0%}, val={1 - args.train_frac:.0%}) ...")
    train_df, val_df = time_based_split(df, train_frac=args.train_frac)
    print(f"  Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].values
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET].values

    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Val churn rate:   {y_val.mean():.1%}")

    # Step 3: Train champion
    print("\n--- Champion: Logistic Regression ---")
    champion_pipeline = train_champion(X_train, y_train)
    champion_probs = predict_proba(champion_pipeline, X_val)
    champion_metrics = compute_metrics(y_val, champion_probs)

    champion_metrics["train_rows"] = len(train_df)
    champion_metrics["val_rows"] = len(val_df)
    champion_metrics["train_churn_rate"] = round(float(y_train.mean()), 4)
    champion_metrics["val_churn_rate"] = round(float(y_val.mean()), 4)

    print(f"  ROC-AUC:   {champion_metrics['roc_auc']}")
    print(f"  Precision: {champion_metrics['precision']}")
    print(f"  Recall:    {champion_metrics['recall']}")
    print(f"  F1:        {champion_metrics['f1']}")

    _save_model(champion_pipeline, champion_metrics, MODEL_NAME, artifacts_dir / "champion")
    print(f"  Saved to {artifacts_dir / 'champion'}")

    # Step 4: Train challenger
    print("\n--- Challenger: Random Forest ---")
    challenger_pipeline = train_challenger(X_train, y_train)
    challenger_probs = predict_proba(challenger_pipeline, X_val)
    challenger_metrics = compute_metrics(y_val, challenger_probs)

    challenger_metrics["train_rows"] = len(train_df)
    challenger_metrics["val_rows"] = len(val_df)
    challenger_metrics["train_churn_rate"] = round(float(y_train.mean()), 4)
    challenger_metrics["val_churn_rate"] = round(float(y_val.mean()), 4)

    print(f"  ROC-AUC:   {challenger_metrics['roc_auc']}")
    print(f"  Precision: {challenger_metrics['precision']}")
    print(f"  Recall:    {challenger_metrics['recall']}")
    print(f"  F1:        {challenger_metrics['f1']}")

    _save_model(challenger_pipeline, challenger_metrics, CHALLENGER_MODEL_NAME, artifacts_dir / "challenger")
    print(f"  Saved to {artifacts_dir / 'challenger'}")

    # Step 5: Compare and promote
    print("\n--- Promotion Decision ---")
    comparison = compare_models(champion_metrics, challenger_metrics)
    print(f"  ROC-AUC delta:          {comparison['roc_auc_delta']:+.4f}")
    print(f"  Meets AUC threshold:    {comparison['meets_auc_requirement']}")
    print(f"  Meets recall floor:     {comparison['meets_recall_requirement']}")
    print(f"  Meets precision floor:  {comparison['meets_precision_requirement']}")
    print(f"  Winner:                 {comparison['winner']}")

    # Save comparison artifact
    comparison_path = artifacts_dir / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"  Comparison saved to {comparison_path}")

    # If challenger wins, copy it to champion directory
    if comparison["promoted"]:
        print("\n  Challenger PROMOTED to champion!")
        champion_dir = artifacts_dir / "champion"
        challenger_dir = artifacts_dir / "challenger"
        # Overwrite champion with challenger artifacts
        shutil.copy2(challenger_dir / "model.joblib", champion_dir / "model.joblib")
        shutil.copy2(challenger_dir / "metrics.json", champion_dir / "metrics.json")
        print(f"  Champion artifacts updated in {champion_dir}")
    else:
        print("\n  Champion RETAINED — challenger did not meet promotion criteria.")


if __name__ == "__main__":
    main()
