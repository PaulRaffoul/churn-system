"""Data validation layer.

Validates datasets before they enter downstream pipelines.
Catches schema problems, duplicates, and value-range violations
early so failures are loud and obvious.
"""

import pandas as pd

# Maximum acceptable churn rate (8%).  Anything above this signals
# a data generation bug or an upstream data quality problem.
MAX_CHURN_RATE = 0.08

# The 16 columns the generator must produce
RAW_REQUIRED_COLUMNS = [
    "customer_id",
    "snapshot_date",
    "signup_date",
    "country",
    "age",
    "subscription_plan",
    "monthly_price",
    "acquisition_channel",
    "tenure_months",
    "monthly_logins",
    "avg_session_minutes",
    "support_tickets_last_30d",
    "payment_failures_last_90d",
    "days_since_last_activity",
    "plan_changes_last_6m",
    "churned_30d",
]

# Columns required in scored output
SCORED_REQUIRED_COLUMNS = [
    "customer_id",
    "snapshot_date",
    "churn_probability",
    "churn_prediction",
    "risk_band",
    "model_name",
]


def validate_raw_dataset(df: pd.DataFrame) -> list[str]:
    """Validate a raw churn dataset (training or scoring input).

    Runs all checks and returns a list of error messages.
    An empty list means the data is valid.

    Args:
        df: DataFrame to validate.

    Returns:
        List of validation error strings (empty = valid).
    """
    errors: list[str] = []

    # --- Check 1: row count > 0 ---
    if len(df) == 0:
        errors.append("Dataset is empty (0 rows).")
        return errors  # no point checking further

    # --- Check 2: required columns exist ---
    missing = set(RAW_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    # --- Check 3: churn label is binary {0, 1} ---
    if "churned_30d" in df.columns:
        unique_labels = set(df["churned_30d"].dropna().unique())
        if not unique_labels.issubset({0, 1}):
            bad_values = unique_labels - {0, 1}
            errors.append(
                f"churned_30d contains invalid values: {bad_values}. "
                "Expected only 0 and 1."
            )
        if df["churned_30d"].isna().any():
            n_null = int(df["churned_30d"].isna().sum())
            errors.append(f"churned_30d has {n_null} null values.")

    # --- Check 4: no duplicate customer_id + snapshot_date ---
    if "customer_id" in df.columns and "snapshot_date" in df.columns:
        dup_count = df.duplicated(subset=["customer_id", "snapshot_date"]).sum()
        if dup_count > 0:
            errors.append(
                f"Found {dup_count} duplicate (customer_id, snapshot_date) rows."
            )

    # --- Check 5: numeric ranges ---
    if "age" in df.columns:
        if (df["age"] < 0).any() or (df["age"] > 150).any():
            errors.append("age contains values outside [0, 150].")

    if "tenure_months" in df.columns:
        if (df["tenure_months"] < 0).any():
            errors.append("tenure_months contains negative values.")

    if "monthly_price" in df.columns:
        if (df["monthly_price"] <= 0).any():
            errors.append("monthly_price contains non-positive values.")

    # --- Check 6: churn rate must not exceed MAX_CHURN_RATE ---
    if "churned_30d" in df.columns:
        churn_rate = df["churned_30d"].mean()
        if churn_rate > MAX_CHURN_RATE:
            errors.append(
                f"Churn rate {churn_rate:.1%} exceeds maximum allowed "
                f"({MAX_CHURN_RATE:.0%}). Check generator calibration."
            )

    return errors


def validate_scored_dataset(df: pd.DataFrame) -> list[str]:
    """Validate a scored output dataset.

    Args:
        df: DataFrame to validate.

    Returns:
        List of validation error strings (empty = valid).
    """
    errors: list[str] = []

    if len(df) == 0:
        errors.append("Scored dataset is empty (0 rows).")
        return errors

    # --- Check required columns ---
    missing = set(SCORED_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required scored columns: {sorted(missing)}")

    # --- Check probabilities in [0, 1] ---
    if "churn_probability" in df.columns:
        probs = df["churn_probability"]
        if probs.isna().any():
            errors.append(
                f"churn_probability has {int(probs.isna().sum())} null values."
            )
        if (probs < 0).any() or (probs > 1).any():
            errors.append("churn_probability contains values outside [0, 1].")

    return errors


def validate_or_raise(
    df: pd.DataFrame,
    dataset_type: str = "raw",
) -> None:
    """Validate and raise ValueError if any checks fail.

    This is the function pipelines should call — it either passes
    silently or blows up with all validation errors listed.

    Args:
        df: DataFrame to validate.
        dataset_type: "raw" for training/input data, "scored" for model output.

    Raises:
        ValueError: If any validation checks fail.
    """
    if dataset_type == "raw":
        errors = validate_raw_dataset(df)
    elif dataset_type == "scored":
        errors = validate_scored_dataset(df)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Use 'raw' or 'scored'.")

    if errors:
        error_list = "\n  - ".join(errors)
        raise ValueError(f"Validation failed with {len(errors)} error(s):\n  - {error_list}")
