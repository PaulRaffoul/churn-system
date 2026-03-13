"""Feature engineering for churn prediction.

Transforms raw customer data into model-ready features.
The SAME function is used for both training and scoring
to prevent training-serving skew.
"""

import pandas as pd


# Columns that the feature engineering step adds
ENGINEERED_FEATURES = [
    "engagement_score",
    "billing_risk_score",
    "support_burden_score",
    "inactivity_flag",
    "tenure_bucket",
    "price_to_usage_ratio",
]

# Numeric features the model will consume (raw + engineered)
NUMERIC_FEATURES = [
    "age",
    "monthly_price",
    "tenure_months",
    "monthly_logins",
    "avg_session_minutes",
    "support_tickets_last_30d",
    "payment_failures_last_90d",
    "days_since_last_activity",
    "plan_changes_last_6m",
    "engagement_score",
    "billing_risk_score",
    "support_burden_score",
    "inactivity_flag",
    "price_to_usage_ratio",
]

# Categorical features that will need encoding before modelling
CATEGORICAL_FEATURES = [
    "country",
    "subscription_plan",
    "acquisition_channel",
    "tenure_bucket",
]

# The target column
TARGET = "churned_30d"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a raw churn dataset.

    This function must be called identically for training data and
    scoring data so the model sees consistent feature distributions.

    Args:
        df: Raw DataFrame from the data generator (or loaded parquet).

    Returns:
        A new DataFrame with the original columns plus engineered features.
    """
    out = df.copy()

    # --- engagement_score ---
    # Normalize logins (0-60) and session minutes (0-120) to 0-1, average.
    login_norm = out["monthly_logins"].clip(0, 60) / 60
    session_norm = out["avg_session_minutes"].clip(0, 120) / 120
    out["engagement_score"] = ((login_norm + session_norm) / 2).round(4)

    # --- billing_risk_score ---
    # Normalize payment failures (0-5) and plan changes (0-4) to 0-1, average.
    failure_norm = out["payment_failures_last_90d"].clip(0, 5) / 5
    change_norm = out["plan_changes_last_6m"].clip(0, 4) / 4
    out["billing_risk_score"] = ((failure_norm + change_norm) / 2).round(4)

    # --- support_burden_score ---
    # Normalize support tickets (0-10) to 0-1.
    out["support_burden_score"] = (
        out["support_tickets_last_30d"].clip(0, 10) / 10
    ).round(4)

    # --- inactivity_flag ---
    # 1 if customer has been inactive for more than 14 days.
    out["inactivity_flag"] = (out["days_since_last_activity"] > 14).astype(int)

    # --- tenure_bucket ---
    # short (<6 months), medium (6-18 months), long (18+ months).
    out["tenure_bucket"] = pd.cut(
        out["tenure_months"],
        bins=[-1, 6, 18, 999],
        labels=["short", "medium", "long"],
    )

    # --- price_to_usage_ratio ---
    # High ratio = paying a lot relative to usage = churn risk.
    # +1 to avoid division by zero.
    out["price_to_usage_ratio"] = (
        out["monthly_price"] / (out["avg_session_minutes"] + 1)
    ).round(4)

    return out
