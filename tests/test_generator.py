"""Tests for the synthetic data generator."""

import pandas as pd

from services.data_generator import generate_churn_dataset

REQUIRED_COLUMNS = [
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


def test_output_shape():
    """Dataset has the requested number of rows."""
    df = generate_churn_dataset(n_customers=100, seed=1)
    assert df.shape[0] == 100


def test_required_columns_exist():
    """All spec-required columns are present."""
    df = generate_churn_dataset(n_customers=50, seed=1)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_no_duplicate_customer_ids():
    """Each customer_id is unique within a snapshot."""
    df = generate_churn_dataset(n_customers=500, seed=1)
    assert df["customer_id"].nunique() == len(df)


def test_churn_label_binary():
    """Churn label must be 0 or 1."""
    df = generate_churn_dataset(n_customers=500, seed=1)
    assert set(df["churned_30d"].unique()).issubset({0, 1})


def test_churn_rate_realistic():
    """Churn rate should be realistic: between 1% and 8%."""
    df = generate_churn_dataset(n_customers=5000, seed=42)
    rate = df["churned_30d"].mean()
    assert 0.01 < rate <= 0.08, f"Churn rate {rate:.2%} outside expected range [1%-8%]"


def test_reproducibility():
    """Same seed produces identical output."""
    df1 = generate_churn_dataset(n_customers=100, seed=99)
    df2 = generate_churn_dataset(n_customers=100, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    """Different seeds produce different data."""
    df1 = generate_churn_dataset(n_customers=100, seed=1)
    df2 = generate_churn_dataset(n_customers=100, seed=2)
    assert not df1["churned_30d"].equals(df2["churned_30d"])


def test_snapshot_date_applied():
    """Snapshot date column matches the requested date."""
    df = generate_churn_dataset(n_customers=10, snapshot_date="2025-06-15", seed=1)
    assert (df["snapshot_date"] == pd.Timestamp("2025-06-15")).all()
