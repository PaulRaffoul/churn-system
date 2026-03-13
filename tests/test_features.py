"""Tests for feature engineering."""

import pandas as pd
import pytest

from services.data_generator import generate_churn_dataset
from services.training.features.feature_engineering import (
    ENGINEERED_FEATURES,
    engineer_features,
)


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return generate_churn_dataset(n_customers=500, seed=7)


@pytest.fixture()
def feat_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(raw_df)


# ---------------------------------------------------------------------------
# General checks
# ---------------------------------------------------------------------------
class TestGeneralFeatures:
    def test_original_columns_preserved(self, raw_df, feat_df):
        """Feature engineering should not drop any original columns."""
        for col in raw_df.columns:
            assert col in feat_df.columns, f"Original column {col} was dropped"

    def test_engineered_columns_added(self, feat_df):
        """All engineered feature columns must be present."""
        for col in ENGINEERED_FEATURES:
            assert col in feat_df.columns, f"Missing engineered column: {col}"

    def test_row_count_unchanged(self, raw_df, feat_df):
        """Feature engineering must not add or remove rows."""
        assert len(feat_df) == len(raw_df)

    def test_no_nulls_in_engineered_features(self, feat_df):
        """Engineered features should have no null values."""
        for col in ENGINEERED_FEATURES:
            assert feat_df[col].isna().sum() == 0, f"{col} has null values"

    def test_does_not_mutate_input(self, raw_df):
        """engineer_features should return a new DataFrame, not mutate input."""
        original_cols = list(raw_df.columns)
        engineer_features(raw_df)
        assert list(raw_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Individual feature checks
# ---------------------------------------------------------------------------
class TestEngagementScore:
    def test_range_0_to_1(self, feat_df):
        assert feat_df["engagement_score"].min() >= 0
        assert feat_df["engagement_score"].max() <= 1

    def test_high_activity_means_high_score(self):
        """Customer with max logins + sessions should score ~1.0."""
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["monthly_logins"] = 60
        df["avg_session_minutes"] = 120.0
        result = engineer_features(df)
        assert result["engagement_score"].iloc[0] == pytest.approx(1.0)

    def test_zero_activity_means_zero_score(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["monthly_logins"] = 0
        df["avg_session_minutes"] = 0.0
        result = engineer_features(df)
        assert result["engagement_score"].iloc[0] == pytest.approx(0.0)


class TestBillingRiskScore:
    def test_range_0_to_1(self, feat_df):
        assert feat_df["billing_risk_score"].min() >= 0
        assert feat_df["billing_risk_score"].max() <= 1

    def test_max_risk(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["payment_failures_last_90d"] = 5
        df["plan_changes_last_6m"] = 4
        result = engineer_features(df)
        assert result["billing_risk_score"].iloc[0] == pytest.approx(1.0)


class TestSupportBurdenScore:
    def test_range_0_to_1(self, feat_df):
        assert feat_df["support_burden_score"].min() >= 0
        assert feat_df["support_burden_score"].max() <= 1

    def test_ten_tickets_is_max(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["support_tickets_last_30d"] = 10
        result = engineer_features(df)
        assert result["support_burden_score"].iloc[0] == pytest.approx(1.0)


class TestInactivityFlag:
    def test_values_are_binary(self, feat_df):
        assert set(feat_df["inactivity_flag"].unique()).issubset({0, 1})

    def test_flag_set_when_inactive(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["days_since_last_activity"] = 30
        result = engineer_features(df)
        assert result["inactivity_flag"].iloc[0] == 1

    def test_flag_clear_when_active(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["days_since_last_activity"] = 3
        result = engineer_features(df)
        assert result["inactivity_flag"].iloc[0] == 0


class TestTenureBucket:
    def test_valid_categories(self, feat_df):
        valid = {"short", "medium", "long"}
        assert set(feat_df["tenure_bucket"].unique()).issubset(valid)

    def test_short_tenure(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["tenure_months"] = 3
        result = engineer_features(df)
        assert result["tenure_bucket"].iloc[0] == "short"

    def test_medium_tenure(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["tenure_months"] = 12
        result = engineer_features(df)
        assert result["tenure_bucket"].iloc[0] == "medium"

    def test_long_tenure(self):
        df = generate_churn_dataset(n_customers=1, seed=1)
        df["tenure_months"] = 24
        result = engineer_features(df)
        assert result["tenure_bucket"].iloc[0] == "long"


class TestPriceToUsageRatio:
    def test_positive_values(self, feat_df):
        """Ratio must always be positive (price > 0 and denominator > 0)."""
        assert (feat_df["price_to_usage_ratio"] > 0).all()

    def test_higher_price_lower_usage_means_higher_ratio(self):
        df = generate_churn_dataset(n_customers=2, seed=1)
        df.loc[0, "monthly_price"] = 39.99
        df.loc[0, "avg_session_minutes"] = 1.0
        df.loc[1, "monthly_price"] = 9.99
        df.loc[1, "avg_session_minutes"] = 100.0
        result = engineer_features(df)
        assert (
            result["price_to_usage_ratio"].iloc[0]
            > result["price_to_usage_ratio"].iloc[1]
        )
