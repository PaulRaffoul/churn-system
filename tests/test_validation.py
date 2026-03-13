"""Tests for the data validation layer."""

import pandas as pd
import pytest

from services.data_generator import generate_churn_dataset
from services.data_generator.validator import (
    validate_or_raise,
    validate_raw_dataset,
    validate_scored_dataset,
)


# ---------------------------------------------------------------------------
# Fixture: a known-good dataset from the generator
# ---------------------------------------------------------------------------
@pytest.fixture()
def good_df() -> pd.DataFrame:
    return generate_churn_dataset(n_customers=200, seed=99)


# ---------------------------------------------------------------------------
# Raw dataset — happy path
# ---------------------------------------------------------------------------
class TestRawValidation:
    def test_valid_dataset_passes(self, good_df: pd.DataFrame):
        """Generator output should pass validation with zero errors."""
        errors = validate_raw_dataset(good_df)
        assert errors == []

    def test_empty_dataframe_fails(self):
        errors = validate_raw_dataset(pd.DataFrame())
        assert any("empty" in e.lower() for e in errors)

    def test_missing_columns_detected(self, good_df: pd.DataFrame):
        df = good_df.drop(columns=["churned_30d", "age"])
        errors = validate_raw_dataset(df)
        assert any("Missing required columns" in e for e in errors)
        assert any("age" in e for e in errors)
        assert any("churned_30d" in e for e in errors)

    def test_invalid_churn_label_detected(self, good_df: pd.DataFrame):
        df = good_df.copy()
        df.loc[0, "churned_30d"] = 2  # invalid
        errors = validate_raw_dataset(df)
        assert any("churned_30d" in e and "invalid" in e for e in errors)

    def test_null_churn_label_detected(self, good_df: pd.DataFrame):
        df = good_df.copy()
        df.loc[0, "churned_30d"] = None
        errors = validate_raw_dataset(df)
        assert any("null" in e.lower() for e in errors)

    def test_duplicate_customer_snapshot_detected(self, good_df: pd.DataFrame):
        # Append a duplicate row
        df = pd.concat([good_df, good_df.iloc[[0]]], ignore_index=True)
        errors = validate_raw_dataset(df)
        assert any("duplicate" in e.lower() for e in errors)

    def test_negative_age_detected(self, good_df: pd.DataFrame):
        df = good_df.copy()
        df.loc[0, "age"] = -5
        errors = validate_raw_dataset(df)
        assert any("age" in e for e in errors)

    def test_negative_tenure_detected(self, good_df: pd.DataFrame):
        df = good_df.copy()
        df.loc[0, "tenure_months"] = -1
        errors = validate_raw_dataset(df)
        assert any("tenure_months" in e for e in errors)

    def test_nonpositive_price_detected(self, good_df: pd.DataFrame):
        df = good_df.copy()
        df.loc[0, "monthly_price"] = 0
        errors = validate_raw_dataset(df)
        assert any("monthly_price" in e for e in errors)

    def test_excessive_churn_rate_detected(self, good_df: pd.DataFrame):
        """Churn rate above 8% should trigger a validation error."""
        df = good_df.copy()
        # Force 50% churn — way above the 8% cap
        df["churned_30d"] = [1, 0] * (len(df) // 2)
        errors = validate_raw_dataset(df)
        assert any("Churn rate" in e for e in errors)

    def test_valid_churn_rate_passes(self, good_df: pd.DataFrame):
        """Generator with realistic bias should produce a valid churn rate."""
        errors = validate_raw_dataset(good_df)
        assert not any("Churn rate" in e for e in errors)


# ---------------------------------------------------------------------------
# Scored dataset
# ---------------------------------------------------------------------------
class TestScoredValidation:
    @pytest.fixture()
    def scored_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "customer_id": ["CUST_000001", "CUST_000002"],
                "snapshot_date": pd.Timestamp("2025-01-01"),
                "churn_probability": [0.2, 0.8],
                "churn_prediction": [0, 1],
                "risk_band": ["low", "high"],
                "model_name": ["logistic_v1", "logistic_v1"],
            }
        )

    def test_valid_scored_passes(self, scored_df: pd.DataFrame):
        errors = validate_scored_dataset(scored_df)
        assert errors == []

    def test_probability_out_of_range(self, scored_df: pd.DataFrame):
        df = scored_df.copy()
        df.loc[0, "churn_probability"] = 1.5
        errors = validate_scored_dataset(df)
        assert any("churn_probability" in e for e in errors)

    def test_missing_scored_columns(self, scored_df: pd.DataFrame):
        df = scored_df.drop(columns=["risk_band"])
        errors = validate_scored_dataset(df)
        assert any("Missing" in e and "risk_band" in e for e in errors)

    def test_empty_scored_dataset(self):
        errors = validate_scored_dataset(pd.DataFrame())
        assert any("empty" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# validate_or_raise convenience function
# ---------------------------------------------------------------------------
class TestValidateOrRaise:
    def test_valid_data_does_not_raise(self, good_df: pd.DataFrame):
        validate_or_raise(good_df, dataset_type="raw")  # should not raise

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="Validation failed"):
            validate_or_raise(pd.DataFrame(), dataset_type="raw")

    def test_unknown_dataset_type_raises(self, good_df: pd.DataFrame):
        with pytest.raises(ValueError, match="Unknown dataset_type"):
            validate_or_raise(good_df, dataset_type="bogus")
