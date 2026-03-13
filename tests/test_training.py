"""Tests for model training and evaluation."""

import numpy as np
import pandas as pd
import pytest

from services.data_generator import generate_churn_dataset
from services.training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    compute_metrics,
    engineer_features,
    predict_proba,
    train_champion,
)
from services.training.models.baseline_model import build_champion_pipeline

from pipelines.train_pipeline import time_based_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def processed_df() -> pd.DataFrame:
    """A processed dataset ready for training."""
    raw = generate_churn_dataset(n_customers=500, seed=42)
    return engineer_features(raw)


@pytest.fixture()
def feature_cols() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


@pytest.fixture()
def trained_pipeline(processed_df, feature_cols):
    X = processed_df[feature_cols]
    y = processed_df[TARGET].values
    return train_champion(X, y)


# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------
class TestTimeBasedSplit:
    def test_split_sizes(self, processed_df):
        train, val = time_based_split(processed_df, train_frac=0.8)
        assert len(train) == int(len(processed_df) * 0.8)
        assert len(val) == len(processed_df) - len(train)

    def test_train_dates_before_val(self, processed_df):
        """Training data should have earlier signup dates than validation."""
        train, val = time_based_split(processed_df, train_frac=0.8)
        assert train["signup_date"].max() <= val["signup_date"].min()

    def test_no_data_lost(self, processed_df):
        train, val = time_based_split(processed_df, train_frac=0.8)
        assert len(train) + len(val) == len(processed_df)


# ---------------------------------------------------------------------------
# Champion model
# ---------------------------------------------------------------------------
class TestChampionModel:
    def test_pipeline_builds(self):
        """Pipeline should be constructable without errors."""
        pipeline = build_champion_pipeline()
        assert pipeline is not None

    def test_train_returns_fitted_pipeline(self, trained_pipeline):
        """train_champion should return a fitted pipeline."""
        # A fitted pipeline has the classes_ attribute on the classifier
        clf = trained_pipeline.named_steps["classifier"]
        assert hasattr(clf, "classes_")

    def test_predict_proba_shape(self, trained_pipeline, processed_df, feature_cols):
        """predict_proba should return one probability per row."""
        X = processed_df[feature_cols]
        probs = predict_proba(trained_pipeline, X)
        assert probs.shape == (len(X),)

    def test_probabilities_in_range(self, trained_pipeline, processed_df, feature_cols):
        """All probabilities should be between 0 and 1."""
        X = processed_df[feature_cols]
        probs = predict_proba(trained_pipeline, X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_model_learns_signal(self, trained_pipeline, processed_df, feature_cols):
        """Model should produce ROC-AUC > 0.5 (better than random)."""
        X = processed_df[feature_cols]
        y = processed_df[TARGET].values
        probs = predict_proba(trained_pipeline, X)
        metrics = compute_metrics(y, probs)
        assert metrics["roc_auc"] > 0.5, (
            f"ROC-AUC {metrics['roc_auc']} is not better than random"
        )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
class TestEvaluation:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_metrics(y_true, y_prob)
        assert metrics["roc_auc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        metrics = compute_metrics(y_true, y_prob)
        assert metrics["roc_auc"] == 0.0
        assert metrics["recall"] == 0.0

    def test_metrics_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6])
        metrics = compute_metrics(y_true, y_prob)
        expected_keys = {"roc_auc", "precision", "recall", "f1", "threshold"}
        assert set(metrics.keys()) == expected_keys

    def test_custom_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        metrics_default = compute_metrics(y_true, y_prob, threshold=0.5)
        metrics_low = compute_metrics(y_true, y_prob, threshold=0.3)
        # Lower threshold → more positives → higher recall
        assert metrics_low["recall"] >= metrics_default["recall"]
