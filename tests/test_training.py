"""Tests for model training, evaluation, and promotion."""

import numpy as np
import pandas as pd
import pytest

from services.data_generator import generate_churn_dataset
from services.training import (
    CATEGORICAL_FEATURES,
    CHALLENGER_MODEL_NAME,
    MODEL_NAME,
    NUMERIC_FEATURES,
    TARGET,
    compare_models,
    compute_metrics,
    engineer_features,
    find_optimal_threshold,
    predict_proba,
    train_challenger,
    train_champion,
)
from services.training.models.baseline_model import build_champion_pipeline
from services.training.models.challenger_model import build_challenger_pipeline

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
def trained_champion(processed_df, feature_cols):
    X = processed_df[feature_cols]
    y = processed_df[TARGET].values
    return train_champion(X, y)


@pytest.fixture()
def trained_challenger(processed_df, feature_cols):
    X = processed_df[feature_cols]
    y = processed_df[TARGET].values
    return train_challenger(X, y)


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

    def test_train_returns_fitted_pipeline(self, trained_champion):
        """train_champion should return a fitted pipeline."""
        clf = trained_champion.named_steps["classifier"]
        assert hasattr(clf, "classes_")

    def test_predict_proba_shape(self, trained_champion, processed_df, feature_cols):
        """predict_proba should return one probability per row."""
        X = processed_df[feature_cols]
        probs = predict_proba(trained_champion, X)
        assert probs.shape == (len(X),)

    def test_probabilities_in_range(self, trained_champion, processed_df, feature_cols):
        """All probabilities should be between 0 and 1."""
        X = processed_df[feature_cols]
        probs = predict_proba(trained_champion, X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_model_learns_signal(self, trained_champion, processed_df, feature_cols):
        """Model should produce ROC-AUC > 0.5 (better than random)."""
        X = processed_df[feature_cols]
        y = processed_df[TARGET].values
        probs = predict_proba(trained_champion, X)
        metrics = compute_metrics(y, probs)
        assert metrics["roc_auc"] > 0.5, (
            f"ROC-AUC {metrics['roc_auc']} is not better than random"
        )


# ---------------------------------------------------------------------------
# Challenger model
# ---------------------------------------------------------------------------
class TestChallengerModel:
    def test_pipeline_builds(self):
        """Pipeline should be constructable without errors."""
        pipeline = build_challenger_pipeline()
        assert pipeline is not None

    def test_model_name(self):
        assert CHALLENGER_MODEL_NAME == "random_forest_v1"

    def test_train_returns_fitted_pipeline(self, trained_challenger):
        """train_challenger should return a fitted pipeline."""
        clf = trained_challenger.named_steps["classifier"]
        assert hasattr(clf, "estimators_")

    def test_predict_proba_shape(self, trained_challenger, processed_df, feature_cols):
        """predict_proba should return one probability per row."""
        X = processed_df[feature_cols]
        probs = predict_proba(trained_challenger, X)
        assert probs.shape == (len(X),)

    def test_probabilities_in_range(self, trained_challenger, processed_df, feature_cols):
        """All probabilities should be between 0 and 1."""
        X = processed_df[feature_cols]
        probs = predict_proba(trained_challenger, X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_model_learns_signal(self, trained_challenger, processed_df, feature_cols):
        """Model should produce ROC-AUC > 0.5 (better than random)."""
        X = processed_df[feature_cols]
        y = processed_df[TARGET].values
        probs = predict_proba(trained_challenger, X)
        metrics = compute_metrics(y, probs)
        assert metrics["roc_auc"] > 0.5, (
            f"ROC-AUC {metrics['roc_auc']} is not better than random"
        )


# ---------------------------------------------------------------------------
# Promotion logic
# ---------------------------------------------------------------------------
class TestPromotion:
    def test_challenger_promoted_when_better(self):
        """Challenger should be promoted when it beats all three gates."""
        champion = {"roc_auc": 0.70, "recall": 0.60, "precision": 0.30}
        challenger = {"roc_auc": 0.72, "recall": 0.70, "precision": 0.25}
        policy = {"min_roc_auc_improvement": 0.01, "min_recall_threshold": 0.65, "min_precision_threshold": 0.15}

        result = compare_models(champion, challenger, policy=policy)
        assert result["promoted"] is True
        assert result["winner"] == "challenger"

    def test_champion_retained_when_auc_insufficient(self):
        """Champion stays if challenger AUC improvement is too small."""
        champion = {"roc_auc": 0.70, "recall": 0.60, "precision": 0.30}
        challenger = {"roc_auc": 0.705, "recall": 0.70, "precision": 0.25}
        policy = {"min_roc_auc_improvement": 0.01, "min_recall_threshold": 0.65, "min_precision_threshold": 0.15}

        result = compare_models(champion, challenger, policy=policy)
        assert result["promoted"] is False
        assert result["winner"] == "champion"

    def test_champion_retained_when_recall_too_low(self):
        """Champion stays if challenger recall is below the floor."""
        champion = {"roc_auc": 0.70, "recall": 0.60, "precision": 0.30}
        challenger = {"roc_auc": 0.75, "recall": 0.50, "precision": 0.25}
        policy = {"min_roc_auc_improvement": 0.01, "min_recall_threshold": 0.65, "min_precision_threshold": 0.15}

        result = compare_models(champion, challenger, policy=policy)
        assert result["promoted"] is False
        assert result["winner"] == "champion"

    def test_champion_retained_when_precision_too_low(self):
        """Champion stays if challenger precision is below the floor."""
        champion = {"roc_auc": 0.70, "recall": 0.60, "precision": 0.30}
        challenger = {"roc_auc": 0.75, "recall": 0.70, "precision": 0.10}
        policy = {"min_roc_auc_improvement": 0.01, "min_recall_threshold": 0.65, "min_precision_threshold": 0.15}

        result = compare_models(champion, challenger, policy=policy)
        assert result["promoted"] is False
        assert result["winner"] == "champion"

    def test_comparison_output_keys(self):
        """Comparison result should contain all expected keys."""
        champion = {"roc_auc": 0.70, "recall": 0.60, "precision": 0.30}
        challenger = {"roc_auc": 0.72, "recall": 0.70, "precision": 0.25}
        policy = {"min_roc_auc_improvement": 0.01, "min_recall_threshold": 0.65, "min_precision_threshold": 0.15}

        result = compare_models(champion, challenger, policy=policy)
        expected_keys = {
            "champion_roc_auc", "challenger_roc_auc", "roc_auc_delta",
            "challenger_recall", "challenger_precision",
            "min_roc_auc_improvement", "min_recall_threshold", "min_precision_threshold",
            "meets_auc_requirement", "meets_recall_requirement", "meets_precision_requirement",
            "promoted", "winner",
        }
        assert set(result.keys()) == expected_keys

    def test_exact_threshold_promoted(self):
        """Challenger at exactly champion + delta should be promoted."""
        champion = {"roc_auc": 0.70, "recall": 0.60, "precision": 0.30}
        challenger = {"roc_auc": 0.71, "recall": 0.65, "precision": 0.20}
        policy = {"min_roc_auc_improvement": 0.01, "min_recall_threshold": 0.65, "min_precision_threshold": 0.15}

        result = compare_models(champion, challenger, policy=policy)
        assert result["promoted"] is True  # 0.71 >= 0.70 + 0.01


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


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------
class TestThresholdOptimization:
    def test_f1_strategy_returns_valid_threshold(self):
        """F1 strategy should return a threshold between 0 and 1."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_prob, strategy="f1")
        assert 0.0 <= threshold <= 1.0

    def test_recall_strategy_meets_target(self):
        """Recall strategy should find a threshold achieving the target recall."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_prob, strategy="recall", min_recall=0.65)
        # Apply the threshold and check recall
        y_pred = (y_prob >= threshold).astype(int)
        recall = y_pred[y_true == 1].sum() / y_true.sum()
        assert recall >= 0.65

    def test_recall_strategy_maximizes_precision(self):
        """Recall strategy should pick the highest threshold that still meets recall."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # Lower min_recall allows a higher threshold
        threshold_low = find_optimal_threshold(y_true, y_prob, strategy="recall", min_recall=0.40)
        threshold_high = find_optimal_threshold(y_true, y_prob, strategy="recall", min_recall=0.80)
        assert threshold_low >= threshold_high

    def test_f1_strategy_better_than_default(self, trained_champion, processed_df, feature_cols):
        """Optimized threshold should produce better F1 than default 0.5."""
        X = processed_df[feature_cols]
        y = processed_df[TARGET].values
        probs = predict_proba(trained_champion, X)

        default_metrics = compute_metrics(y, probs, threshold=0.5)
        optimal_threshold = find_optimal_threshold(y, probs, strategy="f1")
        optimal_metrics = compute_metrics(y, probs, threshold=optimal_threshold)

        assert optimal_metrics["f1"] >= default_metrics["f1"]

    def test_recall_strategy_on_imbalanced_data(self, trained_champion, processed_df, feature_cols):
        """Recall strategy should achieve target recall on realistic imbalanced data."""
        X = processed_df[feature_cols]
        y = processed_df[TARGET].values
        probs = predict_proba(trained_champion, X)

        threshold = find_optimal_threshold(y, probs, strategy="recall", min_recall=0.65)
        metrics = compute_metrics(y, probs, threshold=threshold)
        assert metrics["recall"] >= 0.65

    def test_invalid_strategy_raises(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="Unknown strategy"):
            find_optimal_threshold(y_true, y_prob, strategy="invalid")

    def test_threshold_saved_in_metrics(self):
        """compute_metrics should include the threshold used."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_metrics(y_true, y_prob, threshold=0.3)
        assert metrics["threshold"] == 0.3
