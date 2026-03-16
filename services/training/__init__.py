"""Training Service — public API."""

from services.training.features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    engineer_features,
)
from services.training.models import (
    CHALLENGER_MODEL_NAME,
    MODEL_NAME,
    build_champion_pipeline,
    build_challenger_pipeline,
    compute_metrics,
    predict_proba,
    train_champion,
    train_challenger,
)
from services.training.promotion import compare_models

__all__ = [
    "CATEGORICAL_FEATURES",
    "ENGINEERED_FEATURES",
    "NUMERIC_FEATURES",
    "TARGET",
    "engineer_features",
    "CHALLENGER_MODEL_NAME",
    "MODEL_NAME",
    "build_champion_pipeline",
    "build_challenger_pipeline",
    "compute_metrics",
    "predict_proba",
    "train_champion",
    "train_challenger",
    "compare_models",
]
