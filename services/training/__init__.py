"""Training Service — public API."""

from services.training.features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    engineer_features,
)
from services.training.models import (
    MODEL_NAME,
    build_champion_pipeline,
    compute_metrics,
    predict_proba,
    train_champion,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "ENGINEERED_FEATURES",
    "NUMERIC_FEATURES",
    "TARGET",
    "engineer_features",
    "MODEL_NAME",
    "build_champion_pipeline",
    "compute_metrics",
    "predict_proba",
    "train_champion",
]
