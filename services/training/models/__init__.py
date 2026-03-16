"""Models sub-package — public API."""

from services.training.models.baseline_model import (
    MODEL_NAME,
    build_champion_pipeline,
    predict_proba,
    train_champion,
)
from services.training.models.challenger_model import (
    CHALLENGER_MODEL_NAME,
    build_challenger_pipeline,
    train_challenger,
)
from services.training.models.evaluation import compute_metrics, find_optimal_threshold

__all__ = [
    "MODEL_NAME",
    "CHALLENGER_MODEL_NAME",
    "build_champion_pipeline",
    "build_challenger_pipeline",
    "compute_metrics",
    "find_optimal_threshold",
    "predict_proba",
    "train_champion",
    "train_challenger",
]
