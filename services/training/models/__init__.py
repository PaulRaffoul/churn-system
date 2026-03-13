"""Models sub-package — public API."""

from services.training.models.baseline_model import (
    MODEL_NAME,
    build_champion_pipeline,
    predict_proba,
    train_champion,
)
from services.training.models.evaluation import compute_metrics

__all__ = [
    "MODEL_NAME",
    "build_champion_pipeline",
    "compute_metrics",
    "predict_proba",
    "train_champion",
]
