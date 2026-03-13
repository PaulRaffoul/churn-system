"""Model evaluation metrics.

Pure functions — take arrays in, return metrics out.
No side effects, no file I/O.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics for churn prediction.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_prob: Predicted probabilities of churn (0-1).
        threshold: Decision threshold for binary predictions.

    Returns:
        Dict with roc_auc, precision, recall, f1, and threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": threshold,
    }
