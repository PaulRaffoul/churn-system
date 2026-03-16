"""Model evaluation metrics and threshold optimization.

Pure functions — take arrays in, return metrics out.
No side effects, no file I/O.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
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
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
    min_recall: float = 0.65,
) -> float:
    """Find the optimal decision threshold for churn prediction.

    Strategies:
        "f1": Threshold that maximizes F1 score — balanced precision/recall.
        "recall": Highest threshold that achieves at least min_recall —
                  churn-appropriate since we prioritize catching churners
                  while keeping precision as high as possible.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_prob: Predicted probabilities of churn (0-1).
        strategy: "f1" or "recall".
        min_recall: Minimum recall target (only used with "recall" strategy).

    Returns:
        Optimal threshold as a float.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns one extra element for precisions/recalls
    # (the last point is precision=1, recall=0 with no threshold)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if len(thresholds) == 0:
        return 0.5

    if strategy == "f1":
        # F1 = 2 * (precision * recall) / (precision + recall)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = precisions + recalls
            f1_scores = np.where(
                denominator > 0,
                2 * (precisions * recalls) / denominator,
                0.0,
            )
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx])

    elif strategy == "recall":
        # Find all thresholds where recall >= min_recall,
        # then pick the highest threshold (maximizes precision)
        valid_mask = recalls >= min_recall
        if not valid_mask.any():
            # If no threshold achieves min_recall, use the lowest threshold
            return float(thresholds[np.argmax(recalls)])
        valid_thresholds = thresholds[valid_mask]
        return float(valid_thresholds.max())

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'f1' or 'recall'.")
