"""Champion vs Challenger promotion logic.

Compares two models on validation metrics and applies the promotion
policy from monitoring/promotion_policy.json.

The promotion decision uses ROC-AUC only (threshold-independent) to
keep model comparison fair. Threshold optimization for recall happens
at scoring time, not during model selection.
"""

import json
from pathlib import Path


DEFAULT_POLICY_PATH = Path("monitoring/promotion_policy.json")


def load_promotion_policy(path: Path = DEFAULT_POLICY_PATH) -> dict:
    """Load the promotion policy from disk.

    Args:
        path: Path to promotion_policy.json.

    Returns:
        Dict with min_roc_auc_improvement.
    """
    return json.loads(path.read_text())


def compare_models(
    champion_metrics: dict,
    challenger_metrics: dict,
    policy: dict | None = None,
) -> dict:
    """Compare champion and challenger metrics and decide on promotion.

    Promotion gate: challenger ROC-AUC >= champion ROC-AUC + min_improvement.

    ROC-AUC is threshold-independent — it measures ranking quality across
    all possible thresholds. This avoids the problem of threshold-optimized
    metrics (recall, precision) being artificially equal between models.

    Args:
        champion_metrics: Dict with at least roc_auc key.
        challenger_metrics: Dict with at least roc_auc key.
        policy: Promotion policy dict. If None, loads from default path.

    Returns:
        Dict with comparison details and a 'promoted' boolean.
    """
    if policy is None:
        policy = load_promotion_policy()

    min_improvement = policy["min_roc_auc_improvement"]

    champ_auc = champion_metrics["roc_auc"]
    chall_auc = challenger_metrics["roc_auc"]

    auc_delta = round(chall_auc - champ_auc, 4)
    promoted = chall_auc >= champ_auc + min_improvement

    return {
        "champion_roc_auc": champ_auc,
        "challenger_roc_auc": chall_auc,
        "roc_auc_delta": auc_delta,
        "min_roc_auc_improvement": min_improvement,
        "promoted": promoted,
        "winner": "challenger" if promoted else "champion",
    }
