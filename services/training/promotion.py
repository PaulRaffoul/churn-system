"""Champion vs Challenger promotion logic.

Compares two models on validation metrics and applies the promotion
policy from monitoring/promotion_policy.json.
"""

import json
from pathlib import Path


DEFAULT_POLICY_PATH = Path("monitoring/promotion_policy.json")


def load_promotion_policy(path: Path = DEFAULT_POLICY_PATH) -> dict:
    """Load the promotion policy from disk.

    Args:
        path: Path to promotion_policy.json.

    Returns:
        Dict with min_roc_auc_improvement and min_recall_threshold.
    """
    return json.loads(path.read_text())


def compare_models(
    champion_metrics: dict,
    challenger_metrics: dict,
    policy: dict | None = None,
) -> dict:
    """Compare champion and challenger metrics and decide on promotion.

    Args:
        champion_metrics: Dict with at least roc_auc and recall keys.
        challenger_metrics: Dict with at least roc_auc and recall keys.
        policy: Promotion policy dict. If None, loads from default path.

    Returns:
        Dict with comparison details and a 'promoted' boolean.
    """
    if policy is None:
        policy = load_promotion_policy()

    min_improvement = policy["min_roc_auc_improvement"]
    min_recall = policy["min_recall_threshold"]

    champ_auc = champion_metrics["roc_auc"]
    chall_auc = challenger_metrics["roc_auc"]
    chall_recall = challenger_metrics["recall"]

    auc_delta = round(chall_auc - champ_auc, 4)
    meets_auc = chall_auc >= champ_auc + min_improvement
    meets_recall = chall_recall >= min_recall

    promoted = meets_auc and meets_recall

    return {
        "champion_roc_auc": champ_auc,
        "challenger_roc_auc": chall_auc,
        "roc_auc_delta": auc_delta,
        "challenger_recall": chall_recall,
        "min_roc_auc_improvement": min_improvement,
        "min_recall_threshold": min_recall,
        "meets_auc_requirement": meets_auc,
        "meets_recall_requirement": meets_recall,
        "promoted": promoted,
        "winner": "challenger" if promoted else "champion",
    }
