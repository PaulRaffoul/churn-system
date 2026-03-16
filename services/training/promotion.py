"""Champion vs Challenger promotion logic.

Compares two models on validation metrics and applies the promotion
policy from monitoring/promotion_policy.json.

Churn-specific rationale:
- Recall is prioritized because missing a churner (false negative) costs
  their entire lifetime value, while a false alarm costs only a retention offer.
- Precision floor prevents models that flag everyone to game recall.
- ROC-AUC delta ensures overall ranking quality improvement.
"""

import json
from pathlib import Path


DEFAULT_POLICY_PATH = Path("monitoring/promotion_policy.json")


def load_promotion_policy(path: Path = DEFAULT_POLICY_PATH) -> dict:
    """Load the promotion policy from disk.

    Args:
        path: Path to promotion_policy.json.

    Returns:
        Dict with min_roc_auc_improvement, min_recall_threshold,
        and min_precision_threshold.
    """
    return json.loads(path.read_text())


def compare_models(
    champion_metrics: dict,
    challenger_metrics: dict,
    policy: dict | None = None,
) -> dict:
    """Compare champion and challenger metrics and decide on promotion.

    Three gates must be passed for promotion:
    1. ROC-AUC: challenger >= champion + min_improvement
    2. Recall floor: challenger recall >= threshold (catch enough churners)
    3. Precision floor: challenger precision >= threshold (don't flag everyone)

    Args:
        champion_metrics: Dict with at least roc_auc, recall, precision keys.
        challenger_metrics: Dict with at least roc_auc, recall, precision keys.
        policy: Promotion policy dict. If None, loads from default path.

    Returns:
        Dict with comparison details and a 'promoted' boolean.
    """
    if policy is None:
        policy = load_promotion_policy()

    min_improvement = policy["min_roc_auc_improvement"]
    min_recall = policy["min_recall_threshold"]
    min_precision = policy["min_precision_threshold"]

    champ_auc = champion_metrics["roc_auc"]
    chall_auc = challenger_metrics["roc_auc"]
    chall_recall = challenger_metrics["recall"]
    chall_precision = challenger_metrics["precision"]

    auc_delta = round(chall_auc - champ_auc, 4)
    meets_auc = chall_auc >= champ_auc + min_improvement
    meets_recall = chall_recall >= min_recall
    meets_precision = chall_precision >= min_precision

    promoted = meets_auc and meets_recall and meets_precision

    return {
        "champion_roc_auc": champ_auc,
        "challenger_roc_auc": chall_auc,
        "roc_auc_delta": auc_delta,
        "challenger_recall": chall_recall,
        "challenger_precision": chall_precision,
        "min_roc_auc_improvement": min_improvement,
        "min_recall_threshold": min_recall,
        "min_precision_threshold": min_precision,
        "meets_auc_requirement": meets_auc,
        "meets_recall_requirement": meets_recall,
        "meets_precision_requirement": meets_precision,
        "promoted": promoted,
        "winner": "challenger" if promoted else "champion",
    }
