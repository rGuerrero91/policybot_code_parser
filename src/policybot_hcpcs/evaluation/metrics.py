"""Evaluation metrics for comparing inferred codes against ground truth."""
import logging

from src.policybot_hcpcs.models.schemas import PolicyOutput

logger = logging.getLogger(__name__)


def evaluate(
    results: list[PolicyOutput],
    labels: dict[str, list[str]],
    k: int = 10,
) -> dict[str, float]:
    """
    Compare inferred codes against ground truth labels.

    Args:
        results: Pipeline output (list of PolicyOutput).
        labels: Dict mapping policy_id -> list of true HCPCS codes.
        k: Number of top predictions to consider.

    Returns:
        Dict with recall@k, precision@k, and coverage metrics.
    """
    total_recall = 0.0
    total_precision = 0.0
    policies_with_match = 0
    evaluated = 0

    for result in results:
        true_codes = set(labels.get(result.policy_id, []))
        if not true_codes:
            continue

        predicted_codes = {inf.code for inf in result.inferences[:k]}
        evaluated += 1

        if not predicted_codes:
            continue

        hits = true_codes & predicted_codes
        recall = len(hits) / len(true_codes)
        precision = len(hits) / len(predicted_codes)

        total_recall += recall
        total_precision += precision

        if hits:
            policies_with_match += 1

    if evaluated == 0:
        logger.warning("No policies with ground truth labels found for evaluation")
        return {"recall@k": 0.0, "precision@k": 0.0, "coverage": 0.0}

    return {
        "recall@k": total_recall / evaluated,
        "precision@k": total_precision / evaluated,
        "coverage": policies_with_match / evaluated,
        "evaluated_policies": float(evaluated),
    }
