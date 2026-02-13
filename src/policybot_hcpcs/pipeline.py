"""Pipeline orchestration — load, infer, aggregate, write."""
import logging

from src.policybot_hcpcs.config import PipelineConfig
from src.policybot_hcpcs.inference.registry import discover_methods, get_method
from src.policybot_hcpcs.io.loaders import load_hcpcs_catalog, load_labels, load_policies
from src.policybot_hcpcs.io.writers import write_output
from src.policybot_hcpcs.models.schemas import (
    InferenceMethodType,
    InferenceOutput,
    PipelineRunMetadata,
    PolicyOutput,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full inference pipeline."""

    # Discover registered methods
    discover_methods()

    # 1. Load data
    logger.info("Loading data...")
    policies = load_policies(config.input, limit=config.limit)
    hcpcs_catalog = load_hcpcs_catalog(config.hcpcs_catalog)

    if not policies:
        raise RuntimeError(f"No valid policies loaded from {config.input}")

    # 2. Initialize inference method
    method_type = InferenceMethodType(config.method)
    method = get_method(method_type)

    logger.info(f"Initializing method: {method_type.value}")
    method.initialize(hcpcs_catalog)

    parameters = {"top_k": config.top_k, "threshold": config.threshold}

    # 3. Run inference per policy
    results: list[PolicyOutput] = []
    total_inferences = 0

    for i, policy in enumerate(policies):
        try:
            inferences = method.infer(policy, parameters)
            results.append(PolicyOutput(
                policy_id=policy.policy_id,
                policy_name=policy.policy_name,
                inferences=inferences,
            ))
            total_inferences += len(inferences)
        except Exception as e:
            logger.error(f"Error processing policy {policy.policy_id}: {e}")
            results.append(PolicyOutput(
                policy_id=policy.policy_id,
                policy_name=policy.policy_name,
                inferences=[],
            ))

        if (i + 1) % 10 == 0 or (i + 1) == len(policies):
            logger.info(f"Processed {i + 1}/{len(policies)} policies")

    # 4. Aggregate output
    output = InferenceOutput(
        run_metadata=PipelineRunMetadata(
            method=method_type,
            input_file=config.input,
            hcpcs_catalog_file=config.hcpcs_catalog,
            total_policies=len(results),
            total_inferences=total_inferences,
            parameters=parameters,
        ),
        results=results,
    )

    # 5. Write output
    write_output(output, config.output)
    logger.info(
        f"Pipeline complete: {len(results)} policies, "
        f"{total_inferences} total inferences"
    )

    # 6. Evaluate (optional)
    if config.labels:
        _run_evaluation(results, config.labels, config.top_k)


def _run_evaluation(
    results: list[PolicyOutput], labels_path: str, k: int
) -> None:
    """Run evaluation against ground truth labels."""
    from src.policybot_hcpcs.evaluation.metrics import evaluate

    labels = load_labels(labels_path)
    metrics = evaluate(results, labels, k=k)

    print("\n--- Evaluation Metrics ---")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("--------------------------\n")
