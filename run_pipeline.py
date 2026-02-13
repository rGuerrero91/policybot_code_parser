"""
HCPCS Inference Service v1 — CLI Entrypoint.

Usage:
    python run_pipeline.py --input policies_cleaned.csv --output inferred_codes.json
    python run_pipeline.py --input policies_cleaned.csv --output inferred_codes.json --method llm_openai --limit 5
"""
import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HCPCS Inference Service v1 — infer medical procedure codes from policy text",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to policies CSV (e.g., policies_cleaned.csv)",
    )
    parser.add_argument(
        "--output", default="inferred_codes.json",
        help="Output JSON path (default: inferred_codes.json)",
    )
    parser.add_argument(
        "--hcpcs-catalog", default="hcpcs.csv",
        help="Path to HCPCS reference CSV (default: hcpcs.csv)",
    )
    parser.add_argument(
        "--method", default="lexical_tfidf",
        choices=["lexical_tfidf", "llm_openai"],
        help="Inference method to use (default: lexical_tfidf)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N policies (for testing)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Max codes to infer per policy (default: 10)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Minimum confidence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Path to labels CSV for evaluation (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from src.policybot_hcpcs.config import PipelineConfig
        from src.policybot_hcpcs.pipeline import run_pipeline

        config = PipelineConfig(
            input=args.input,
            output=args.output,
            hcpcs_catalog=args.hcpcs_catalog,
            method=args.method,
            limit=args.limit,
            top_k=args.top_k,
            threshold=args.threshold,
            labels=args.labels,
        )
        run_pipeline(config)
    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
