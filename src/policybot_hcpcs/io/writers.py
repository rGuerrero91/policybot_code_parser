"""Serialize inference output to JSON."""
import logging
from pathlib import Path

from src.policybot_hcpcs.models.schemas import InferenceOutput

logger = logging.getLogger(__name__)


def write_output(output: InferenceOutput, path: str) -> None:
    """Serialize the full output to JSON with ISO datetime formatting."""
    json_str = output.model_dump_json(indent=2)
    Path(path).write_text(json_str, encoding="utf-8")
    logger.info(f"Wrote {len(output.results)} results to {path}")
