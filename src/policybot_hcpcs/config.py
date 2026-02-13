"""Pipeline-wide constants and defaults."""
from dataclasses import dataclass
from typing import Optional

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD = 0.05
DEFAULT_METHOD = "lexical_tfidf"

# TF-IDF parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# LLM parameters
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
LLM_MAX_POLICY_CHARS = 12000

# Confidence boosting
LITERAL_CODE_MATCH_BOOST = 0.3


@dataclass
class PipelineConfig:
    """Typed configuration for a pipeline run, decoupled from argparse."""
    input: str
    output: str = "inferred_codes.json"
    hcpcs_catalog: str = "hcpcs.csv"
    method: str = DEFAULT_METHOD
    limit: Optional[int] = None
    top_k: int = DEFAULT_TOP_K
    threshold: float = DEFAULT_THRESHOLD
    labels: Optional[str] = None
