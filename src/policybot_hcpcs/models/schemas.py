"""
Pydantic v2 models for the HCPCS inference pipeline.

These models define the API contract for v1 of the inference service.
All models use strict typing and validation constraints.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.policybot_hcpcs.config import PIPELINE_VERSION, SCHEMA_VERSION


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class CodeSystem(str, Enum):
    """Supported medical code systems."""
    HCPCS = "HCPCS"
    CPT = "CPT"
    ICD10 = "ICD-10"


class InferenceMethodType(str, Enum):
    """Registry of known inference method identifiers."""
    LEXICAL_TFIDF = "lexical_tfidf"
    LLM_OPENAI = "llm_openai"
    LLM_MOCK = "llm_mock"
    # v2 slots (defined for forward-compatibility, not yet implemented):
    RAG_EMBEDDING = "rag_embedding"
    ENSEMBLE = "ensemble"


class EvidenceType(str, Enum):
    """How evidence was gathered."""
    TEXT_MATCH = "text_match"
    TFIDF_SIMILARITY = "tfidf_similarity"
    LLM_COMPLETION = "llm_completion"
    # v2 slots:
    RAG_RETRIEVAL = "rag_retrieval"
    HUMAN_ANNOTATION = "human_annotation"


# ──────────────────────────────────────────────
# Input Models
# ──────────────────────────────────────────────

class PolicyInput(BaseModel):
    """A single policy document to be processed."""
    policy_id: str = Field(..., description="Unique policy identifier")
    policy_text: str = Field(..., min_length=1, description="Cleaned policy text")
    policy_name: Optional[str] = Field(None, description="Human-readable policy name")
    plan_name: Optional[str] = Field(None, description="Insurance plan name")
    plan_state: Optional[str] = Field(None, description="US state abbreviation")
    policy_type: Optional[str] = Field(None, description="e.g., MEDICAL_POLICY")
    source_url: Optional[str] = Field(None, description="Original document URL")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata passthrough for extensibility",
    )


class HCPCSCode(BaseModel):
    """A single entry from the HCPCS reference catalog."""
    code: str = Field(..., description="HCPCS/CPT code, e.g., 'J3031'")
    description: str = Field(..., description="Short description of the code")


# ──────────────────────────────────────────────
# Evidence & Provenance
# ──────────────────────────────────────────────

class TextSpan(BaseModel):
    """Character offsets into the original policy text."""
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    text: str = Field(..., description="The extracted snippet")

    @model_validator(mode="after")
    def validate_span(self) -> TextSpan:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting a code inference."""
    evidence_type: EvidenceType
    source: str = Field(
        ...,
        description="Where this evidence came from, e.g., 'policy_text', 'hcpcs_catalog', 'openai_gpt-4o-mini'",
    )
    snippet: Optional[str] = Field(None, description="Relevant text excerpt")
    span: Optional[TextSpan] = Field(None, description="Location in source text")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Raw similarity score")


class Provenance(BaseModel):
    """
    Full audit trail for a single inference.

    Given the same inputs_hash and parameters with a deterministic method,
    the result can be reproduced.
    """
    pipeline_version: str = Field(
        default=PIPELINE_VERSION,
        description="Semantic version of this pipeline",
    )
    method: InferenceMethodType = Field(
        ...,
        description="Which inference method produced this result",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this inference was generated (UTC)",
    )
    inputs_hash: str = Field(
        ...,
        description="SHA-256 hash of (policy_text + hcpcs_catalog_version) for reproducibility",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters, e.g., {'top_k': 10, 'threshold': 0.1}",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Supporting evidence items",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional reproducibility data, e.g., model version, prompt template hash",
    )


# ──────────────────────────────────────────────
# Inference Output
# ──────────────────────────────────────────────

class CodeInference(BaseModel):
    """A single inferred code with confidence and full audit trail."""
    code: str = Field(..., description="The inferred code, e.g., 'J3031'")
    code_system: CodeSystem = Field(
        default=CodeSystem.HCPCS,
        description="Which code system this belongs to",
    )
    code_description: Optional[str] = Field(
        None,
        description="Human-readable description from HCPCS catalog",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score, 0 = no confidence, 1 = certain",
    )
    justification: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of why this code was inferred",
    )
    provenance: Provenance = Field(
        ...,
        description="Full audit/reproducibility metadata",
    )


class PolicyOutput(BaseModel):
    """Complete inference result for a single policy."""
    policy_id: str
    policy_name: Optional[str] = None
    inferences: list[CodeInference] = Field(
        default_factory=list,
        description="All inferred codes, sorted by confidence descending",
    )

    @field_validator("inferences")
    @classmethod
    def sort_by_confidence(cls, v: list[CodeInference]) -> list[CodeInference]:
        return sorted(v, key=lambda x: x.confidence, reverse=True)


# ──────────────────────────────────────────────
# Top-Level Pipeline Output (the JSON file)
# ──────────────────────────────────────────────

class PipelineRunMetadata(BaseModel):
    """Top-level metadata about the entire pipeline run."""
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Output schema version for consumer compatibility checks",
    )
    run_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    method: InferenceMethodType
    input_file: str
    hcpcs_catalog_file: str
    total_policies: int
    total_inferences: int
    parameters: dict[str, Any] = Field(default_factory=dict)


class InferenceOutput(BaseModel):
    """
    The root object serialized to inferred_codes.json.
    This is the v1 API contract.
    """
    run_metadata: PipelineRunMetadata
    results: list[PolicyOutput]


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def compute_inputs_hash(policy_text: str, catalog_hash: str) -> str:
    """Deterministic hash for reproducibility tracking."""
    payload = f"{policy_text}|{catalog_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_catalog_hash(codes: list[HCPCSCode]) -> str:
    """Hash the entire HCPCS catalog for versioning."""
    sorted_entries = sorted(f"{c.code}:{c.description}" for c in codes)
    payload = "|".join(sorted_entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
