"""TF-IDF + literal match inference method (primary v1 method)."""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.policybot_hcpcs.config import (
    LITERAL_CODE_MATCH_BOOST,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.policybot_hcpcs.inference.base import BaseInferenceMethod
from src.policybot_hcpcs.inference.registry import register_method
from src.policybot_hcpcs.inference.utils import find_literal_codes
from src.policybot_hcpcs.models.schemas import (
    CodeInference,
    CodeSystem,
    EvidenceItem,
    EvidenceType,
    HCPCSCode,
    InferenceMethodType,
    PolicyInput,
    Provenance,
    TextSpan,
    compute_catalog_hash,
    compute_inputs_hash,
)

logger = logging.getLogger(__name__)


@dataclass
class _Candidate:
    """Internal helper to track candidate codes during inference."""
    code: str
    description: str
    tfidf_score: float
    literal_match: TextSpan | None


@register_method(InferenceMethodType.LEXICAL_TFIDF)
class LexicalTfidfMethod(BaseInferenceMethod):
    """
    Infers HCPCS codes using TF-IDF cosine similarity between
    policy text and HCPCS code descriptions, boosted by literal
    code matches found in the policy text.
    """

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._hcpcs_matrix = None
        self._catalog: list[HCPCSCode] = []
        self._catalog_hash: str = ""
        self._code_set: set[str] = set()
        self._code_desc_map: dict[str, str] = {}

    @property
    def method_type(self) -> InferenceMethodType:
        return InferenceMethodType.LEXICAL_TFIDF

    def initialize(self, hcpcs_catalog: list[HCPCSCode], **kwargs: Any) -> None:
        """Fit TF-IDF vectorizer on HCPCS descriptions."""
        self._catalog = hcpcs_catalog
        self._catalog_hash = compute_catalog_hash(hcpcs_catalog)
        self._code_set = {c.code for c in hcpcs_catalog}
        self._code_desc_map = {c.code: c.description for c in hcpcs_catalog}

        descriptions = [c.description for c in hcpcs_catalog]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
        )
        self._hcpcs_matrix = self._vectorizer.fit_transform(descriptions)
        logger.info(
            f"TF-IDF vectorizer fitted on {len(descriptions)} HCPCS descriptions "
            f"(matrix shape: {self._hcpcs_matrix.shape})"
        )

    def infer(
        self,
        policy: PolicyInput,
        parameters: dict[str, Any] | None = None,
    ) -> list[CodeInference]:
        params = parameters or {}
        top_k = params.get("top_k", 10)
        threshold = params.get("threshold", 0.05)

        if self._vectorizer is None:
            raise RuntimeError("Method not initialized. Call initialize() first.")

        inputs_hash = compute_inputs_hash(policy.policy_text, self._catalog_hash)

        # TF-IDF similarity
        policy_vec = self._vectorizer.transform([policy.policy_text])
        similarities = cosine_similarity(policy_vec, self._hcpcs_matrix).flatten()

        # Find literal code matches in the policy text
        literal_matches = find_literal_codes(policy.policy_text, self._code_set)

        # Build candidates: combine TF-IDF top-k with literal matches
        candidates: dict[str, _Candidate] = {}

        # Add TF-IDF candidates
        top_indices = np.argsort(similarities)[::-1][:top_k]
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue
            code_obj = self._catalog[idx]
            candidates[code_obj.code] = _Candidate(
                code=code_obj.code,
                description=code_obj.description,
                tfidf_score=score,
                literal_match=None,
            )

        # Add/boost literal match candidates
        for code_str, span in literal_matches.items():
            if code_str in candidates:
                candidates[code_str].literal_match = span
            else:
                candidates[code_str] = _Candidate(
                    code=code_str,
                    description=self._code_desc_map.get(code_str, "Unknown"),
                    tfidf_score=0.0,
                    literal_match=span,
                )

        # Convert candidates to CodeInference objects
        inferences = []
        for cand in candidates.values():
            confidence = cand.tfidf_score
            evidence: list[EvidenceItem] = []

            if cand.tfidf_score > 0:
                evidence.append(EvidenceItem(
                    evidence_type=EvidenceType.TFIDF_SIMILARITY,
                    source="hcpcs_catalog",
                    snippet=cand.description,
                    score=cand.tfidf_score,
                ))

            if cand.literal_match is not None:
                confidence = min(confidence + LITERAL_CODE_MATCH_BOOST, 1.0)
                evidence.append(EvidenceItem(
                    evidence_type=EvidenceType.TEXT_MATCH,
                    source="policy_text",
                    snippet=cand.literal_match.text,
                    span=cand.literal_match,
                ))

            justification = self._build_justification(cand)

            inferences.append(CodeInference(
                code=cand.code,
                code_system=CodeSystem.HCPCS,
                code_description=cand.description,
                confidence=round(confidence, 4),
                justification=justification,
                provenance=Provenance(
                    method=self.method_type,
                    inputs_hash=inputs_hash,
                    parameters={"top_k": top_k, "threshold": threshold},
                    evidence=evidence,
                ),
            ))

        return inferences

    @staticmethod
    def _build_justification(cand: _Candidate) -> str:
        parts = []
        if cand.tfidf_score > 0:
            parts.append(
                f"HCPCS {cand.code} ({cand.description}) matched via "
                f"TF-IDF similarity (score={cand.tfidf_score:.3f})"
            )
        if cand.literal_match is not None:
            parts.append(f"Code '{cand.code}' found literally in policy text")
        return ". ".join(parts) + "."
