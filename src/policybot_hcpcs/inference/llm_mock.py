"""Mock LLM method — regex-based code extraction (no API key required)."""
import logging
from typing import Any

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
    compute_catalog_hash,
    compute_inputs_hash,
)

logger = logging.getLogger(__name__)

MOCK_CONFIDENCE = 0.6


@register_method(InferenceMethodType.LLM_MOCK)
class LlmMockMethod(BaseInferenceMethod):
    """
    Simulates LLM inference by extracting literal code patterns
    from policy text via regex and cross-referencing the HCPCS catalog.

    Useful for testing the pipeline without an API key.
    """

    def __init__(self) -> None:
        self._code_map: dict[str, str] = {}
        self._code_set: set[str] = set()
        self._catalog_hash: str = ""

    @property
    def method_type(self) -> InferenceMethodType:
        return InferenceMethodType.LLM_MOCK

    def initialize(self, hcpcs_catalog: list[HCPCSCode], **kwargs: Any) -> None:
        self._code_map = {c.code: c.description for c in hcpcs_catalog}
        self._code_set = set(self._code_map)
        self._catalog_hash = compute_catalog_hash(hcpcs_catalog)
        logger.info(f"Mock method initialized with {len(self._code_map)} codes")

    def infer(
        self,
        policy: PolicyInput,
        parameters: dict[str, Any] | None = None,
    ) -> list[CodeInference]:
        params = parameters or {}
        top_k = params.get("top_k", 10)
        inputs_hash = compute_inputs_hash(policy.policy_text, self._catalog_hash)

        found_codes = find_literal_codes(policy.policy_text, self._code_set)

        inferences = []
        for code, span in list(found_codes.items())[:top_k]:
            description = self._code_map[code]
            inferences.append(CodeInference(
                code=code,
                code_system=CodeSystem.HCPCS,
                code_description=description,
                confidence=MOCK_CONFIDENCE,
                justification=(
                    f"Code '{code}' ({description}) found literally in policy text "
                    f"(mock LLM extraction)."
                ),
                provenance=Provenance(
                    method=self.method_type,
                    inputs_hash=inputs_hash,
                    parameters={"top_k": top_k},
                    evidence=[
                        EvidenceItem(
                            evidence_type=EvidenceType.TEXT_MATCH,
                            source="policy_text",
                            snippet=span.text,
                            span=span,
                        ),
                    ],
                    artifacts={"note": "Mock method — regex extraction only"},
                ),
            ))

        return inferences
