"""OpenAI LLM inference method — uses GPT for HCPCS code inference."""
import hashlib
import json
import logging
import os
import time
from typing import Any

from src.policybot_hcpcs.config import (
    LLM_MAX_POLICY_CHARS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.policybot_hcpcs.inference.base import BaseInferenceMethod
from src.policybot_hcpcs.inference.registry import register_method
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

PROMPT_TEMPLATE = """\
You are a medical coding specialist. Given the following healthcare policy text \
and HCPCS code catalog, identify all HCPCS/CPT codes that are relevant to this policy.

For each code, provide:
- code: the HCPCS/CPT code
- confidence: a float 0-1 indicating how confident you are this code is relevant
- justification: a brief explanation of why this code is relevant to the policy

Policy text:
{policy_text}

HCPCS Catalog (code | description):
{hcpcs_table}

Respond with JSON only: {{"codes": [{{"code": "...", "confidence": 0.X, "justification": "..."}}]}}
"""


@register_method(InferenceMethodType.LLM_OPENAI)
class LlmOpenaiMethod(BaseInferenceMethod):
    """
    Uses OpenAI's chat completion API to infer HCPCS codes from policy text.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self) -> None:
        self._client = None
        self._catalog_hash: str = ""
        self._code_desc_map: dict[str, str] = {}
        self._hcpcs_table: str = ""
        self._prompt_hash: str = ""

    @property
    def method_type(self) -> InferenceMethodType:
        return InferenceMethodType.LLM_OPENAI

    def initialize(self, hcpcs_catalog: list[HCPCSCode], **kwargs: Any) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Use --method lexical_tfidf or --method llm_mock instead."
            )

        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._catalog_hash = compute_catalog_hash(hcpcs_catalog)
        self._code_desc_map = {c.code: c.description for c in hcpcs_catalog}

        # TODO: For scalability, pre-filter the catalog per-policy (e.g., TF-IDF
        # top-50) before sending to the LLM. Currently sends all 1320 codes in
        # every prompt (~40k tokens per call).
        lines = [f"{c.code} | {c.description}" for c in hcpcs_catalog]
        self._hcpcs_table = "\n".join(lines)

        self._prompt_hash = hashlib.sha256(
            PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest()[:16]

        logger.info(f"OpenAI method initialized (model={LLM_MODEL})")

    def infer(
        self,
        policy: PolicyInput,
        parameters: dict[str, Any] | None = None,
    ) -> list[CodeInference]:
        import openai

        params = parameters or {}
        top_k = params.get("top_k", 10)
        inputs_hash = compute_inputs_hash(policy.policy_text, self._catalog_hash)

        # Truncate policy text to stay within token limits
        policy_text = policy.policy_text[:LLM_MAX_POLICY_CHARS]
        prompt = PROMPT_TEMPLATE.format(
            policy_text=policy_text,
            hcpcs_table=self._hcpcs_table,
        )

        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
            raw_codes = parsed.get("codes", [])[:top_k]

            # Rate limiting
            time.sleep(0.5)

        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI API error for policy {policy.policy_id}: {e}")
            return []
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse LLM response for policy {policy.policy_id}: {e}")
            return []

        inferences = []
        for item in raw_codes:
            code = str(item.get("code", "")).strip()
            confidence = float(item.get("confidence", 0.5))
            justification = str(item.get("justification", "Inferred by LLM"))

            confidence = max(0.0, min(1.0, confidence))

            inferences.append(CodeInference(
                code=code,
                code_system=CodeSystem.HCPCS,
                code_description=self._code_desc_map.get(code),
                confidence=round(confidence, 4),
                justification=justification,
                provenance=Provenance(
                    method=self.method_type,
                    inputs_hash=inputs_hash,
                    parameters={"top_k": top_k, "model": LLM_MODEL, "temperature": LLM_TEMPERATURE},
                    evidence=[
                        EvidenceItem(
                            evidence_type=EvidenceType.LLM_COMPLETION,
                            source=f"openai_{LLM_MODEL}",
                            snippet=justification,
                        ),
                    ],
                    artifacts={
                        "model": LLM_MODEL,
                        "prompt_template_hash": self._prompt_hash,
                        "response_id": getattr(response, "id", None),
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        } if hasattr(response, "usage") and response.usage else None,
                    },
                ),
            ))

        return inferences
