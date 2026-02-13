"""Abstract base class for inference methods (strategy pattern)."""
from abc import ABC, abstractmethod
from typing import Any

from src.policybot_hcpcs.models.schemas import (
    CodeInference,
    HCPCSCode,
    InferenceMethodType,
    PolicyInput,
)


class BaseInferenceMethod(ABC):
    """
    Strategy interface for all inference methods.

    Lifecycle:
        1. initialize(hcpcs_catalog) — called once, stores the catalog
        2. infer(policy, parameters)  — called per-policy

    Subclasses must implement `method_type` and `infer()`.
    """

    @property
    @abstractmethod
    def method_type(self) -> InferenceMethodType:
        """Unique identifier for this method."""
        ...

    @abstractmethod
    def infer(
        self,
        policy: PolicyInput,
        parameters: dict[str, Any] | None = None,
    ) -> list[CodeInference]:
        """
        Run inference on a single policy.

        The HCPCS catalog is available via self (set during initialize()).
        Returns a list of CodeInference objects with full provenance.
        """
        ...

    def initialize(self, hcpcs_catalog: list[HCPCSCode], **kwargs: Any) -> None:
        """
        One-time setup: store the catalog and perform any expensive prep
        (e.g., build TF-IDF matrix, load embeddings).
        """
        pass
