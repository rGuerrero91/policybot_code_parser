"""Method registry and factory for inference strategies."""
import importlib
import pkgutil
from typing import Type

from src.policybot_hcpcs.inference.base import BaseInferenceMethod
from src.policybot_hcpcs.models.schemas import InferenceMethodType

_REGISTRY: dict[InferenceMethodType, Type[BaseInferenceMethod]] = {}

_SKIP_MODULES = {"base", "registry", "utils"}


def register_method(method_type: InferenceMethodType):
    """Decorator to register an inference method class."""
    def wrapper(cls: Type[BaseInferenceMethod]):
        _REGISTRY[method_type] = cls
        return cls
    return wrapper


def get_method(method_type: InferenceMethodType, **kwargs) -> BaseInferenceMethod:
    """Instantiate a registered method by its type."""
    if method_type not in _REGISTRY:
        available = ", ".join(m.value for m in _REGISTRY)
        raise ValueError(
            f"Unknown method '{method_type.value}'. Available: {available}"
        )
    return _REGISTRY[method_type](**kwargs)


def discover_methods() -> None:
    """Auto-import all method modules in this package so @register_method decorators run."""
    import src.policybot_hcpcs.inference as pkg

    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        if module_name not in _SKIP_MODULES:
            importlib.import_module(f"src.policybot_hcpcs.inference.{module_name}")
