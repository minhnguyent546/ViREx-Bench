from virex_bench.models.backends import load_backend
from virex_bench.models.base import GenerationConfig, LanguageModel


def get_model(model: str, backend: str = "echo") -> LanguageModel:
    return load_backend(backend, model)


__all__ = [
    "GenerationConfig",
    "LanguageModel",
    "get_model",
    "load_backend",
]
