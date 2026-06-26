from virex_bench.models.backends import load_backend
from virex_bench.models.base import BaseLM


def get_model(
    model: str,
    backend: str = "openai",
    api_base: str | None = None,
    api_key: str | None = None,
) -> BaseLM:
    return load_backend(backend, model, api_base=api_base, api_key=api_key)


__all__ = [
    "BaseLM",
    "get_model",
    "load_backend",
]
