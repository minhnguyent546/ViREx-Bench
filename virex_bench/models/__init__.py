from virex_bench.models.backends import load_backend
from virex_bench.models.base import BaseLM, CapturingLMWrapper


def get_model(
    model: str,
    backend: str = "openai",
    api_base: str | None = None,
    api_key: str | None = None,
    cache: bool = False,
    **kwargs,
) -> BaseLM:
    return load_backend(
        backend,
        model,
        api_base=api_base,
        api_key=api_key or "<empty>",
        cache=cache,
        **kwargs,
    )


__all__ = [
    "BaseLM",
    "CapturingLMWrapper",
    "get_model",
    "load_backend",
]
