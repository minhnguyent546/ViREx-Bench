from typing import Any

from virex_bench.decoding.base import DecodingStrategy, SinglePass
from virex_bench.decoding.self_certainty import SelfCertainty
from virex_bench.decoding.self_consistency import SelfConsistency
from virex_bench.strategies.base import ReasoningStrategy

_DECODING_REGISTRY: dict[str, type[DecodingStrategy]] = {
    SinglePass.name: SinglePass,
    SelfConsistency.name: SelfConsistency,
    SelfCertainty.name: SelfCertainty,
}


def get_decoding(
    name: str,
    strategy: ReasoningStrategy,
    **kwargs: Any,
) -> DecodingStrategy:
    """Resolve a decoding strategy by name and construct it with *strategy* inside."""
    if name not in _DECODING_REGISTRY:
        available = ", ".join(list_decoding())
        raise KeyError(f"Unknown decoding {name!r}. Available: {available}")
    return _DECODING_REGISTRY[name](strategy, **kwargs)  # pyright: ignore[reportCallIssue]


def list_decoding() -> list[str]:
    return sorted(_DECODING_REGISTRY)


def decoding_descriptions() -> dict[str, str]:
    """Map each decoding name to its short description."""
    return {name: cls.description for name, cls in _DECODING_REGISTRY.items()}
