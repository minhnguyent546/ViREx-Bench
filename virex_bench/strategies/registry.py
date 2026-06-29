import dspy
from pydantic.fields import FieldInfo

from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.cot import CoTStrategy
from virex_bench.strategies.direct import DirectStrategy
from virex_bench.strategies.tot import ToTStrategy

# Implemented strategies. The remaining axes from the project plan
# (self_consistency, mctot, pot_z3) will be added here as they land.
_STRATEGY_REGISTRY: dict[str, type[ReasoningStrategy]] = {
    DirectStrategy.name: DirectStrategy,
    CoTStrategy.name: CoTStrategy,
    ToTStrategy.name: ToTStrategy,
}

# Strategies that expose a CLI variant axis: `<base>-<variant>` (hyphen separator).
# Used for both `get_strategy` validation and `list_strategies` discovery.
# Phased: "beam" (Phase 1), "dfs" (Phase 2), "mcts" (Phase 3).
_STRATEGY_VARIANTS: dict[str, list[str]] = {
    "tot": ["beam"],
}

_VARIANT_SEPARATOR = "-"


def parse_strategy_name(name: str) -> tuple[str, str | None]:
    """Split a composite strategy name into ``(base, variant)``.

    ``"tot-mcts"`` -> ``("tot", "mcts")``; ``"tot"`` -> ``("tot", None)``.
    """
    base, separator, variant = name.partition(_VARIANT_SEPARATOR)
    return (base, variant) if separator else (name, None)


def get_strategy(
    name: str,
    signature: type[dspy.Signature],
    rationale_field: FieldInfo | None = None,
    rationale_field_type: type = str,
) -> ReasoningStrategy:
    base_name, variant = parse_strategy_name(name)
    if base_name not in _STRATEGY_REGISTRY:
        available = ", ".join(list_strategies())
        raise KeyError(f"Unknown strategy {name!r}. Available strategies: {available}")
    strategy_cls = _STRATEGY_REGISTRY[base_name]
    if variant is None:
        return strategy_cls(signature, rationale_field, rationale_field_type)
    if not getattr(strategy_cls, "accepts_variant", False):
        raise ValueError(f"Strategy {base_name!r} has no variant axis; use {base_name!r}")
    allowed = _STRATEGY_VARIANTS.get(base_name, [])
    if variant not in allowed:
        available = ", ".join(f"{base_name}-{entry}" for entry in allowed) or "none"
        raise ValueError(
            f"Unknown variant {variant!r} for strategy {base_name!r}. Available: {available}"
        )
    # Only ToTStrategy accepts `variant`, validated above via `accepts_variant`.
    return strategy_cls(signature, rationale_field, rationale_field_type, variant=variant)  # pyright: ignore[reportCallIssue]


def list_strategies() -> list[str]:
    names: list[str] = []
    for base in _STRATEGY_REGISTRY:
        names.append(base)
        for variant in _STRATEGY_VARIANTS.get(base, []):
            names.append(f"{base}{_VARIANT_SEPARATOR}{variant}")
    return sorted(names)
