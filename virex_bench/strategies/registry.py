import dspy

from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.cot import CoTStrategy
from virex_bench.strategies.direct import DirectStrategy

# Implemented strategies. The remaining axes from the project plan
# (self_consistency, tot, mctot, pot_z3) will be added here as they land.
_STRATEGY_REGISTRY: dict[str, type[ReasoningStrategy]] = {
    DirectStrategy.name: DirectStrategy,
    CoTStrategy.name: CoTStrategy,
}


def get_strategy(name: str, signature: type[dspy.Signature]) -> ReasoningStrategy:
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise KeyError(f"Unknown strategy {name!r}. Available strategies: {available}")
    return _STRATEGY_REGISTRY[name](signature)


def list_strategies() -> list[str]:
    return sorted(_STRATEGY_REGISTRY)
