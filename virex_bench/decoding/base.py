from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.registry import parse_strategy_name

if TYPE_CHECKING:
    from virex_bench.tasks.base import ReasoningTask


class DecodingStrategy(dspy.Module):
    """Base class for decoding / answer-aggregation strategies.

    A decoding strategy wraps a :class:`ReasoningStrategy` and controls how many
    candidate predictions are sampled and how they are merged into a single final
    prediction. The simplest concrete subclass, :class:`SinglePass`, delegates to the
    wrapped strategy (one sample, no aggregation).

    Concrete subclasses implement :meth:`forward`, which has the same signature as a
    strategy's forward so the evaluator can treat a decoding module transparently.

    Strategy compatibility
    ----------------------
    A subclass may restrict which reasoning strategies it supports by setting
    :attr:`compatible_strategies` to a set of base strategy names (e.g.
    ``{"direct", "cot"}``); ``None`` (the default) accepts all strategies.
    Compatibility is verified once at construction; :meth:`supports_strategy` exposes
    the same check for callers (CLI/tests) that want to query without building.
    """

    name: str = "base"
    description: str = ""
    # Base strategy names this decoding supports, or None for "all". A strategy
    # instance with a composite name (e.g. ``tot-mcts``) is reduced to its base
    # name (``tot``) before the membership check.
    compatible_strategies: set[str] | None = None

    def __init__(self, strategy: ReasoningStrategy) -> None:
        super().__init__()
        if not self.supports_strategy(strategy):
            base_name, _variant = parse_strategy_name(strategy.name)
            allowed = type(self).compatible_strategies
            assert allowed is not None
            allowed_str = ", ".join(sorted(allowed))
            raise ValueError(
                f"Decoding {type(self).name!r} is not compatible with strategy "
                f"{strategy.name!r} (base {base_name!r}); "
                f"compatible strategies: {allowed_str}."
            )
        self.strategy = strategy

    @classmethod
    def supports_strategy(cls, strategy: ReasoningStrategy) -> bool:
        """Whether this decoding can wrap *strategy* (declarative class-level guard).

        Maps a composite strategy name (e.g. ``tot-mcts``) to its base name before
        checking :attr:`compatible_strategies`. Returns ``True`` when
        ``compatible_strategies is None`` (accept-all default).
        """
        if cls.compatible_strategies is None:
            return True
        base_name, _variant = parse_strategy_name(strategy.name)
        return base_name in cls.compatible_strategies

    def configure_from_task(self, task: ReasoningTask) -> None:
        """Hook for the evaluator to pass task-specific config to the decoding strategy.

        Called once after construction, before any :meth:`forward` call. Subclasses
        override this to pick up task-provided resources such as a custom aggregation
        signature. The base implementation is a no-op.
        """

    @property
    def config(self) -> dict[str, object]:
        """Configuration parameters to record in the evaluation report."""
        return {}

    @property
    def display_name(self) -> str:
        """Human-readable name for logging. May include key parameters (e.g. ``@N``)."""
        return self.name

    def forward(self, **inputs: object) -> dspy.Prediction:
        raise NotImplementedError


class SinglePass(DecodingStrategy):
    """Baseline decoding: a single candidate from the wrapped strategy."""

    name = "single-pass"
    description = "Baseline: a single candidate from the wrapped strategy."

    def forward(self, **inputs: object) -> dspy.Prediction:
        return self.strategy(**inputs)
