import dspy

from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningTask


class DecodingStrategy(dspy.Module):
    """Base class for decoding / answer-aggregation strategies.

    A decoding strategy wraps a :class:`ReasoningStrategy` and controls how many
    candidate predictions are sampled and how they are merged into a single final
    prediction. The simplest concrete subclass, :class:`SinglePass`, delegates to the
    wrapped strategy (one sample, no aggregation).

    Concrete subclasses implement :meth:`forward`, which has the same signature as a
    strategy's forward so the evaluator can treat a decoding module transparently.
    """

    name: str = "base"

    def __init__(self, strategy: ReasoningStrategy) -> None:
        super().__init__()
        self.strategy = strategy

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

    def forward(self, **inputs: object) -> dspy.Prediction:
        raise NotImplementedError


class SinglePass(DecodingStrategy):
    """Baseline decoding: a single candidate from the wrapped strategy."""

    name = "single-pass"

    def forward(self, **inputs: object) -> dspy.Prediction:
        return self.strategy(**inputs)
