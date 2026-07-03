"""Tests that ``prediction["search_stats"]`` flows into ``TaskResult.extra``.

Verifies the telemetry plumbing in ``_process_example`` — a strategy that emits
``search_stats`` on its prediction has those stats captured in the result's
``extra`` dict, serialized for the results JSON.
"""

from typing import Any

import dspy

from virex_bench.decoding import SinglePass
from virex_bench.evaluation.evaluate import _process_example  # pyright: ignore[reportPrivateUsage]
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import ReasoningExample


class _FakeSignature(dspy.Signature):
    """Minimal signature for strategy construction."""

    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _FakeStrategy(ReasoningStrategy):
    """Returns a prediction with ``search_stats`` set, mimicking ToTStrategy."""

    name = "fake"

    def forward(self, **inputs: object) -> dspy.Prediction:
        prediction = dspy.Prediction(answer="test_answer")
        prediction["reasoning"] = "test_reasoning"
        prediction["search_stats"] = {
            "algorithm": "beam",
            "nodes_visited": 4,
            "propose_calls": 2,
            "evaluate_calls": 4,
            "depth_reached": 2,
            "best_score": 9.0,
        }
        return prediction


class _FakeTask(ReasoningTask):
    """Bare task — relies on ReasoningTask defaults for the methods _process_example calls."""


def _make_example() -> ReasoningExample:
    return ReasoningExample(
        premises=["All cats are animals.", "Tom is a cat."],
        question="Is Tom an animal?",
        answer="Yes",
        example_id="test-001",
    )


def test_search_stats_flow_into_extra() -> None:
    """A strategy emitting search_stats has them captured in TaskResult.extra."""
    example = _make_example()
    task = _FakeTask()
    strategy = _FakeStrategy(_FakeSignature)
    decoding_strategy = SinglePass(strategy)

    def metric_func(_example: ReasoningExample, _prediction: dspy.Prediction) -> float:
        return 1.0

    result = _process_example(
        example,
        task=task,
        decoding_strategy=decoding_strategy,
        metric_func=metric_func,
    )

    assert "search_stats" in result.extra
    stats: dict[str, Any] = result.extra["search_stats"]  # type: ignore[assignment]
    assert stats["algorithm"] == "beam"
    assert stats["nodes_visited"] == 4
    assert stats["best_score"] == 9.0
    assert result.extra["reasoning"] == "test_reasoning"
    assert result.predicted == "test_answer"


def test_no_search_stats_when_strategy_does_not_emit_them() -> None:
    """Strategies without search_stats (direct, cot) leave extra without that key."""

    class _PlainStrategy(ReasoningStrategy):
        name = "plain"

        def forward(self, **inputs: object) -> dspy.Prediction:
            prediction = dspy.Prediction(answer="yes")
            prediction["reasoning"] = "because"
            return prediction

    example = _make_example()
    task = _FakeTask()
    strategy = _PlainStrategy(_FakeSignature)
    decoding_strategy = SinglePass(strategy)

    def metric_func(_example: ReasoningExample, _prediction: dspy.Prediction) -> float:
        return 1.0

    result = _process_example(
        example,
        task=task,
        decoding_strategy=decoding_strategy,
        metric_func=metric_func,
    )

    assert "search_stats" not in result.extra
    assert result.extra["reasoning"] == "because"
