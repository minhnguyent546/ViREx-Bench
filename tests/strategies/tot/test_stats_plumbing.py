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


def test_pot_info_flows_into_extra() -> None:
    """A strategy emitting pot_info has it captured in TaskResult.extra.

    PoT-Z3 is a single-pass generate-execute-regenerate pipeline, not a tree
    search, so it emits ``pot_info`` (and the report aggregates it into
    ``pot_stats``) and does NOT emit ``search_stats``.
    """

    class _PoTFakeStrategy(ReasoningStrategy):
        name = "pot-fake"

        def forward(self, **inputs: object) -> dspy.Prediction:
            prediction = dspy.Prediction(answer="Có")
            prediction["reasoning"] = "solver ran"
            prediction["pot_info"] = {
                "generated_code": "result = find_solution([])",
                "execution_output": "{'answer': 'Có'}",
                "execution_error": None,
                "execution_success": True,
                "generate_calls": 1,
                "regenerate_calls": 0,
                "execute_calls": 1,
                "total_llm_calls": 2,
            }
            return prediction

    example = _make_example()
    task = _FakeTask()
    strategy = _PoTFakeStrategy(_FakeSignature)
    decoding_strategy = SinglePass(strategy)

    def metric_func(_example: ReasoningExample, _prediction: dspy.Prediction) -> float:
        return 1.0

    result = _process_example(
        example,
        task=task,
        decoding_strategy=decoding_strategy,
        metric_func=metric_func,
    )

    assert "pot_info" in result.extra
    pot_info: dict[str, Any] = result.extra["pot_info"]  # type: ignore[assignment]
    assert pot_info["generated_code"] == "result = find_solution([])"
    assert pot_info["execution_output"] == "{'answer': 'Có'}"
    assert pot_info["execution_error"] is None
    assert pot_info["execution_success"] is True
    assert pot_info["generate_calls"] == 1
    assert pot_info["regenerate_calls"] == 0
    assert pot_info["execute_calls"] == 1
    assert pot_info["total_llm_calls"] == 2
    # PoT does not emit search_stats -- it is a pipeline, not a tree search.
    assert "search_stats" not in result.extra


def test_no_pot_info_when_strategy_does_not_emit_it() -> None:
    """Strategies without pot_info (direct, cot, tot) leave extra without that key."""

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

    assert "pot_info" not in result.extra


def test_decoding_stats_flow_into_extra() -> None:
    """A decoding strategy emitting ``decoding_stats`` has it captured in
    ``TaskResult.extra``.

    Mirrors the search_stats/pot_info plumbing: ``SelfConsistency`` sets
    ``decoding_stats`` on the prediction (confidence, vote counts,
    aggregator_applied); the evaluator routes it into ``extra`` and aggregates
    it into the report's ``decoding_stats`` header field.
    """

    class _SCFakeStrategy(ReasoningStrategy):
        name = "sc-fake"

        def forward(self, **inputs: object) -> dspy.Prediction:
            prediction = dspy.Prediction(answer="Có")
            prediction["reasoning"] = "majority vote"
            prediction["decoding_stats"] = {
                "confidence": 0.6,
                "vote_count": 3,
                "total_samples": 5,
                "requested_samples": 5,
                "aggregator_applied": False,
            }
            return prediction

    example = _make_example()
    task = _FakeTask()
    strategy = _SCFakeStrategy(_FakeSignature)
    decoding_strategy = SinglePass(strategy)

    def metric_func(_example: ReasoningExample, _prediction: dspy.Prediction) -> float:
        return 1.0

    result = _process_example(
        example,
        task=task,
        decoding_strategy=decoding_strategy,
        metric_func=metric_func,
    )

    assert "decoding_stats" in result.extra
    stats: dict[str, Any] = result.extra["decoding_stats"]  # type: ignore[assignment]
    assert stats["confidence"] == 0.6
    assert stats["vote_count"] == 3
    assert stats["total_samples"] == 5
    assert stats["aggregator_applied"] is False
    assert result.extra["reasoning"] == "majority vote"
    assert result.predicted == "Có"


def test_no_decoding_stats_when_strategy_does_not_emit_them() -> None:
    """Strategies/decodings without ``decoding_stats`` leave extra without it."""

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

    assert "decoding_stats" not in result.extra
