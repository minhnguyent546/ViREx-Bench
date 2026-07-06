"""Unit tests for ``SelfConsistency`` decoding with a stubbed strategy and aggregator.

Exercises validation, task configuration, the deterministic majority-vote helpers,
the LLM-aggregator override extraction, and the ``forward`` merge/fallback paths
(no real LM calls): aggregator-disabled vote winner, total-path-failure, closed and
open-ended aggregator acceptance, and the three fallback cases (empty answer,
closed-answer no-match, open-answer below the fuzzy threshold).
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest

from virex_bench.decoding.self_consistency import SelfConsistency
from virex_bench.models.base import BaseLM
from virex_bench.strategies.base import ReasoningStrategy


class _TestSignature(dspy.Signature):
    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _AggSignature(dspy.Signature):
    """Aggregation signature carrying the candidate_answers input field."""

    candidate_answers: str = dspy.InputField()
    answer: str = dspy.OutputField()
    answer_type: str = dspy.OutputField()
    explanation: str = dspy.OutputField()


class _FakeTask:
    name = "fake-task"
    aggregation_signature: type[dspy.Signature] | None = _AggSignature


class _FakeTaskNoAgg:
    name = "fake-task-no-agg"
    aggregation_signature: type[dspy.Signature] | None = None


class _FakeStrategy(ReasoningStrategy):
    """Returns canned ``dspy.Prediction``s in sequence, one per ``forward`` call."""

    name = "fake"

    def __init__(self, predictions: list[dspy.Prediction]) -> None:
        super().__init__(_TestSignature)
        self._predictions = list(predictions)
        self._index = 0
        self.call_count = 0

    def forward(self, **_inputs: object) -> dspy.Prediction:
        if self._index >= len(self._predictions):
            raise AssertionError("FakeStrategy exhausted")
        prediction = self._predictions[self._index]
        self._index += 1
        self.call_count += 1
        return prediction


class _RaisingStrategy(ReasoningStrategy):
    name = "raising"

    def __init__(self) -> None:
        super().__init__(_TestSignature)
        self.call_count = 0

    def forward(self, **_inputs: object) -> dspy.Prediction:
        self.call_count += 1
        raise RuntimeError("path failed")


class _FakeAggregator:
    """Returns a canned prediction and records the kwargs it was called with."""

    def __init__(self, prediction: dspy.Prediction) -> None:
        self._prediction = prediction
        self.call_count = 0
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: object) -> dspy.Prediction:
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
        return self._prediction


class _RaisingAggregator:
    """Always raises, so ``_aggregate`` catches and returns ``None``."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, **_kwargs: object) -> dspy.Prediction:
        self.call_count += 1
        raise RuntimeError("aggregator crashed")


def _make_lm() -> BaseLM:
    return BaseLM(model="openai/test-model", api_key="test-key", cache=True)


def _pred(answer: str, *, reasoning: str = "r", answer_type: str = "mcq") -> dspy.Prediction:
    return dspy.Prediction(answer=answer, reasoning=reasoning, answer_type=answer_type)


_INPUTS: dict[str, str] = {"premises": "some premises", "question": "some question"}


def test_num_samples_below_one_raises() -> None:
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        SelfConsistency(_FakeStrategy([]), num_samples=0, max_workers=1)


def test_max_workers_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=0)


def test_configure_from_task_without_aggregation_signature_raises() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    with pytest.raises(ValueError, match="aggregation_signature"):
        sc.configure_from_task(_FakeTaskNoAgg())  # pyright: ignore[reportArgumentType]


def test_configure_from_task_wires_aggregator_and_signature() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]  # pyright: ignore[reportArgumentType]
    assert sc._aggregation_signature is _AggSignature  # pyright: ignore[reportPrivateUsage]
    assert sc._aggregator is not None  # pyright: ignore[reportPrivateUsage]


def test_config_and_display_name() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=3, max_workers=2, use_aggregator=False)
    assert sc.config["num_samples"] == 3
    assert sc.config["use_aggregator"] is False
    assert sc.display_name == "self-consistency@3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello  World ", "hello world"),
        ("Có", "có"),
        ("A, B", "a, b"),
        ("", ""),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert SelfConsistency._normalize_answer(raw) == expected  # pyright: ignore[reportPrivateUsage]


def test_majority_vote_picks_largest_group() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    results = [_pred("A"), _pred("B"), _pred("A"), _pred("A")]
    winning_answer, winner_indices = sc._majority_vote(results)  # pyright: ignore[reportPrivateUsage]
    assert winning_answer == "A"
    assert winner_indices == [0, 2, 3]


def test_majority_vote_tiebreaks_first_seen() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    results = [_pred("B"), _pred("A")]  # 1-1 tie -> first-seen "B" wins
    winning_answer, winner_indices = sc._majority_vote(results)  # pyright: ignore[reportPrivateUsage]
    assert winning_answer == "B"
    assert winner_indices == [0]


def test_find_matching_result_returns_first_match() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    results = [_pred("A"), _pred("B")]
    assert sc._find_matching_result(results, "b") is results[1]  # pyright: ignore[reportPrivateUsage]


def test_find_matching_result_returns_none_when_no_match() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    results = [_pred("A"), _pred("B")]
    assert sc._find_matching_result(results, "C") is None  # pyright: ignore[reportPrivateUsage]


def test_best_answer_similarity_returns_highest_ratio() -> None:
    sc = SelfConsistency(_FakeStrategy([]), num_samples=1, max_workers=1)
    results = [_pred("the cat sat"), _pred("a dog ran")]
    # Exact normalized match against the first candidate -> ratio 1.0.
    assert sc._best_answer_similarity(results, "the cat sat") == 1.0  # pyright: ignore[reportPrivateUsage]
    # No overlap with any candidate -> ratio near 0.
    assert sc._best_answer_similarity(results, "zzzzzzz") == 0.0  # pyright: ignore[reportPrivateUsage]


def test_extract_overrides_drops_reasoning_maps_explanation_skips_none() -> None:
    agg = dspy.Prediction(
        answer="A",
        reasoning="internal cot",
        explanation="because A",
        answer_type="mcq",
    )
    overrides = SelfConsistency._extract_overrides(agg)  # pyright: ignore[reportPrivateUsage]
    assert overrides == {"answer": "A", "reasoning": "because A", "answer_type": "mcq"}


def test_copy_lm_with_request_timeout_sets_cache_and_timeout_without_mutating_base() -> None:
    base_lm = _make_lm()
    copied = SelfConsistency._copy_lm_with_request_timeout(base_lm, 42, cache=False)  # pyright: ignore[reportPrivateUsage]
    assert copied.cache is False
    assert copied.kwargs["timeout"] == 42
    # The base LM is left untouched.
    assert base_lm.cache is True
    assert "timeout" not in base_lm.kwargs


def test_forward_aggregator_disabled_returns_vote_winner_with_confidence() -> None:
    strategy = _FakeStrategy([_pred("A"), _pred("A"), _pred("B")])
    sc = SelfConsistency(strategy, num_samples=3, max_workers=2, use_aggregator=False)
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"
    assert result.decoding_stats["confidence"] == pytest.approx(2 / 3)
    assert result.decoding_stats["vote_count"] == 2
    assert result.decoding_stats["total_samples"] == 3
    assert result.decoding_stats["requested_samples"] == 3
    assert result.decoding_stats["aggregator_applied"] is False
    assert strategy.call_count == 3


def test_forward_all_paths_fail_raises_runtime_error() -> None:
    strategy = _RaisingStrategy()
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=False)
    with dspy.context(lm=_make_lm()):
        with pytest.raises(RuntimeError, match="paths failed"):
            sc.forward(**_INPUTS)


def test_forward_aggregator_closed_match_merges_overrides() -> None:
    strategy = _FakeStrategy([_pred("A"), _pred("A")])
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=True)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]  # pyright: ignore[reportArgumentType]
    aggregator = _FakeAggregator(
        dspy.Prediction(answer="A", answer_type="mcq", explanation="because A")
    )
    sc._aggregator = aggregator  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"
    # explanation is mapped onto reasoning by the override extractor.
    assert result.reasoning == "because A"
    assert aggregator.call_count == 1
    # The aggregator receives the candidate answers as a JSON string.
    assert "candidate_answers" in (aggregator.last_kwargs or {})
    assert result.decoding_stats["aggregator_applied"] is True


def test_forward_aggregator_empty_answer_falls_back_to_vote_winner() -> None:
    strategy = _FakeStrategy([_pred("A"), _pred("A")])
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=True)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]
    sc._aggregator = _FakeAggregator(  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        dspy.Prediction(answer="", answer_type="mcq", explanation="")
    )
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"
    assert result.decoding_stats["aggregator_applied"] is False


def test_forward_aggregator_closed_no_match_falls_back_to_vote_winner() -> None:
    strategy = _FakeStrategy([_pred("A"), _pred("A")])
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=True)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]
    sc._aggregator = _FakeAggregator(  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        dspy.Prediction(answer="B", answer_type="mcq", explanation="why not")
    )
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"
    assert result.decoding_stats["aggregator_applied"] is False


def test_forward_aggregator_open_answer_above_threshold_merges() -> None:
    candidates = [
        _pred("the cat sat on the mat", answer_type="open_ended"),
        _pred("the cat sat on the mat", answer_type="open_ended"),
    ]
    strategy = _FakeStrategy(candidates)
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=True)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]
    sc._aggregator = _FakeAggregator(  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        dspy.Prediction(
            answer="the cat sat on the mat",
            answer_type="open_ended",
            explanation="paraphrase",
        )
    )
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "the cat sat on the mat"
    assert result.decoding_stats["aggregator_applied"] is True


def test_forward_aggregator_open_answer_below_threshold_falls_back() -> None:
    candidates = [
        _pred("the cat sat on the mat", answer_type="open_ended"),
        _pred("the cat sat on the mat", answer_type="open_ended"),
    ]
    strategy = _FakeStrategy(candidates)
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=True)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]
    sc._aggregator = _FakeAggregator(  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        dspy.Prediction(
            answer="a completely unrelated answer about quantum physics",
            answer_type="open_ended",
            explanation="off topic",
        )
    )
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    # The divergent aggregator answer is rejected; the vote winner is used.
    assert result.answer == "the cat sat on the mat"
    assert result.decoding_stats["aggregator_applied"] is False


def test_forward_aggregator_call_fails_falls_back_to_vote_winner() -> None:
    """When the aggregator itself raises (``_aggregate`` returns None), forward
    falls back to the deterministic vote winner rather than propagating."""
    strategy = _FakeStrategy([_pred("A"), _pred("A")])
    sc = SelfConsistency(strategy, num_samples=2, max_workers=2, use_aggregator=True)
    sc.configure_from_task(_FakeTask())  # pyright: ignore[reportArgumentType]
    raising_agg = _RaisingAggregator()
    sc._aggregator = raising_agg  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"
    assert result.decoding_stats["vote_count"] == 2
    assert result.decoding_stats["total_samples"] == 2
    assert result.decoding_stats["aggregator_applied"] is False
    assert raising_agg.call_count == 1
