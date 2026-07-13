"""Unit tests for ``SelfCertainty`` decoding."""

# pyright: reportPrivateUsage=false

from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from virex_bench.decoding.base import SinglePass
from virex_bench.decoding.common import (
    copy_lm_with_request_timeout,
    normalize_case_and_whitespaces,
)
from virex_bench.decoding.registry import get_decoding, list_decoding
from virex_bench.decoding.self_certainty import (
    SelfCertainty,
    SelfCertaintyLM,
    _extract_token_logprobs,
    _last_completion_logprobs,
    _mean_logprob,
)
from virex_bench.decoding.self_consistency import SelfConsistency
from virex_bench.models.base import BaseLM
from virex_bench.strategies.base import ReasoningStrategy


class _TestSignature(dspy.Signature):
    premises: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _TokenLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def _logprobs(values: list[float]) -> SimpleNamespace:
    """Build a minimal stand-in for a choice's logprobs blob (`.content[i].logprob`)."""
    blob = SimpleNamespace()
    blob.content = [_TokenLogprob(value) for value in values]
    return blob


class _RecordingBaseLM:
    """Minimal stand-in for a BaseLM: records call kwargs and returns canned outputs."""

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.recorded_kwargs: dict[str, Any] | None = None
        self.model = "test-model"
        self.kwargs: dict[str, Any] = {"temperature": 0.7}

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        self.recorded_kwargs = dict(kwargs)
        return self.outputs


class _FakeStrategy(ReasoningStrategy):
    """Returns a canned prediction per call and seeds the active LM's logprobs.

    Each sample is a ``(prediction, logprobs)`` pair; the logprobs are pushed onto
    the active :class:`SelfCertaintyLM` (read from ``dspy.settings.lm``) so the
    certainty pipeline sees them exactly as a real LM call would leave them.

    ``name`` is a compatible base strategy so the self-certainty compatibility guard
    accepts it (it stands in for a single-call strategy like ``direct``).
    """

    name = "direct"

    def __init__(self, samples: list[tuple[dspy.Prediction, Any]]) -> None:
        super().__init__(_TestSignature)
        self._samples = list(samples)
        self._index = 0
        self.call_count = 0

    def forward(self, **_inputs: object) -> dspy.Prediction:
        if self._index >= len(self._samples):
            raise AssertionError("FakeStrategy exhausted")
        prediction, logprobs = self._samples[self._index]
        self._index += 1
        self.call_count += 1
        active_lm = dspy.settings.lm
        if isinstance(active_lm, SelfCertaintyLM):
            active_lm.last_logprobs = logprobs
        return prediction


class _RaisingStrategy(ReasoningStrategy):
    name = "direct"

    def __init__(self) -> None:
        super().__init__(_TestSignature)
        self.call_count = 0

    def forward(self, **_inputs: object) -> dspy.Prediction:
        self.call_count += 1
        raise RuntimeError("path failed")


class _NamedStub(ReasoningStrategy):
    """Minimal strategy carrying an arbitrary ``name`` for compatibility tests."""

    def __init__(self, strategy_name: str) -> None:
        super().__init__(_TestSignature)
        self.name = strategy_name

    def forward(self, **_inputs: object) -> dspy.Prediction:
        return dspy.Prediction(answer="x")


def _make_lm() -> BaseLM:
    return BaseLM(model="openai/test-model", api_key="test-key", cache=True)


def _pred(answer: str, *, reasoning: str = "r") -> dspy.Prediction:
    return dspy.Prediction(answer=answer, reasoning=reasoning)


_INPUTS: dict[str, str] = {"premises": "some premises", "question": "some question"}


def test_extract_token_logprobs_object_form() -> None:
    assert _extract_token_logprobs(_logprobs([-0.1, -0.2, -0.3])) == [-0.1, -0.2, -0.3]


def test_extract_token_logprobs_dict_form() -> None:
    blob = {"content": [{"logprob": -0.5}, {"logprob": -0.6}]}
    assert _extract_token_logprobs(blob) == [-0.5, -0.6]


def test_extract_token_logprobs_none_and_empty() -> None:
    assert _extract_token_logprobs(None) == []
    assert _extract_token_logprobs(SimpleNamespace(content=None)) == []
    assert _extract_token_logprobs(SimpleNamespace(content=[])) == []


def test_mean_logprob_basic() -> None:
    assert _mean_logprob(_logprobs([-0.1, -0.3])) == pytest.approx(-0.2)


def test_mean_logprob_unavailable_returns_none() -> None:
    assert _mean_logprob(None) is None
    assert _mean_logprob(SimpleNamespace(content=[])) is None


def test_last_completion_logprobs_picks_last_with_logprobs() -> None:
    outputs = [
        {"text": "a"},
        {"text": "b", "logprobs": _logprobs([-0.1])},
        {"text": "c", "logprobs": _logprobs([-0.2, -0.3])},
    ]
    result = _last_completion_logprobs(outputs)
    assert _mean_logprob(result) == pytest.approx(-0.25)


def test_last_completion_logprobs_none_when_absent() -> None:
    assert _last_completion_logprobs([{"text": "a"}, "b"]) is None


def test_num_samples_below_one_raises() -> None:
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        SelfCertainty(_FakeStrategy([]), num_samples=0, max_workers=1)


def test_max_workers_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=0)


def test_borda_power_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="borda_power must be >= 0"):
        SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=-0.1)


def test_config_and_display_name() -> None:
    sc = SelfCertainty(_FakeStrategy([]), num_samples=4, max_workers=2, borda_power=1.5)
    assert sc.config["num_samples"] == 4
    assert sc.config["borda_power"] == 1.5
    assert sc.display_name == "self-certainty@4"


def test_self_certainty_lm_injects_logprobs_and_captures() -> None:
    base = _RecordingBaseLM([{"text": "answer", "logprobs": _logprobs([-0.1, -0.2])}])
    wrapper = SelfCertaintyLM(base)  # pyright: ignore[reportArgumentType]
    result = wrapper(messages=[{"role": "user", "content": "hi"}], temperature=0.6)

    assert result == base.outputs
    assert base.recorded_kwargs is not None
    assert base.recorded_kwargs["logprobs"] is True  # injected into the request
    assert base.recorded_kwargs["temperature"] == 0.6  # other kwargs preserved
    assert _mean_logprob(wrapper.last_logprobs) == pytest.approx(-0.15)  # captured


def test_self_certainty_lm_last_logprobs_none_when_no_logprobs_returned() -> None:
    base = _RecordingBaseLM([{"text": "answer"}])
    wrapper = SelfCertaintyLM(base)  # pyright: ignore[reportArgumentType]
    wrapper(messages=[{"role": "user", "content": "hi"}])
    assert wrapper.last_logprobs is None


def test_self_certainty_lm_delegates_attributes() -> None:
    base = _RecordingBaseLM([])
    wrapper = SelfCertaintyLM(base)  # pyright: ignore[reportArgumentType]
    assert wrapper.model == "test-model"
    assert wrapper.kwargs == {"temperature": 0.7}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  Hello  World ", "hello world"), ("Có", "có"), ("A, B", "a, b"), ("", "")],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_case_and_whitespaces(raw) == expected


def test_copy_lm_with_request_timeout_sets_cache_and_timeout_without_mutating_base() -> None:
    base_lm = _make_lm()
    copied = copy_lm_with_request_timeout(base_lm, 42, cache=False)
    assert copied.cache is False
    assert copied.kwargs["timeout"] == 42
    assert base_lm.cache is True
    assert "timeout" not in base_lm.kwargs


def _borda(
    sc: SelfCertainty,
    answers: list[str],
    certainties: list[float | None],
) -> tuple[str, list[int], float, int]:
    results = [_pred(answer) for answer in answers]
    return sc._borda_vote(results, certainties)  # pyright: ignore[reportPrivateUsage]


def test_borda_power_zero_matches_majority_vote() -> None:
    # "A" has 2 votes vs "B" 1 -> "A" wins under plain majority.
    sc = SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=0.0)
    winning_answer, winner_indices, _confidence, _representative = _borda(
        sc, ["A", "A", "B"], [-0.9, -0.95, -0.1]
    )
    assert winning_answer == "A"
    assert winner_indices == [0, 1]


def test_borda_large_power_picks_highest_certainty_answer() -> None:
    # Same split, but "B" is far more confident -> large power flips it to "B".
    sc = SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=10.0)
    winning_answer, winner_indices, _confidence, _representative = _borda(
        sc, ["A", "A", "B"], [-0.9, -0.95, -0.1]
    )
    assert winning_answer == "B"
    assert winner_indices == [2]


def test_borda_representative_is_most_confident_in_winning_group() -> None:
    sc = SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=0.0)
    _answer, _indices, _confidence, representative = _borda(
        sc, ["A", "A", "B"], [-0.9, -0.5, -0.1]
    )
    # Among the winning "A" group (indices 0,1), index 1 has the higher certainty.
    assert representative == 1


def test_borda_tiebreaks_first_seen_on_equal_weight() -> None:
    # Two equally-confident singletons -> first-seen wins.
    sc = SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=0.5)
    winning_answer, _indices, _confidence, _representative = _borda(sc, ["B", "A"], [-0.2, -0.2])
    assert winning_answer == "B"


def test_borda_all_none_certainties_degrades_to_majority_vote() -> None:
    sc = SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=0.5)
    winning_answer, winner_indices, confidence, representative = _borda(
        sc, ["A", "B", "A", "B", "A"], [None, None, None, None, None]
    )
    assert winning_answer == "A"  # 3 vs 2
    assert winner_indices == [0, 2, 4]
    assert confidence == pytest.approx(3 / 5)
    assert representative == 0  # first-seen when no logprobs


def test_borda_partial_none_certainties_ranked_last() -> None:
    # Two "A" candidates; the None-certainty one ranks below everything.
    sc = SelfCertainty(_FakeStrategy([]), num_samples=1, max_workers=1, borda_power=2.0)
    results = [_pred("A"), _pred("A"), _pred("B")]
    certainties: list[float | None] = [-0.1, None, -0.2]
    winning_answer, winner_indices, _confidence, representative = sc._borda_vote(  # pyright: ignore[reportPrivateUsage]
        results, certainties
    )
    assert winning_answer == "A"
    assert winner_indices == [0, 1]
    # The available "A" candidate (index 0, -0.1) is more confident than the None one.
    assert representative == 0


def test_forward_with_logprobs_picks_borda_winner_and_records_stats() -> None:
    strategy = _FakeStrategy(
        [
            (_pred("A"), _logprobs([-0.1])),
            (_pred("A"), _logprobs([-0.5])),
            (_pred("B"), _logprobs([-0.2])),
        ]
    )
    sc = SelfCertainty(strategy, num_samples=3, max_workers=2, borda_power=0.5)
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)

    assert result.answer == "A"  # majority + borda favor "A"
    stats = result.decoding_stats
    assert stats["total_samples"] == 3
    assert stats["requested_samples"] == 3
    assert stats["vote_count"] == 2  # two "A" candidates
    assert stats["borda_power"] == 0.5
    assert stats["logprobs_available"] is True
    # Representative is the most-confident "A" candidate (certainty -0.1).
    assert stats["winner_certainty"] == pytest.approx(-0.1)
    assert stats["mean_certainty"] == pytest.approx((-0.1 + -0.5 + -0.2) / 3)
    assert 0.0 < stats["confidence"] <= 1.0
    assert strategy.call_count == 3


def test_forward_power_zero_pure_majority_but_still_records_certainty() -> None:
    # Majority says "A" (2) but "B" is the most confident single candidate.
    strategy = _FakeStrategy(
        [
            (_pred("A"), _logprobs([-0.9])),
            (_pred("A"), _logprobs([-0.95])),
            (_pred("B"), _logprobs([-0.1])),
        ]
    )
    sc = SelfCertainty(strategy, num_samples=3, max_workers=2, borda_power=0.0)
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"  # power=0 == majority vote
    assert result.decoding_stats["winner_certainty"] == pytest.approx(-0.9)


def test_forward_no_logprobs_degrades_to_majority_vote() -> None:
    strategy = _FakeStrategy(
        [
            (_pred("A"), None),
            (_pred("B"), None),
            (_pred("A"), None),
        ]
    )
    sc = SelfCertainty(strategy, num_samples=3, max_workers=2, borda_power=0.5)
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)

    assert result.answer == "A"  # 2 vs 1 majority
    stats = result.decoding_stats
    assert stats["logprobs_available"] is False
    assert stats["winner_certainty"] is None
    assert stats["mean_certainty"] is None
    assert stats["confidence"] == pytest.approx(2 / 3)


def test_forward_all_paths_fail_raises_runtime_error() -> None:
    strategy = _RaisingStrategy()
    sc = SelfCertainty(strategy, num_samples=2, max_workers=2)
    with dspy.context(lm=_make_lm()):
        with pytest.raises(RuntimeError, match="paths failed"):
            sc.forward(**_INPUTS)


def test_forward_returns_prediction_with_reasoning_from_representative() -> None:
    # The representative is the most-confident "A" candidate; its reasoning wins.
    strategy = _FakeStrategy(
        [
            (_pred("A", reasoning="weak"), _logprobs([-0.8])),
            (_pred("A", reasoning="strong"), _logprobs([-0.05])),
            (_pred("B", reasoning="other"), _logprobs([-0.1])),
        ]
    )
    sc = SelfCertainty(strategy, num_samples=3, max_workers=2, borda_power=0.5)
    with dspy.context(lm=_make_lm()):
        result = sc.forward(**_INPUTS)
    assert result.answer == "A"
    assert result.reasoning == "strong"  # from the most-confident "A" candidate


def test_registry_lists_and_resolves_self_certainty() -> None:
    assert "self-certainty" in list_decoding()
    strategy = _FakeStrategy([(_pred("A"), None)])
    decoding = get_decoding("self-certainty", strategy, num_samples=1, max_workers=1)
    assert isinstance(decoding, SelfCertainty)
    assert decoding.num_samples == 1


def test_registry_rejects_unknown_decoding() -> None:
    with pytest.raises(KeyError, match="Unknown decoding"):
        get_decoding("nope", _FakeStrategy([]))  # pyright: ignore[reportArgumentType]
