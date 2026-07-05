"""Unit tests for BeamSearch with fake proposer/evaluator modules.

Drives ``BeamSearch.search`` with canned thoughts/scores to verify path
selection, pruning, early-stop, stall, and counter correctness — without any
real LM calls.
"""

from typing import Any

import dspy
import pytest

from virex_bench.strategies.tot.search import SearchConfig
from virex_bench.strategies.tot.search.beam import BeamSearch


class _FakeCompletions:
    """Stand-in for ``dspy.Prediction.completions``."""

    def __init__(self, **fields: list[Any]) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


class _FakePrediction:
    """Stand-in for ``dspy.Prediction`` with ``.completions.<field>`` access."""

    def __init__(self, **completion_fields: list[Any]) -> None:
        self.completions = _FakeCompletions(**completion_fields)


class _FakeProposer:
    """Returns canned thought-lists in sequence, one per ``__call__``."""

    def __init__(self, thoughts_per_call: list[list[str]]) -> None:
        self._thoughts = list(thoughts_per_call)
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        thoughts = self._thoughts[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(next_thought=thoughts)


class _FakeEvaluator:
    """Returns canned score-lists in sequence, one per ``__call__``."""

    def __init__(self, scores_per_call: list[list[str]]) -> None:
        self._scores = list(scores_per_call)
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        scores = self._scores[self._index]
        self._index += 1
        self.call_count += 1
        return _FakePrediction(score=scores)


class _ConfigRecordingModule:
    """Records the ``config`` kwarg of each call; returns a fixed canned prediction.

    Used to assert the search loop forwards the pre-computed sampling config
    (temperature-omitted when inherited, temperature-pinned when set) verbatim.
    Returning constant data keeps it robust to however many calls the search makes.
    """

    def __init__(self, prediction: _FakePrediction) -> None:
        self.recorded: list[dict[str, Any]] = []
        self._prediction = prediction

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.recorded.append(dict(kwargs.get("config") or {}))
        return self._prediction


class _OverflowProposer:
    """Acts like ``_FakeProposer`` but raises on its Nth call (1-indexed)."""

    def __init__(self, thoughts_before_overflow: list[list[str]], overflow_on_call: int) -> None:
        self._thoughts = list(thoughts_before_overflow)
        self._overflow_on_call = overflow_on_call
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        if self.call_count == self._overflow_on_call:
            raise dspy.ContextWindowExceededError(message="simulated overflow")
        thoughts = self._thoughts[self._index]
        self._index += 1
        return _FakePrediction(next_thought=thoughts)


class _OverflowEvaluator:
    """Acts like ``_FakeEvaluator`` but raises on its Nth call (1-indexed)."""

    def __init__(self, scores_before_overflow: list[list[str]], overflow_on_call: int) -> None:
        self._scores = list(scores_before_overflow)
        self._overflow_on_call = overflow_on_call
        self._index = 0
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        if self.call_count == self._overflow_on_call:
            raise dspy.ContextWindowExceededError(message="simulated overflow")
        scores = self._scores[self._index]
        self._index += 1
        return _FakePrediction(score=scores)


def _make_config(**overrides: Any) -> SearchConfig:
    defaults: dict[str, Any] = {
        "max_depth": 2,
        "branching_factor": 2,
        "n_eval_samples": 1,
        "propose_temperature": 0.7,
        "evaluate_temperature": 0.0,
        "early_stop_threshold": None,
        "beam_width": 1,
    }
    defaults.update(overrides)
    return SearchConfig(**defaults)


def test_beam_selects_highest_scoring_path() -> None:
    """The overall best-scoring path across all depths is returned."""
    config = _make_config(max_depth=2, beam_width=1, branching_factor=2)
    proposer = _FakeProposer(
        [
            ["A", "B"],  # depth 0: root proposes A, B
            ["A1", "A2"],  # depth 1: A (beam survivor) proposes A1, A2
        ]
    )
    evaluator = _FakeEvaluator(
        [
            ["8"],  # A -> 8
            ["5"],  # B -> 5
            ["9"],  # A1 -> 9
            ["7"],  # A2 -> 7
        ]
    )
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A", "A1"]
    assert result.best_score == 9.0
    assert result.depth_reached == 2
    assert result.nodes_visited == 4
    assert result.propose_calls == 2
    assert result.evaluate_calls == 4


def test_beam_prunes_lower_scoring_candidates() -> None:
    """All candidates are evaluated, but only beam_width survive to the next layer."""
    config = _make_config(max_depth=1, beam_width=1, branching_factor=3)
    proposer = _FakeProposer([["A", "B", "C"]])
    evaluator = _FakeEvaluator(
        [
            ["8"],  # A
            ["5"],  # B
            ["3"],  # C
        ]
    )
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    # All three are evaluated (nodes_visited=3) even though only beam_width=1 survives.
    assert result.best_path == ["A"]
    assert result.best_score == 8.0
    assert result.nodes_visited == 3
    assert result.propose_calls == 1
    assert result.evaluate_calls == 3
    assert result.depth_reached == 1


def test_beam_early_stop_when_threshold_met() -> None:
    """Search stops as soon as best_leaf.score >= early_stop_threshold."""
    config = _make_config(max_depth=3, beam_width=1, branching_factor=1, early_stop_threshold=8.0)
    proposer = _FakeProposer(
        [
            ["A"],  # depth 0
            ["A1"],  # depth 1 (should not be reached)
            ["A2"],  # depth 2 (should not be reached)
        ]
    )
    evaluator = _FakeEvaluator(
        [
            ["8"],  # A -> 8, triggers early stop
            ["10"],
            ["10"],
        ]
    )
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A"]
    assert result.best_score == 8.0
    assert result.depth_reached == 1
    assert result.nodes_visited == 1
    assert proposer.call_count == 1  # did not advance to depth 1
    assert evaluator.call_count == 1


def test_beam_stalls_on_empty_proposals() -> None:
    """When the proposer returns no thoughts, the search stalls immediately."""
    config = _make_config(max_depth=3, beam_width=2, branching_factor=3)
    proposer = _FakeProposer(
        [
            [],  # root produces nothing
            ["A", "B"],
        ]
    )
    evaluator = _FakeEvaluator([["8"], ["5"]])
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    # No nodes expanded; best_leaf stays as the root placeholder.
    assert result.best_path == []
    assert result.best_score == 0.0
    assert result.depth_reached == 0
    assert result.nodes_visited == 0
    assert result.propose_calls == 1
    assert result.evaluate_calls == 0


def test_beam_dedupes_duplicate_thoughts() -> None:
    """Near-duplicate proposals are dropped before evaluation."""
    config = _make_config(max_depth=1, beam_width=2, branching_factor=4)
    proposer = _FakeProposer([["same thought", "Same Thought", "SAME THOUGHT", "unique"]])
    evaluator = _FakeEvaluator(
        [
            ["9"],  # "same thought" (kept)
            ["5"],  # "unique" (kept)
        ]
    )
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    # Only 2 unique thoughts survive dedup -> only 2 evaluated.
    assert result.nodes_visited == 2
    assert result.evaluate_calls == 2
    assert result.best_score == 9.0


def test_beam_width_validation() -> None:
    """BeamSearch rejects a config without beam_width or with beam_width < 1."""
    config_no_width = _make_config(beam_width=None)
    with pytest.raises(ValueError, match="beam_width"):
        BeamSearch(config_no_width)

    config_zero = _make_config(beam_width=0)
    with pytest.raises(ValueError, match="beam_width"):
        BeamSearch(config_zero)


def test_beam_returns_best_path_on_proposer_overflow() -> None:
    """A context-window overflow on a later proposer call returns the best path
    found so far instead of crashing. The overflowing call is not counted."""
    config = _make_config(max_depth=3, beam_width=1, branching_factor=2)
    proposer = _OverflowProposer(
        thoughts_before_overflow=[["A", "B"]],  # depth 0 succeeds
        overflow_on_call=2,  # depth 1 proposer call overflows (path too long)
    )
    evaluator = _FakeEvaluator([["8"], ["5"]])  # A -> 8, B -> 5
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A"]
    assert result.best_score == 8.0
    assert result.depth_reached == 1
    assert result.propose_calls == 1  # the overflowing call did not complete
    assert result.evaluate_calls == 2


def test_beam_returns_best_path_on_evaluator_overflow() -> None:
    """A context-window overflow mid-evaluation still returns the best leaf
    evaluated before the overflow (a partial depth's worth of candidates)."""
    config = _make_config(max_depth=2, beam_width=1, branching_factor=2)
    proposer = _FakeProposer([["A", "B"]])
    evaluator = _OverflowEvaluator(
        scores_before_overflow=[["8"]],  # A -> 8 (recorded as best)
        overflow_on_call=2,  # B's evaluation overflows
    )
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A"]
    assert result.best_score == 8.0
    assert result.depth_reached == 1
    assert result.evaluate_calls == 1  # B's evaluation did not complete


def test_beam_overflow_at_root_returns_empty_path() -> None:
    """If even the first proposer call overflows, the search returns an empty
    path (root placeholder) rather than raising -- the aggregator then answers
    from premises + question alone."""
    config = _make_config(max_depth=3, beam_width=1, branching_factor=2)
    proposer = _OverflowProposer(thoughts_before_overflow=[], overflow_on_call=1)
    evaluator = _FakeEvaluator([])
    search = BeamSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == []
    assert result.best_score == 0.0
    assert result.depth_reached == 0
    assert result.propose_calls == 0
    assert result.evaluate_calls == 0


def test_beam_forwards_inherited_sampling_config() -> None:
    """The search loop forwards the pre-computed sampling config verbatim.

    When temperatures are None the proposer/evaluator receive a dict with `n`
    only -- no `temperature` key -- so dspy inherits the LM's --model-kwargs
    profile (CR's inheritance policy). This guards against a regression to the
    old inline dict, which unconditionally wrote ``"temperature": None``.
    """
    config = _make_config(
        max_depth=1,
        beam_width=1,
        branching_factor=2,
        propose_temperature=None,
        evaluate_temperature=None,
    )
    proposer = _ConfigRecordingModule(_FakePrediction(next_thought=["A", "B"]))
    evaluator = _ConfigRecordingModule(_FakePrediction(score=[5]))
    search = BeamSearch(config)
    search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    recorded = [*proposer.recorded, *evaluator.recorded]
    assert recorded, "expected at least one propose/evaluate call"
    for call_config in recorded:
        assert "n" in call_config
        assert "temperature" not in call_config


def test_beam_forwards_pinned_sampling_config() -> None:
    """When temperatures are pinned, the search forwards them verbatim and keeps
    propose/evaluate on their respective roles (distinguished by `n`)."""
    config = _make_config(
        max_depth=1,
        beam_width=1,
        branching_factor=2,
        propose_temperature=0.9,
        evaluate_temperature=0.1,
    )
    proposer = _ConfigRecordingModule(_FakePrediction(next_thought=["A", "B"]))
    evaluator = _ConfigRecordingModule(_FakePrediction(score=[5]))
    search = BeamSearch(config)
    search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    # Propose calls carry branching_factor as `n`; evaluate calls carry n_eval_samples.
    propose_configs = proposer.recorded
    evaluate_configs = evaluator.recorded
    assert propose_configs, "expected at least one proposer call"
    assert evaluate_configs, "expected at least one evaluator call"
    assert all(c["n"] == config.branching_factor for c in propose_configs)
    assert all(c["temperature"] == 0.9 for c in propose_configs)
    assert all(c["n"] == config.n_eval_samples for c in evaluate_configs)
    assert all(c["temperature"] == 0.1 for c in evaluate_configs)
