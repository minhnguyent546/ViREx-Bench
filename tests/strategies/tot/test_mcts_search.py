"""Unit tests for MCTSSearch with fake proposer/evaluator modules.

Drives ``MCTSSearch.search`` with canned thoughts/scores to verify UCT
exploitation, the mean-backprop / robust-child mechanics, the ``nodes_visited``
(expansion-only) vs ``evaluate_calls`` (incl. re-evaluations) split, reward
normalization, early stop, default ``max_iterations``, and per-branch
context-overflow recovery -- without any real LM calls.

The robust-child final selection (``_robust_child_selection``) is also exercised
directly with hand-built trees, decoupling it from UCT traversal nondeterminism.
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest

from virex_bench.strategies.tot.search import SearchConfig
from virex_bench.strategies.tot.search.mcts import (
    _DEFAULT_MAX_ITERATIONS,  # pyright: ignore[reportPrivateUsage]
    MCTSSearch,
    _MCTSNode,  # pyright: ignore[reportPrivateUsage]
    _robust_child_selection,  # pyright: ignore[reportPrivateUsage]
)


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


class _PathKeyedEvaluator:
    """Score by the LAST thought in the reasoning path (order-independent).

    Lets multi-iteration UCT tests feed scores without predicting the exact
    child-selection order: the score depends only on which leaf is being
    evaluated, not on call sequence.
    """

    def __init__(self, score_by_last_thought: dict[str, str]) -> None:
        self.score_by_last_thought = score_by_last_thought
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> _FakePrediction:
        self.call_count += 1
        path = kwargs["reasoning_path"]
        # render_thought_path formats each step as "[Bước N] {thought}"; the
        # last step follows the final "[Bước " marker.
        marker = path.rpartition("[Bước ")[2]
        last_thought = marker.partition("] ")[2] if "] " in marker else marker
        score = self.score_by_last_thought.get(last_thought, "5")
        return _FakePrediction(score=[score])


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
        "max_depth": 3,
        "branching_factor": 2,
        "n_eval_samples": 1,
        "propose_temperature": 0.7,
        "evaluate_temperature": 0.0,
        "early_stop_threshold": None,
        "beam_width": None,
        "exploration_constant": 1.414,
        "max_iterations": 5,
        "success_threshold": None,
    }
    defaults.update(overrides)
    return SearchConfig(**defaults)


def _run(
    config: SearchConfig,
    proposer: Any,
    evaluator: Any,
) -> Any:
    return MCTSSearch(config).search(
        premises=["p1"],
        question="q",
        propose=proposer,
        evaluate=evaluator,
    )


def test_mcts_single_expansion_returns_best_child() -> None:
    """One expansion: both children evaluated; robust child is the higher-Q one
    (visit tie broken by Q). best_score denormalizes the leaf's mean Q."""
    config = _make_config(max_depth=1, branching_factor=2, max_iterations=1)
    proposer = _FakeProposer([["A", "B"]])
    evaluator = _FakeEvaluator([["9"], ["5"]])
    result = _run(config, proposer, evaluator)

    assert result.best_path == ["A"]
    assert result.best_score == 9.0  # denormalize(8/9) = 8/9*9 + 1 = 9.0
    assert result.depth_reached == 1
    assert result.nodes_visited == 2
    assert result.propose_calls == 1
    assert result.evaluate_calls == 2


def test_mcts_uct_exploits_higher_scoring_child() -> None:
    """With max_depth=1, all post-root expansions are re-evaluations. UCT
    exploitation drives more visits to the higher-Q child, so robust-child
    selection lands on it. Re-evals count toward evaluate_calls but NOT
    nodes_visited (the expansion-only metric)."""
    config = _make_config(max_depth=1, branching_factor=2, max_iterations=5)
    proposer = _FakeProposer([["A", "B"]])  # only the root expansion is proposed
    evaluator = _PathKeyedEvaluator({"A": "9", "B": "3"})
    result = _run(config, proposer, evaluator)

    # 1 expansion (propose + 2 evals) + 4 re-evals = 6 evaluate calls; only the
    # 2 expansion children count as nodes_visited.
    assert result.best_path == ["A"]
    assert result.best_score == 9.0
    assert result.propose_calls == 1
    assert result.evaluate_calls == 6
    assert result.nodes_visited == 2
    assert result.depth_reached == 1


def test_mcts_reevaluation_not_counted_as_nodes_visited() -> None:
    """Re-evaluations (MCTSr repeated sampling) hit evaluate_calls but never
    nodes_visited -- the canonical expanded-node count stays comparable to
    beam/DFS."""
    config = _make_config(max_depth=1, branching_factor=2, max_iterations=3)
    proposer = _FakeProposer([["A", "B"]])
    evaluator = _PathKeyedEvaluator({"A": "9", "B": "3"})
    result = _run(config, proposer, evaluator)

    assert result.nodes_visited == 2  # the 2 expansion children only
    assert result.evaluate_calls == 4  # 2 expansion + 2 re-evals
    assert result.propose_calls == 1


def test_mcts_max_iterations_honored() -> None:
    """``max_iterations`` caps the number of proposer calls (expansions). Two
    iterations -> root expansion + one child expansion -> 2 propose calls."""
    config = _make_config(max_depth=3, branching_factor=2, max_iterations=2)
    proposer = _FakeProposer([["A", "B"], ["A1", "A2"]])
    evaluator = _PathKeyedEvaluator({"A": "8", "B": "3", "A1": "7", "A2": "6"})
    result = _run(config, proposer, evaluator)

    assert result.propose_calls == 2
    assert result.evaluate_calls == 4
    assert result.nodes_visited == 4
    assert result.depth_reached == 2
    # Robust child: A (3 visits) over B (1); then A1 (tie visits, higher Q).
    assert result.best_path == ["A", "A1"]
    assert result.best_score == 7.0  # denormalize(6/9)


def test_mcts_robust_child_picks_most_visited_not_highest_score() -> None:
    """``_robust_child_selection`` descends by visit count (tiebreak: Q), so a
    more-visited / lower-raw-score child beats a less-visited / higher-one."""
    root = _MCTSNode(path=[], depth=0)
    high_visits = _MCTSNode(path=["A"], depth=1, parent=root, score=5.0)
    high_visits.visits = 5
    high_visits.total_value = 2.5  # Q = 0.5 -> denormalize = 5.5
    high_score = _MCTSNode(path=["B"], depth=1, parent=root, score=9.0)
    high_score.visits = 2
    high_score.total_value = 1.8  # Q = 0.9 -> denormalize = 9.1
    root.children = [high_visits, high_score]

    _leaf, path, score = _robust_child_selection(root)
    assert path == ["A"]
    assert score == pytest.approx(5.5)


def test_mcts_robust_child_tiebreaks_on_q_value() -> None:
    """Equal visit counts -> the higher-Q child wins the tiebreak."""
    root = _MCTSNode(path=[], depth=0)
    low_q = _MCTSNode(path=["A"], depth=1, parent=root)
    low_q.visits = 3
    low_q.total_value = 0.6  # Q = 0.2
    high_q = _MCTSNode(path=["B"], depth=1, parent=root)
    high_q.visits = 3
    high_q.total_value = 2.4  # Q = 0.8
    root.children = [low_q, high_q]

    _leaf, path, score = _robust_child_selection(root)
    assert path == ["B"]
    assert score == pytest.approx(8.2)  # denormalize(0.8) = 0.8*9 + 1


def test_mcts_robust_child_empty_root_returns_zero() -> None:
    """A root with no children (empty/dead-end stall) -> empty path, zero."""
    root = _MCTSNode(path=[], depth=0)
    _leaf, path, score = _robust_child_selection(root)
    assert path == []
    assert score == 0.0


def test_mcts_reward_normalization_round_trip() -> None:
    """A single child scored 5 -> normalized 4/9 for UCT -> denormalized back to
    5.0 for best_score, proving the [0,1]<->[1,10] mapping is symmetric."""
    config = _make_config(max_depth=1, branching_factor=1, max_iterations=1)
    proposer = _FakeProposer([["A"]])
    evaluator = _FakeEvaluator([["5"]])
    result = _run(config, proposer, evaluator)

    assert result.best_path == ["A"]
    assert result.best_score == 5.0
    assert result.nodes_visited == 1


def test_mcts_default_max_iterations_when_unset() -> None:
    """When ``max_iterations`` is None it defaults to DEFAULT_MAX_ITERATIONS."""
    config = _make_config(max_iterations=None)
    search = MCTSSearch(config)
    assert search.max_iterations == _DEFAULT_MAX_ITERATIONS


def test_mcts_max_iterations_validation() -> None:
    """``max_iterations < 1`` is rejected."""
    config = _make_config(max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        MCTSSearch(config)


def test_mcts_exploration_constant_required_and_non_negative() -> None:
    """MCTSSearch rejects a missing or negative exploration constant."""
    config_missing = _make_config(exploration_constant=None)
    with pytest.raises(ValueError, match="exploration_constant"):
        MCTSSearch(config_missing)
    config_negative = _make_config(exploration_constant=-1.0)
    with pytest.raises(ValueError, match="exploration_constant"):
        MCTSSearch(config_negative)


def test_mcts_empty_proposals_stall_immediately() -> None:
    """When the root proposer returns no thoughts, the search yields an empty
    path and stops -- mirroring beam/DFS stall."""
    config = _make_config(max_depth=3, branching_factor=2, max_iterations=5)
    proposer = _FakeProposer([[]])
    evaluator = _PathKeyedEvaluator({})
    result = _run(config, proposer, evaluator)

    assert result.best_path == []
    assert result.best_score == 0.0
    assert result.depth_reached == 0
    assert result.nodes_visited == 0
    assert result.propose_calls == 1
    assert result.evaluate_calls == 0


def test_mcts_stop_on_success() -> None:
    """MCTS halts once the max raw score reaches ``success_threshold``, even
    with budget remaining -- consistent with beam/DFS stop-on-success."""
    config = _make_config(
        max_depth=5, branching_factor=2, max_iterations=10, success_threshold=9.0
    )
    proposer = _FakeProposer([["A", "B"]])
    evaluator = _PathKeyedEvaluator({"A": "9", "B": "5"})
    result = _run(config, proposer, evaluator)

    # iter1 expands root (max_raw_score -> 9.0); iter2 checks and breaks.
    assert result.best_path == ["A"]
    assert result.best_score == 9.0
    assert result.propose_calls == 1
    assert result.evaluate_calls == 2
    assert result.nodes_visited == 2


def test_mcts_root_propose_overflow_returns_empty() -> None:
    """An overflow on the root expansion returns the empty path; the
    overflowing propose call is not counted -- consistent with beam/DFS."""
    config = _make_config(max_depth=3, branching_factor=2, max_iterations=5)
    proposer = _OverflowProposer(thoughts_before_overflow=[], overflow_on_call=1)
    evaluator = _PathKeyedEvaluator({})
    result = _run(config, proposer, evaluator)

    assert result.best_path == []
    assert result.best_score == 0.0
    assert result.depth_reached == 0
    assert result.nodes_visited == 0
    assert result.propose_calls == 0
    assert result.evaluate_calls == 0


def test_mcts_child_evaluate_overflow_skips_child() -> None:
    """An overflow while evaluating one child skips just that child (graceful
    degradation); the surviving child still shapes the tree and the result."""
    config = _make_config(max_depth=1, branching_factor=2, max_iterations=1)
    proposer = _FakeProposer([["A", "B"]])
    evaluator = _OverflowEvaluator(scores_before_overflow=[["8"]], overflow_on_call=2)
    result = _run(config, proposer, evaluator)

    # A evaluated (score 8); B's evaluation overflowed -> skipped, not counted.
    assert result.best_path == ["A"]
    assert result.best_score == 8.0
    assert result.nodes_visited == 1
    assert result.propose_calls == 1
    assert result.evaluate_calls == 1
    assert evaluator.call_count == 2  # invoked twice (the 2nd overflowed)
