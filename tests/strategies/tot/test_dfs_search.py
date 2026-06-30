"""Unit tests for DFSSearch with fake proposer/evaluator modules.

Drives ``DFSSearch.search`` with canned thoughts/scores to verify depth-first descent,
backtracking on dead-ends, value pruning (ToT's ``v_th``), the budget cap, and counter
correctness -- without any real LM calls.
"""

from typing import Any

import pytest

from virex_bench.strategies.tot.search import SearchConfig
from virex_bench.strategies.tot.search.dfs import DFSSearch


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


def _make_config(**overrides: Any) -> SearchConfig:
    defaults: dict[str, Any] = {
        "max_depth": 3,
        "branching_factor": 2,
        "n_eval_samples": 1,
        "propose_temperature": 0.7,
        "evaluate_temperature": 0.0,
        "early_stop_threshold": None,
        "max_iterations": 20,
    }
    defaults.update(overrides)
    return SearchConfig(**defaults)


def test_dfs_descends_into_best_child_first() -> None:
    """DFS dives into the highest-scoring child before trying its siblings."""
    config = _make_config(max_depth=2, branching_factor=2, max_iterations=20)
    # Root -> [A(5), B(8)]; DFS takes B first (best), B -> [B1(6), B2(9)]; then A -> [A1(3), A2(4)].
    proposer = _FakeProposer(
        [
            ["A", "B"],  # root
            ["B1", "B2"],  # B (best root child, descended into first)
            ["A1", "A2"],  # A (after B's subtree is exhausted)
        ]
    )
    evaluator = _FakeEvaluator(
        [
            ["5"],  # A
            ["8"],  # B
            ["6"],  # B1
            ["9"],  # B2
            ["3"],  # A1
            ["4"],  # A2
        ]
    )
    search = DFSSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["B", "B2"]
    assert result.best_score == 9.0
    assert result.depth_reached == 2
    assert result.nodes_visited == 6
    assert result.propose_calls == 3
    assert result.evaluate_calls == 6


def test_dfs_backtracks_to_sibling_on_dead_end() -> None:
    """When the best child's subtree dead-ends, DFS backtracks and finds a better path."""
    # threshold=7: A(9) & B(8) survive at root; A's children all pruned -> backtrack to B.
    config = _make_config(
        max_depth=3, branching_factor=2, early_stop_threshold=7.0, max_iterations=20
    )
    proposer = _FakeProposer(
        [
            ["A", "B"],  # root -> A(9), B(8)
            ["A1", "A2"],  # A -> both pruned (dead-end)
            ["B1", "B2"],  # B -> B1(10), B2(7)
            ["B1a"],  # B1 (depth 2) -> B1a(8)
            ["B2a"],  # B2 (depth 2) -> B2a(5), pruned
        ]
    )
    evaluator = _FakeEvaluator(
        [
            ["9"],  # A
            ["8"],  # B
            ["3"],  # A1 (pruned)
            ["4"],  # A2 (pruned)
            ["10"],  # B1
            ["7"],  # B2
            ["8"],  # B1a
            ["5"],  # B2a (pruned)
        ]
    )
    search = DFSSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["B", "B1"]
    assert result.best_score == 10.0
    assert result.depth_reached == 3
    assert result.nodes_visited == 8  # pruned nodes still count
    assert result.propose_calls == 5
    assert result.evaluate_calls == 8


def test_dfs_prunes_low_scoring_children_but_still_counts_them() -> None:
    """Children below the threshold are evaluated (counted) but never expanded."""
    config = _make_config(
        max_depth=3, branching_factor=2, early_stop_threshold=8.0, max_iterations=20
    )
    proposer = _FakeProposer(
        [
            ["A", "B"],  # root -> A(9) survives, B(5) pruned
            ["A1", "A2"],  # A -> A1(10) survives, A2(7) pruned
            ["A1a"],  # A1 (depth 2) -> A1a(6) pruned (dead-end)
        ]
    )
    evaluator = _FakeEvaluator(
        [
            ["9"],  # A
            ["5"],  # B (pruned)
            ["10"],  # A1
            ["7"],  # A2 (pruned)
            ["6"],  # A1a (pruned)
        ]
    )
    search = DFSSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A", "A1"]
    assert result.best_score == 10.0
    assert result.depth_reached == 3
    assert result.nodes_visited == 5  # A, B, A1, A2, A1a (pruned ones included)
    assert result.propose_calls == 3
    assert result.evaluate_calls == 5


def test_dfs_budget_cap_limits_expansions() -> None:
    """``max_iterations`` caps the number of proposer calls (node expansions)."""
    config = _make_config(max_depth=5, branching_factor=1, max_iterations=2)
    proposer = _FakeProposer(
        [
            ["A"],  # root -> A(5)
            ["A1"],  # A -> A1(6); propose_calls now 2 == cap, search stops
            ["A2"],  # never reached
        ]
    )
    evaluator = _FakeEvaluator([["5"], ["6"]])
    search = DFSSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A", "A1"]
    assert result.best_score == 6.0
    assert result.propose_calls == 2  # exactly max_iterations
    assert proposer.call_count == 2
    assert result.nodes_visited == 2


def test_dfs_does_not_expand_nodes_at_max_depth() -> None:
    """A child at ``max_depth`` is evaluated but never expanded."""
    config = _make_config(max_depth=1, branching_factor=1, max_iterations=20)
    proposer = _FakeProposer(
        [
            ["A"],  # root -> A(9); A is at depth 1 == max_depth, not expanded
            ["A1"],  # never reached
        ]
    )
    evaluator = _FakeEvaluator([["9"]])
    search = DFSSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == ["A"]
    assert result.best_score == 9.0
    assert result.depth_reached == 1
    assert result.nodes_visited == 1
    assert result.propose_calls == 1
    assert proposer.call_count == 1


def test_dfs_default_max_iterations_when_unset() -> None:
    """When ``max_iterations`` is None it defaults to ``max_depth * branching_factor``."""
    config = _make_config(max_depth=3, branching_factor=4, max_iterations=None)
    search = DFSSearch(config)
    assert search.max_iterations == 12


def test_dfs_max_iterations_validation() -> None:
    """``max_iterations < 1`` is rejected."""
    config = _make_config(max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        DFSSearch(config)


def test_dfs_empty_proposals_stall_immediately() -> None:
    """When the root proposer returns no thoughts, the search produces an empty path."""
    config = _make_config(max_depth=3, branching_factor=2, max_iterations=20)
    proposer = _FakeProposer([[]])  # root produces nothing
    evaluator = _FakeEvaluator([])
    search = DFSSearch(config)
    result = search.search(
        premises=["p1"],
        question="q",
        propose=proposer,  # type: ignore[arg-type]
        evaluate=evaluator,  # type: ignore[arg-type]
    )

    assert result.best_path == []
    assert result.best_score == 0.0
    assert result.depth_reached == 0
    assert result.nodes_visited == 0
    assert result.propose_calls == 1
    assert result.evaluate_calls == 0
