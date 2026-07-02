"""Core abstractions for pluggable ToT search algorithms.

Every search algorithm (beam, DFS, MCTS) implements :class:`ThoughtSearch` and
returns a :class:`SearchResult`. The strategy injects its proposer/evaluator
modules into ``search()``, keeping each algorithm LM-free and unit-testable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import dspy


@dataclass
class SearchResult:
    """What every search algorithm returns."""

    best_path: list[str]  # winning ordered thought-path, fed to the aggregator
    best_score: float  # evaluator score of best_path (1..10)
    depth_reached: int  # how deep the search actually went (<= max_depth)
    nodes_visited: int  # cross-algorithm efficiency metric (see definition below)
    propose_calls: int  # number of proposer LM calls
    evaluate_calls: int  # number of evaluator LM calls


@dataclass
class SearchConfig:
    """Shared + algorithm-specific knobs, populated from env vars."""

    # shared
    max_depth: int
    branching_factor: int
    n_eval_samples: int
    propose_temperature: float
    evaluate_temperature: float
    # Variant-specific semantics, resolved by _build_search_config: stop-on-success
    # for beam (early_stop_threshold); ToT's v_th pruning for DFS
    # (early_stop_threshold = prune threshold); MCTS stop-on-success (success_threshold).
    early_stop_threshold: float | None
    # algorithm-specific (optional; ignored by algos that don't use them)
    beam_width: int | None = None  # beam
    exploration_constant: float | None = None  # MCTS (c) -- assumes [0,1]-normalized Q
    max_iterations: int | None = None  # DFS / MCTS hard budget cap (beam ignores)
    success_threshold: float | None = (
        None  # DFS / MCTS stop-on-success (beam uses early_stop_threshold)
    )

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {self.branching_factor}")
        if self.n_eval_samples < 1:
            raise ValueError(f"n_eval_samples must be >= 1, got {self.n_eval_samples}")


class ThoughtSearch(ABC):
    """Search a tree of thought-paths using an injected proposer and evaluator.

    A node counts as **visited** when it is **expanded** -- i.e. it received at
    least one proposer expansion AND was scored by the evaluator. This count
    (``SearchResult.nodes_visited``) is comparable across algorithms. For MCTS,
    it equals the number of **expansion** steps, *not* the number of
    selection-phase traversals (which would inflate the count unfairly vs
    beam/DFS).

    The proposer/evaluator are DSPy modules owned and injected by
    :class:`~virex_bench.strategies.tot.strategy.ToTStrategy`; the search
    algorithm stays pure-algorithm (no LM construction). This keeps it
    unit-testable with fake callables.
    """

    name: str  # "beam" | "dfs" | "mcts"

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    @property
    def effective_max_iterations(self) -> int | None:
        """The iteration budget this search actually runs with, for the report.

        Beam has a fixed ``max_depth`` budget and no iteration cap, so the raw
        ``config.max_iterations`` (None) is already accurate. DFS and MCTS derive a
        concrete cap when the env leaves ``config.max_iterations`` unset, so they
        override this to report the resolved value instead of a misleading None.
        """
        return self.config.max_iterations

    @abstractmethod
    def search(
        self,
        *,
        premises: object,
        question: object,
        propose: dspy.Predict,
        evaluate: dspy.Predict,
    ) -> SearchResult: ...
