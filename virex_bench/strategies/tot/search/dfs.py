"""Depth-first search with backtracking over the thought tree (ToT's "DFS").

Implements ToT's DFS (Yao et al. 2023, Algorithm 2): from a node, propose
``branching_factor`` candidate next-thoughts, evaluate each, **prune** children whose
evaluator score falls below ``early_stop_threshold`` (ToT's value-pruning threshold
``v_th`` -- defaults to 3.0, the "dead end" boundary of the evaluator's 1-10 scale), then
recurse depth-first into the highest-scoring surviving child. When a node has no
surviving children (a dead-end) the search **backtracks** to the parent and tries the
next sibling.

A **stop-on-success** threshold (``config.success_threshold``) halts the entire search
once the best path scores >= this value -- analogous to beam's
``early_stop_threshold``. Without it, DFS always exhausts its ``max_iterations`` budget
even after finding a perfect path.

Unlike beam, DFS keeps no frontier: it commits to one path at a time and can recover
branches beam would prune irrevocably. It has no natural budget cap, so
``config.max_iterations`` caps the number of node expansions (one proposer call each) as a
hard safety valve; the pruning threshold does the real budget control in the common case.
When ``max_iterations`` is unset it defaults to ``max_depth * branching_factor``
expansions so ``tot-dfs`` runs out of the box without risking exponential blowup.

Context-window overflows on a deep branch are caught **per-branch**: the overflowing
branch is treated as a dead-end and DFS backtracks to try shorter siblings, rather than
aborting the entire search. An overflow on the root expansion (rare) falls through to the
outer handler and returns the best path found so far.

Note on the shared threshold: ``early_stop_threshold`` is ToT's ``v_th`` here -- children
scored below it are evaluated and counted toward ``nodes_visited`` but are not expanded.
Beam reuses the same config field as a stop-on-success threshold; see each algorithm's
docstring for its exact semantics.
"""

from dataclasses import dataclass

import dspy

from virex_bench.logger import init_logger
from virex_bench.strategies.tot.common import (
    ThoughtNode,
    completion_values,
    dedupe_thoughts,
    is_better,
    mean_score,
    render_thought_path,
)
from virex_bench.strategies.tot.search.base import SearchConfig, SearchResult, ThoughtSearch

logger = init_logger(__name__)


@dataclass
class _DFSFrame:
    """One stack frame: a node, its surviving children, and the next-sibling index.

    Surviving children are sorted best-first (highest score first) so DFS descends into
    the most promising child first; ``next_index`` tracks which sibling to try next when
    the search backtracks to this frame.
    """

    node: ThoughtNode
    survivors: list[ThoughtNode]
    next_index: int


class DFSSearch(ThoughtSearch):
    """Depth-first search with backtracking + value pruning (ToT's ``v_th``)."""

    name = "dfs"

    def __init__(self, config: SearchConfig) -> None:
        super().__init__(config)
        max_iterations = config.max_iterations
        if max_iterations is None:
            # DFS has no natural cap; default to a tree-shape-aware safety valve so
            # `tot-dfs` runs out of the box without risking exponential blowup.
            max_iterations = config.max_depth * config.branching_factor
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        self.max_iterations = max_iterations

    @property
    def effective_max_iterations(self) -> int | None:
        return self.max_iterations

    def search(
        self,
        *,
        premises: object,
        question: object,
        propose: dspy.Predict,
        evaluate: dspy.Predict,
    ) -> SearchResult:
        config = self.config
        early_stop_threshold = config.early_stop_threshold
        success_threshold = config.success_threshold
        max_iterations = self.max_iterations

        nodes_visited = 0
        propose_calls = 0
        evaluate_calls = 0
        depth_reached = 0
        best_leaf = ThoughtNode(path=[], score=0.0, depth=0)

        def expand(node: ThoughtNode) -> list[ThoughtNode]:
            """Propose, evaluate, and prune the children of ``node``.

            Returns surviving children (score >= ``early_stop_threshold`` when set),
            sorted by score descending so the best is recursed into first. Every
            evaluated child updates the shared counters and ``best_leaf`` -- consistent
            with beam's ``nodes_visited`` = expanded-and-evaluated definition.
            """
            nonlocal propose_calls, evaluate_calls, nodes_visited, depth_reached, best_leaf

            proposed = propose(
                premises=premises,
                question=question,
                reasoning_so_far=render_thought_path(node.path),
                config={
                    "n": config.branching_factor,
                    "temperature": config.propose_temperature,
                },
            )
            propose_calls += 1
            thoughts = dedupe_thoughts(completion_values(proposed, "next_thought"))

            survivors: list[ThoughtNode] = []
            for thought in thoughts:
                child_path = [*node.path, thought]
                evaluated = evaluate(
                    premises=premises,
                    question=question,
                    reasoning_path=render_thought_path(child_path),
                    config={
                        "n": config.n_eval_samples,
                        "temperature": config.evaluate_temperature,
                    },
                )
                evaluate_calls += 1
                score = mean_score(completion_values(evaluated, "score"))
                child = ThoughtNode(path=child_path, score=score, depth=node.depth + 1)
                nodes_visited += 1
                if child.depth > depth_reached:
                    depth_reached = child.depth
                if is_better(child, best_leaf):
                    best_leaf = child
                # Prune dead-end children (ToT's v_th): evaluated & counted, not expanded.
                if early_stop_threshold is not None and score < early_stop_threshold:
                    continue
                survivors.append(child)
            survivors.sort(key=lambda candidate: candidate.score, reverse=True)
            return survivors

        # Iterative DFS with an explicit stack. The root is expanded first (its children
        # are depth 1); the root itself is never scored, mirroring beam's uncounted root
        # placeholder. Each frame tracks which surviving child to descend into next.
        root = ThoughtNode(path=[], score=0.0, depth=0)
        try:
            stack: list[_DFSFrame] = [_DFSFrame(node=root, survivors=expand(root), next_index=0)]

            while stack and propose_calls < max_iterations:
                if success_threshold is not None and best_leaf.score >= success_threshold:
                    logger.debug(
                        f"ToT DFS early stop: best score {best_leaf.score:.1f} "
                        f">= threshold {success_threshold}"
                    )
                    break

                frame = stack[-1]
                if frame.next_index < len(frame.survivors):
                    child = frame.survivors[frame.next_index]
                    frame.next_index += 1
                    # Leaves at max_depth cannot be expanded; try the next sibling.
                    if child.depth >= config.max_depth:
                        continue
                    try:
                        stack.append(_DFSFrame(node=child, survivors=expand(child), next_index=0))
                    except dspy.ContextWindowExceededError:
                        logger.debug(
                            f"ToT DFS: context window exceeded at depth {child.depth}; "
                            f"treating branch as dead-end"
                        )
                        continue
                else:
                    # All siblings exhausted -> backtrack to the parent.
                    stack.pop()
        except dspy.ContextWindowExceededError:
            logger.debug(
                f"ToT DFS: context window exceeded at root; returning best path so far "
                f"(depth {best_leaf.depth}, score {best_leaf.score:.2f})"
            )

        logger.debug(
            f"ToT DFS done: explored {nodes_visited} nodes over {propose_calls} expansions, "
            f"best_leaf depth={best_leaf.depth} score={best_leaf.score:.2f}"
        )

        return SearchResult(
            best_path=best_leaf.path,
            best_score=best_leaf.score,
            depth_reached=depth_reached,
            nodes_visited=nodes_visited,
            propose_calls=propose_calls,
            evaluate_calls=evaluate_calls,
        )
