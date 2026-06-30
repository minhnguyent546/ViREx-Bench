"""Beam search over the thought tree (ToT's "BFS", Yao et al. 2023).

Level-by-level greedy beam search: at each depth, every frontier node proposes
``branching_factor`` next thoughts; each candidate path is scored by the
evaluator (averaged over ``n_eval_samples`` votes); the ``beam_width`` best
survive to the next layer. The highest-scoring path seen across all depths is
returned as the winner.

Lifted verbatim from the original single-file ``strategies/tot.py`` forward
loop; only the bookkeeping (counter tracking + ``SearchResult`` return) is new.
"""

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


class BeamSearch(ThoughtSearch):
    """Greedy beam search — keep the top ``beam_width`` candidates per layer."""

    name = "beam"

    def __init__(self, config: SearchConfig) -> None:
        super().__init__(config)
        beam_width = config.beam_width
        if beam_width is None:
            raise ValueError("BeamSearch requires config.beam_width to be set")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    def search(
        self,
        *,
        premises: object,
        question: object,
        propose: dspy.Predict,
        evaluate: dspy.Predict,
    ) -> SearchResult:
        config = self.config
        beam_width = config.beam_width
        assert beam_width is not None  # validated in __init__

        frontier: list[ThoughtNode] = [ThoughtNode(path=[], score=0.0, depth=0)]
        best_leaf = frontier[0]
        nodes_visited = 0
        propose_calls = 0
        evaluate_calls = 0
        depth_reached = 0

        for depth in range(config.max_depth):
            candidates: list[ThoughtNode] = []
            for node in frontier:
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
                    child = ThoughtNode(path=child_path, score=score, depth=depth + 1)
                    candidates.append(child)
                    nodes_visited += 1
                    depth_reached = depth + 1
                    if is_better(child, best_leaf):
                        best_leaf = child

            if (
                config.early_stop_threshold is not None
                and best_leaf.score >= config.early_stop_threshold
            ):
                logger.debug(
                    f"ToT early stop at depth {depth + 1}: best score {best_leaf.score:.1f} "
                    f">= threshold {config.early_stop_threshold}"
                )
                break
            # TODO: plateau/stall detection -- break when best_leaf.score does not
            # strictly improve for `patience` consecutive depths. This catches
            # chains that flatline below early_stop_threshold (e.g. the proposer
            # emitting "no further reasoning needed" filler the evaluator scores
            # identically to real progress). Needs a `patience` knob on
            # SearchConfig + a VIREX_BENCH_TOT_BEAM_PATIENCE env var.
            if not candidates:
                logger.debug(f"ToT search stalled at depth {depth + 1} (no candidates)")
                break
            candidates.sort(key=lambda node: node.score, reverse=True)
            frontier = candidates[:beam_width]

        logger.debug(
            f"ToT search done: explored {nodes_visited} nodes, "
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
