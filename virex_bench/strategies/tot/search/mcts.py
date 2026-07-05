"""Monte-Carlo Tree Search over the thought tree (ToT's "MCTS", Phase 3).

UCT-driven MCTS (Kocsis & Szepesvari 2006; Browne et al. 2012) adapted for LLM
reasoning per the LLM-MCTS literature (RAP, MCTSr, AlphaMath, LATS). Key
adaptations:

1. **No separate rollout -- the evaluator IS the value function** (AlphaGo Zero
   / MCTSr ``lambda=0`` style): the value backpropagated from a newly expanded
   child is its normalized evaluator score. A separate rollout call would cost
   an extra LM call per iteration and compound multi-step rollout noise on top
   of the already-noisy evaluator; ``evaluate`` already estimates "how close to
   a correct final answer". This keeps ``total_llm_calls = propose_calls +
   evaluate_calls + 1`` comparable across beam/DFS/MCTS.

2. **Reward normalization to [0,1]**: evaluator scores (1-10) are normalized
   via ``(score - 1) / 9`` for UCT; ``best_score`` is reported in the raw 1-10
   scale via ``denormalize(Q) = Q * 9 + 1``.

3. **Mean backpropagation** (``Q = W / N``) -- recommended for low-simulation
   regimes and does not amplify the evaluator's false-positive rate (Coulom 2006).

4. **Full expansion per iteration** (MCTSr-style): select a frontier leaf,
   propose ``branching_factor`` children, evaluate *all* deduped children, and
   backpropagate each. Matches beam/DFS's per-node cost and ``nodes_visited``
   (every evaluated child counts).

5. **Best-scored final selection**: return the explored node with the highest
   evaluator score (tie: shorter path), matching beam/DFS's "best path seen"
   behavior and avoiding discarding a complete high-scoring derivation just
   because robust-child leaf visits are sparse under small budgets; robust-child
   is logged as a diagnostic.

6. **Re-evaluation on revisit** (MCTSr repeated sampling): a non-expandable leaf
   (max-depth terminal or expanded dead-end) is re-evaluated and the new sample
   is backpropagated. These count toward ``evaluate_calls`` but NOT
   ``nodes_visited`` (no proposer call), so for MCTS ``evaluate_calls >=
   nodes_visited`` is expected.

Each iteration is one proposer call plus ``b`` evaluator calls (on expansion)
or one evaluator call (on re-evaluation). Early stop mirrors beam/DFS: halts
when the max raw score reaches ``config.success_threshold``. Context-window
overflows are caught per-branch (consistent with DFS): overflow on propose
exhausts the node; on a child's evaluate skips that child; on re-evaluation
exhausts the leaf. An empty root expansion returns the empty path (mirrors
beam/DFS stall).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import dspy

from virex_bench.logger import init_logger
from virex_bench.strategies.tot.common import (
    completion_values,
    dedupe_thoughts,
    mean_score,
    render_thought_path,
)
from virex_bench.strategies.tot.search.base import SearchConfig, SearchResult, ThoughtSearch

logger = init_logger(__name__)

# Evaluator scores are integers in 1..10 (see common._EVAL_* / the evaluator
# signature's score bands). UCT assumes [0,1] rewards, so we normalize for the
# internal MCTS value math and denormalize again when reporting best_score.
_EVAL_MIN_SCORE = 1
_EVAL_MAX_SCORE = 10
_SCORE_RANGE = _EVAL_MAX_SCORE - _EVAL_MIN_SCORE

# Numerical safety inside UCT (MCTSr Eq. 4 form): guards against log(0) /
# divide-by-zero. Children always have visits >= 1 (evaluated on creation).
_LOG_OFFSET = 1
_VISIT_EPSILON = 1e-6

_DEFAULT_MAX_ITERATIONS = 30


def _normalize_score(raw_score: float) -> float:
    """Map a raw 1-10 evaluator score to the [0,1] interval UCT expects."""
    return (raw_score - _EVAL_MIN_SCORE) / _SCORE_RANGE


def _denormalize_score(value: float) -> float:
    """Map a [0,1] MCTS value back to the raw 1-10 evaluator scale."""
    return value * _SCORE_RANGE + _EVAL_MIN_SCORE


def _robust_child_selection(root: _MCTSNode) -> tuple[_MCTSNode | None, list[str], float]:
    """Pick the most-visited root-to-leaf path; score = denormalize(leaf.Q).

    Robust-child selection dampens evaluator noise via visit count (max-child
    would inherit the beam/DFS false-positive failure mode). Returns
    ``(None, [], 0)`` when the root produced no children.
    """
    if not root.children:
        logger.debug("ToT MCTS robust_child: empty root -> no path")
        return None, [], 0.0
    node = root
    while node.children:
        # Most-visited wins; ties broken by highest Q for a deterministic,
        # quality-aware resolution.
        node = max(node.children, key=lambda child: (child.visits, child.q_value))

    return node, node.path, _denormalize_score(node.q_value)


def _best_scored_selection(root: _MCTSNode) -> tuple[_MCTSNode | None, list[str], float]:
    """Pick the highest evaluator-scored explored node; tie-break on shorter path."""

    best_node: _MCTSNode | None = None
    stack = list(root.children)
    while stack:
        node = stack.pop()
        if (
            best_node is None
            or node.score > best_node.score
            or (node.score == best_node.score and len(node.path) < len(best_node.path))
        ):
            best_node = node
        stack.extend(node.children)
    if best_node is None:
        return None, [], 0.0
    return best_node, best_node.path, best_node.score


@dataclass
class _MCTSNode:
    """One node of the MCTS tree.

    ``visits`` (N) and ``total_value`` (W) implement ``Q = W / N`` over
    normalized [0,1] rewards. ``score`` is the node's raw 1-10 evaluator score
    (0.0 for the un-scored root) for diagnostics; the value flowing through
    UCT/backprop is normalized. ``is_expanded`` distinguishes "propose has been
    called here" (descend via UCT) from "not yet expanded" (frontier -> expand)
    and "expanded with no children" (dead-end -> re-evaluate). ``is_terminal``
    means the evaluator scored the path at the success threshold (more thoughts
    would be redundant). ``is_exhausted`` means an overflow / terminal / all
    descendants exhausted; the node stays available to final selection but UCT
    spends no more budget on it.
    """

    path: list[str]
    depth: int
    parent: _MCTSNode | None = None
    children: list[_MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    score: float = 0.0  # raw 1-10 from the node's own evaluation; 0.0 for root
    is_expanded: bool = False
    is_terminal: bool = False
    is_exhausted: bool = False

    @property
    def q_value(self) -> float:
        """Mean backpropagated value in [0,1] (0 when unvisited)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MCTSSearch(ThoughtSearch):
    """UCT MCTS with value-function-as-rollout (no separate rollout LM call)."""

    name = "mcts"

    def __init__(self, config: SearchConfig) -> None:
        super().__init__(config)
        max_iterations = config.max_iterations
        if max_iterations is None:
            # MCTS has no natural cap -- the iteration budget IS the search.
            # 30 full expansions (1 propose + b evaluate each) is the
            # empirically tuned default.
            max_iterations = _DEFAULT_MAX_ITERATIONS
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        self.max_iterations = max_iterations

        exploration_constant = config.exploration_constant
        if exploration_constant is None:
            raise ValueError("MCTSSearch requires config.exploration_constant to be set")
        if exploration_constant < 0.0:
            raise ValueError(f"exploration_constant must be >= 0, got {exploration_constant}")
        self.exploration_constant = exploration_constant

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
        success_threshold = config.success_threshold
        max_iterations = self.max_iterations
        exploration_constant = self.exploration_constant

        nodes_visited = 0
        propose_calls = 0
        evaluate_calls = 0
        depth_reached = 0
        max_raw_score = 0.0  # raw 1-10, drives the early-stop check

        root = _MCTSNode(path=[], depth=0)

        def uct_select(node: _MCTSNode) -> _MCTSNode | None:
            """Descend from ``node`` via UCT to a frontier (actionable) node.

            Frontier nodes (not yet expanded), max-depth terminals, and expanded
            dead-ends are returned as-is so the main loop can take the right
            action (expand vs re-evaluate). Exhausted branches are skipped so an
            overflowed leaf cannot consume the budget through repeated
            selection.
            """
            if node.is_exhausted:
                return None
            if not (node.is_expanded and node.children and node.depth < config.max_depth):
                return node

            selectable_children = [child for child in node.children if not child.is_exhausted]
            if not selectable_children:
                node.is_exhausted = True
                return None

            log_parent = math.log(node.visits + _LOG_OFFSET)
            ranked_children = sorted(
                selectable_children,
                key=lambda child: (
                    child.q_value
                    + exploration_constant
                    * math.sqrt(log_parent / (child.visits + _VISIT_EPSILON))
                ),
                reverse=True,
            )
            for child in ranked_children:
                selected = uct_select(child)
                if selected is not None:
                    return selected
            node.is_exhausted = True
            return None

        def backprop(node: _MCTSNode, value: float) -> None:
            """Propagate a normalized [0,1] value up to the root (every-visit)."""
            current: _MCTSNode | None = node
            while current is not None:
                current.visits += 1
                current.total_value += value
                current = current.parent

        def expand(node: _MCTSNode) -> bool:
            """Propose, evaluate all deduped children, add them, backprop each.

            Returns False if the proposer call overflowed (the node is marked
            expanded + exhausted); True otherwise. Per-child evaluate overflows
            skip just that child (consistent with DFS).
            """
            nonlocal propose_calls, evaluate_calls, nodes_visited, depth_reached, max_raw_score

            try:
                proposed = propose(
                    premises=premises,
                    question=question,
                    reasoning_so_far=render_thought_path(node.path),
                    config=config.propose_sampling_config,
                )
            except dspy.ContextWindowExceededError:
                logger.debug(
                    f"ToT MCTS: context window exceeded proposing at depth {node.depth}; "
                    f"treating branch as dead-end"
                )
                node.is_expanded = True
                node.is_exhausted = True
                return False
            propose_calls += 1
            thoughts = dedupe_thoughts(completion_values(proposed, "next_thought"))

            node.is_expanded = True
            for thought in thoughts:
                child_path = [*node.path, thought]
                try:
                    evaluated = evaluate(
                        premises=premises,
                        question=question,
                        reasoning_path=render_thought_path(child_path),
                        config=config.evaluate_sampling_config,
                    )
                except dspy.ContextWindowExceededError:
                    logger.debug(
                        f"ToT MCTS: context window exceeded evaluating a child at "
                        f"depth {node.depth + 1}; skipping that child"
                    )
                    continue
                evaluate_calls += 1
                raw_score = mean_score(completion_values(evaluated, "score"))
                normalized = _normalize_score(raw_score)
                child = _MCTSNode(
                    path=child_path,
                    depth=node.depth + 1,
                    parent=node,
                    score=raw_score,
                )
                if success_threshold is not None and raw_score >= success_threshold:
                    child.is_terminal = True
                    child.is_exhausted = True
                node.children.append(child)
                # Seed the child with its own first sample, then propagate the
                # same value up the path to the root (every-visit backprop).
                backprop(child, normalized)
                nodes_visited += 1
                if child.depth > depth_reached:
                    depth_reached = child.depth
                if raw_score > max_raw_score:
                    max_raw_score = raw_score
            return True

        def reevaluate(node: _MCTSNode) -> None:
            """Re-evaluate a non-expandable leaf and backprop the new sample.

            Refines ``Q`` for deep leaves (MCTSr repeated sampling) and prevents
            stalling when all frontier nodes are terminals. Does NOT count toward
            ``nodes_visited`` (no proposer call -> not an expansion).
            """
            nonlocal evaluate_calls, max_raw_score
            try:
                evaluated = evaluate(
                    premises=premises,
                    question=question,
                    reasoning_path=render_thought_path(node.path),
                    config=config.evaluate_sampling_config,
                )
            except dspy.ContextWindowExceededError:
                logger.debug(
                    f"ToT MCTS: context window exceeded re-evaluating at depth "
                    f"{node.depth}; skipping re-evaluation"
                )
                node.is_exhausted = True
                return
            evaluate_calls += 1
            raw_score = mean_score(completion_values(evaluated, "score"))
            backprop(node, _normalize_score(raw_score))
            # Early-stop tracks the raw max; a re-eval sample can lift it too.
            if raw_score > max_raw_score:
                max_raw_score = raw_score

        early_stopped = False
        reeval_count = 0

        for _ in range(max_iterations):
            if success_threshold is not None and max_raw_score >= success_threshold:
                logger.debug(
                    f"ToT MCTS early stop: max raw score {max_raw_score:.1f} "
                    f">= threshold {success_threshold}"
                )
                early_stopped = True
                break

            frontier = uct_select(root)
            if frontier is None:
                break
            if not frontier.is_expanded and frontier.depth < config.max_depth:
                expand(frontier)
                # Empty/dead-end root expansion: nothing to search. Mirror
                # beam/DFS stall and stop immediately.
                if frontier is root and not root.children:
                    break
            else:
                # Terminal (max depth) or expanded dead-end: re-evaluate to
                # refine Q (MCTSr repeated sampling).
                reeval_count += 1
                reevaluate(frontier)

        logger.debug(
            f"ToT MCTS done: {nodes_visited} nodes over {propose_calls} expansions, "
            f"{evaluate_calls} evaluations, best raw score {max_raw_score:.2f}"
        )

        robust_leaf, robust_path, robust_score = _robust_child_selection(root)
        best_leaf, best_path, best_score = _best_scored_selection(root)
        mechanism = (
            "engaged" if robust_leaf is not None and robust_leaf.visits > 1 else "idle (N=1)"
        )
        logger.debug(
            f"ToT MCTS summary: best_score={best_score:.2f} "
            f"depth={depth_reached} nodes={nodes_visited} "
            f"propose={propose_calls} eval={evaluate_calls} "
            f"reevals={reeval_count} early_stopped={early_stopped} "
            f"selected_N={best_leaf.visits if best_leaf is not None else 0} "
            f"robust_score={robust_score:.2f} robust_depth={len(robust_path)} "
            f"robust_child={mechanism}"
        )
        return SearchResult(
            best_path=best_path,
            best_score=best_score,
            depth_reached=depth_reached,
            nodes_visited=nodes_visited,
            propose_calls=propose_calls,
            evaluate_calls=evaluate_calls,
        )
