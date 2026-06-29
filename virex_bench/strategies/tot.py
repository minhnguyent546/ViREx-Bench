"""Tree-of-Thoughts (ToT) prompting strategy (Yao et al., 2023).

ToT deliberates over a problem by maintaining a *tree of partial reasoning
states* and searching over it with beam search, instead of committing to a
single chain like CoT. Each search step performs three operations, each backed
by an internal :class:`dspy.Signature` (these are strategy-internal, not defined
by the task — the task only supplies the final answer contract):

1. **Propose** — from each frontier node, sample several distinct next reasoning
   steps (a thought) given the premises, question, and the reasoning so far.
2. **Evaluate** — score how promising each candidate thought-path is on a 1–10
   scale (averaged over ``n_eval_samples`` independent votes).
3. **Select** — keep the ``beam_width`` highest-scoring candidates as the next
   frontier (greedy beam search).

After the search, a final **aggregate** step feeds the best explored thought-path
back into the task signature (with an added ``reasoning_path`` input) so the
model commits to a well-formed answer grounded in the exploration. The recorded
``reasoning`` on the returned prediction is the chosen thought-path, so the
evaluator and judge see the actual ToT trace.

Config knobs are read from environment variables (see :mod:`virex_bench.envs`,
prefix ``VIREX_BENCH_TOT_*``); explicit ``__init__`` kwargs override them for
programmatic use (e.g. tests). The LM is picked up from ``dspy.settings.lm``
(set by the evaluator) — per-call ``config`` overrides temperature / ``n`` so no
global state is mutated, keeping the strategy safe under the evaluator's thread
pool.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import dspy
from pydantic.fields import FieldInfo

from virex_bench import envs
from virex_bench.logger import init_logger
from virex_bench.strategies.base import ReasoningStrategy

logger = init_logger(__name__)

# Score band used by the evaluator. Values outside it are clamped on parse.
_EVAL_MIN_SCORE = 1
_EVAL_MAX_SCORE = 10
# Neutral score used when the evaluator emits an unparseable value.
_FALLBACK_SCORE = (_EVAL_MIN_SCORE + _EVAL_MAX_SCORE) / 2


class ThoughtProposerSignature(dspy.Signature):
    """Propose the NEXT distinct reasoning step over the premises.

    You are given the premises (the only source of truth), the question, and the
    reasoning already established. Produce a single next reasoning step that:
      - draws ONLY on information stated in the premises (no outside knowledge,
        no closed-world assumptions),
      - makes explicit forward progress toward answering the question,
      - is self-contained enough to be read on its own,
      - cites the 1-based premise index/indices it relies on.

    Do NOT restate a step already present in `reasoning_so_far`. Write the step
    in Vietnamese to match the task language.
    """

    premises: list[str] = dspy.InputField(desc="The premises; the only source of truth.")
    question: str = dspy.InputField(desc="The question to answer.")
    reasoning_so_far: str = dspy.InputField(
        desc="The reasoning steps established so far, concatenated. Empty at the root."
    )
    next_thought: str = dspy.OutputField(
        desc=(
            "A single next reasoning step in Vietnamese, citing the premise indices it uses "
            "(e.g. 'Theo tiền đề 3, ...')."
        )
    )


class ThoughtEvaluatorSignature(dspy.Signature):
    """Rate how promising a partial reasoning path is.

    Judge how likely `reasoning_path` is to lead to a correct, well-supported
    answer to `question`, using ONLY the `premises`. Penalize: unsound steps not
    entailed by the premises, circularity, no forward progress, reliance on
    outside knowledge, and premature commitment. Reward: sound entailment, clear
    citation of premise indices, and incremental progress.

    Output an integer between 1 (a dead end) and 10 (certain to yield the
    correct, well-justified answer).
    """

    premises: list[str] = dspy.InputField(desc="The premises; the only source of truth.")
    question: str = dspy.InputField(desc="The question to answer.")
    reasoning_path: str = dspy.InputField(
        desc="The reasoning accumulated so far, ending at the thought being scored."
    )
    score: int = dspy.OutputField(desc="Integer promise score from 1 (dead end) to 10 (certain).")


@dataclass
class ThoughtNode:
    """A node in the thought tree. Its state is the full thought `path`."""

    path: list[str]
    score: float
    depth: int


def _render_thought_path(path: Sequence[str]) -> str:
    """Join ordered thoughts into a readable, step-numbered block.

    Uses Vietnamese step markers ("Bước" = "Step") so the recorded reasoning
    matches the task language. Returns an empty string for the (unexplored) root.
    """
    if not path:
        return ""
    return "\n\n".join(f"[Bước {index + 1}] {thought}" for index, thought in enumerate(path))


def _dedupe_thoughts(thoughts: Sequence[str]) -> list[str]:
    """Drop near-duplicate proposals so they do not waste beam slots.

    Normalizes whitespace and case before comparison, but keeps the original
    text of the first occurrence of each distinct thought.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for thought in thoughts:
        key = re.sub(r"\s+", " ", thought.strip().lower())
        if key == "" or key in seen:
            continue
        seen.add(key)
        unique.append(thought)
    return unique


def _parse_score(raw: object) -> float:
    """Parse an evaluator score, clamping to the 1–10 band.

    Falls back to a neutral midpoint when no integer can be extracted, so a
    single malformed judge output does not sink an otherwise-good path.
    """
    match = re.search(r"\d+", str(raw))
    if match is None:
        logger.debug(
            f"Could not parse evaluator score from {raw!r}; defaulting to {_FALLBACK_SCORE}"
        )
        return _FALLBACK_SCORE
    value = int(match.group())
    return float(max(_EVAL_MIN_SCORE, min(_EVAL_MAX_SCORE, value)))


def _mean_score(raw_scores: Sequence[object]) -> float:
    """Average parsed evaluator scores; neutral fallback when none parse."""
    if not raw_scores:
        return _FALLBACK_SCORE
    return sum(_parse_score(score) for score in raw_scores) / len(raw_scores)


def _completion_values(prediction: dspy.Prediction, field: str) -> list[Any]:
    """Collect the sampled values for ``field`` across a prediction's completions.

    DSPy only populates ``prediction.completions`` when ``n`` completions are
    requested via the call ``config``; its declared type is ``None`` otherwise, so
    this accesses it loosely and returns an empty list when it is absent (which
    also lets a failed/sparse proposal or evaluation degrade gracefully instead
    of raising).
    """
    completions = getattr(prediction, "completions", None)
    if completions is None:
        return []
    return list(getattr(completions, field, []))


def _is_better(candidate: ThoughtNode, current: ThoughtNode) -> bool:
    """Whether `candidate` should replace `current` as the best leaf.

    Higher score wins; ties prefer the deeper (more complete) path.
    """
    if candidate.score != current.score:
        return candidate.score > current.score
    return len(candidate.path) > len(current.path)


class ToTStrategy(ReasoningStrategy):
    """Tree of Thoughts: beam-search over reasoning steps, then commit an answer.

    The search is a greedy beam-BFS: at each depth, every frontier node proposes
    ``branching_factor`` next thoughts; each candidate path is scored by the
    evaluator (averaged over ``n_eval_samples`` votes); the ``beam_width`` best
    survive to the next layer. The highest-scoring path seen across all depths is
    fed to the aggregator to produce the final task-formatted answer.

    All knobs default from the ``VIREX_BENCH_TOT_*`` env vars; explicit kwargs
    override them. Cost is roughly ``max_depth * beam_width`` proposer calls and
    ``max_depth * beam_width * branching_factor`` evaluator calls per example —
    far higher than CoT's single call, which is the tradeoff ToT exists to study.
    """

    name = "tot"

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        *,
        max_depth: int | None = None,
        branching_factor: int | None = None,
        beam_width: int | None = None,
        n_eval_samples: int | None = None,
        propose_temperature: float | None = None,
        evaluate_temperature: float | None = None,
        early_stop_threshold: float | None = None,
    ) -> None:
        super().__init__(signature, rationale_field, rationale_field_type)
        self.max_depth = max_depth if max_depth is not None else envs.VIREX_BENCH_TOT_MAX_DEPTH
        self.branching_factor = (
            branching_factor
            if branching_factor is not None
            else envs.VIREX_BENCH_TOT_BRANCHING_FACTOR
        )
        self.beam_width = beam_width if beam_width is not None else envs.VIREX_BENCH_TOT_BEAM_WIDTH
        self.n_eval_samples = (
            n_eval_samples if n_eval_samples is not None else envs.VIREX_BENCH_TOT_EVAL_SAMPLES
        )
        self.propose_temperature = (
            propose_temperature
            if propose_temperature is not None
            else envs.VIREX_BENCH_TOT_PROPOSE_TEMPERATURE
        )
        self.evaluate_temperature = (
            evaluate_temperature
            if evaluate_temperature is not None
            else envs.VIREX_BENCH_TOT_EVALUATE_TEMPERATURE
        )
        self.early_stop_threshold = (
            early_stop_threshold
            if early_stop_threshold is not None
            else envs.VIREX_BENCH_TOT_EARLY_STOP_THRESHOLD
        )

        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {self.branching_factor}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.n_eval_samples < 1:
            raise ValueError(f"n_eval_samples must be >= 1, got {self.n_eval_samples}")

        self.propose = dspy.Predict(ThoughtProposerSignature)
        self.evaluate = dspy.Predict(ThoughtEvaluatorSignature)
        # Feed the explored path into the task signature as an extra input so the
        # committed answer reuses the task's full output contract (answer,
        # supporting_premise_indices, ...). `prepend` is the same mechanism the
        # CoT module uses to add its `reasoning` output field.
        aggregator_signature = self.signature.prepend(
            name="reasoning_path",
            field=dspy.InputField(
                desc=(
                    "A chain of reasoning steps already explored over the premises. "
                    "Use it as the basis for the final answer; do not contradict it."
                )
            ),
            type_=str,
        )
        self.aggregate = dspy.Predict(aggregator_signature)

    def forward(self, **inputs: object) -> dspy.Prediction:
        premises = inputs.get("premises")
        question = inputs.get("question")
        if premises is None or question is None:
            raise ValueError(
                "ToTStrategy requires 'premises' and 'question' inputs, "
                f"got keys: {sorted(inputs)}"
            )

        frontier: list[ThoughtNode] = [ThoughtNode(path=[], score=0.0, depth=0)]
        best_leaf = frontier[0]
        nodes_explored = 0

        for depth in range(self.max_depth):
            candidates: list[ThoughtNode] = []
            for node in frontier:
                proposed = self.propose(
                    premises=premises,
                    question=question,
                    reasoning_so_far=_render_thought_path(node.path),
                    config={"n": self.branching_factor, "temperature": self.propose_temperature},
                )
                thoughts = _dedupe_thoughts(_completion_values(proposed, "next_thought"))
                for thought in thoughts:
                    child_path = [*node.path, thought]
                    evaluated = self.evaluate(
                        premises=premises,
                        question=question,
                        reasoning_path=_render_thought_path(child_path),
                        config={
                            "n": self.n_eval_samples,
                            "temperature": self.evaluate_temperature,
                        },
                    )
                    score = _mean_score(_completion_values(evaluated, "score"))
                    child = ThoughtNode(path=child_path, score=score, depth=depth + 1)
                    candidates.append(child)
                    nodes_explored += 1
                    if _is_better(child, best_leaf):
                        best_leaf = child

            if (
                self.early_stop_threshold is not None
                and best_leaf.score >= self.early_stop_threshold
            ):
                logger.debug(
                    f"ToT early stop at depth {depth + 1}: best score {best_leaf.score:.1f} "
                    f">= threshold {self.early_stop_threshold}"
                )
                break
            if not candidates:
                logger.debug(f"ToT search stalled at depth {depth + 1} (no candidates)")
                break
            candidates.sort(key=lambda node: node.score, reverse=True)
            frontier = candidates[: self.beam_width]

        logger.debug(
            f"ToT search done: explored {nodes_explored} nodes, "
            f"best_leaf depth={best_leaf.depth} score={best_leaf.score:.2f}"
        )

        reasoning_path = _render_thought_path(best_leaf.path)
        prediction = self.aggregate(
            premises=premises,
            question=question,
            reasoning_path=reasoning_path,
        )
        prediction["reasoning"] = reasoning_path
        return prediction

    async def aforward(self, **inputs: object) -> dspy.Prediction:
        premises = inputs.get("premises")
        question = inputs.get("question")
        if premises is None or question is None:
            raise ValueError(
                "ToTStrategy requires 'premises' and 'question' inputs, "
                f"got keys: {sorted(inputs)}"
            )

        frontier: list[ThoughtNode] = [ThoughtNode(path=[], score=0.0, depth=0)]
        best_leaf = frontier[0]
        nodes_explored = 0

        for depth in range(self.max_depth):
            candidates: list[ThoughtNode] = []
            for node in frontier:
                proposed = await self.propose.acall(
                    premises=premises,
                    question=question,
                    reasoning_so_far=_render_thought_path(node.path),
                    config={"n": self.branching_factor, "temperature": self.propose_temperature},
                )
                thoughts = _dedupe_thoughts(_completion_values(proposed, "next_thought"))
                for thought in thoughts:
                    child_path = [*node.path, thought]
                    evaluated = await self.evaluate.acall(
                        premises=premises,
                        question=question,
                        reasoning_path=_render_thought_path(child_path),
                        config={
                            "n": self.n_eval_samples,
                            "temperature": self.evaluate_temperature,
                        },
                    )
                    score = _mean_score(_completion_values(evaluated, "score"))
                    child = ThoughtNode(path=child_path, score=score, depth=depth + 1)
                    candidates.append(child)
                    nodes_explored += 1
                    if _is_better(child, best_leaf):
                        best_leaf = child

            if (
                self.early_stop_threshold is not None
                and best_leaf.score >= self.early_stop_threshold
            ):
                logger.debug(
                    f"ToT early stop at depth {depth + 1}: best score {best_leaf.score:.1f} "
                    f">= threshold {self.early_stop_threshold}"
                )
                break
            if not candidates:
                logger.debug(f"ToT search stalled at depth {depth + 1} (no candidates)")
                break
            candidates.sort(key=lambda node: node.score, reverse=True)
            frontier = candidates[: self.beam_width]

        logger.debug(
            f"ToT search done: explored {nodes_explored} nodes, "
            f"best_leaf depth={best_leaf.depth} score={best_leaf.score:.2f}"
        )

        reasoning_path = _render_thought_path(best_leaf.path)
        prediction = await self.aggregate.acall(
            premises=premises,
            question=question,
            reasoning_path=reasoning_path,
        )
        prediction["reasoning"] = reasoning_path
        return prediction
