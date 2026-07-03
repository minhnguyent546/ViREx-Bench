"""Shared signatures + helpers for the ToT search algorithms.

Moved verbatim from the original single-file ``strategies/tot.py`` so every search
algorithm (beam, DFS, MCTS) reuses the same proposer/evaluator contracts and
utility functions.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import dspy

from virex_bench import envs
from virex_bench.logger import init_logger

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

    Be concise: produce at most three short sentences. Do not include hidden
    thinking traces or alternative branches.

    Do NOT restate a step already present in `reasoning_so_far`. Write the step
    in Vietnamese to match the task language.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block (Premise 1, Premise 2, ...); the only source of truth."
    )
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
    """Rate how close a partial reasoning path is to a correct, final answer.

    Judge `reasoning_path` against `question` using ONLY the `premises`. Be
    STRICT and discriminating -- do not hand out high scores generously.

    Hard penalties (cap the score at 4 regardless of other strengths):
      - any step not entailed by the premises, or relying on outside knowledge;
      - circular reasoning, or a step that merely restates an earlier step;
      - no forward progress toward answering the question;
      - a missing or incorrect premise citation.

    Score bands:
       1-3  Dead end / mostly unsound or stalled.
       4-6  On track but incomplete; the answer is NOT yet derived.
       7-8  Nearly there; one or two sound steps short of the answer.
       9-10 RESERVED for a path that has ALREADY derived a correct, complete,
            well-supported answer. Reserve 10 for an airtight, fully-cited
            derivation. If any step remains to be done, the score must be <= 8.

    Output an integer in 1..10.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block (Premise 1, Premise 2, ...); the only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer.")
    reasoning_path: str = dspy.InputField(
        desc="The reasoning accumulated so far, ending at the thought being scored."
    )
    score: int = dspy.OutputField(
        desc=(
            "Integer in 1..10. Use 9-10 ONLY if the path already derives the final "
            "answer; cap at 8 if any reasoning remains."
        )
    )


@dataclass
class ThoughtNode:
    """A node in the thought tree. Its state is the full thought `path`."""

    path: list[str]
    score: float
    depth: int


def render_thought_path(path: Sequence[str]) -> str:
    """Join ordered thoughts into a readable, step-numbered block.

    Uses Vietnamese step markers ("Bước" = "Step") so the recorded reasoning
    matches the task language. Returns an empty string for the (unexplored) root.
    """
    if not path:
        return ""
    return "\n\n".join(f"[Bước {index + 1}] {thought}" for index, thought in enumerate(path))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Word-set Jaccard similarity: |intersection| / |union|."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dedupe_thoughts(
    thoughts: Sequence[str],
    *,
    similarity_threshold: float | None = None,
) -> list[str]:
    """Drop near-duplicate proposals so they do not waste beam slots.

    Normalizes case, tokenizes into word sets, then compares each thought
    against already-accepted ones via :func:`jaccard_similarity`.  A candidate
    is dropped when its similarity to any accepted thought meets or exceeds
    ``similarity_threshold``.  Exact duplicates (identical word sets,
    Jaccard = 1.0) are always caught.  The original text of the first
    occurrence of each distinct thought is kept.
    """
    threshold = (
        envs.VIREX_BENCH_TOT_DEDUPE_SIMILARITY_THRESHOLD
        if similarity_threshold is None
        else similarity_threshold
    )
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"similarity_threshold must be between 0 and 1, got {threshold}")

    accepted_token_sets: list[set[str]] = []
    unique: list[str] = []
    for thought in thoughts:
        tokens = set(thought.strip().lower().split())
        if not tokens:
            continue
        if any(
            jaccard_similarity(tokens, existing) >= threshold for existing in accepted_token_sets
        ):
            continue
        accepted_token_sets.append(tokens)
        unique.append(thought)
    return unique


def _parse_score(raw: object) -> float:
    """Parse an evaluator score, clamping to the 1-10 band.

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


def mean_score(raw_scores: Sequence[object]) -> float:
    """Average parsed evaluator scores; neutral fallback when none parse."""
    if not raw_scores:
        return _FALLBACK_SCORE
    return sum(_parse_score(score) for score in raw_scores) / len(raw_scores)


def completion_values(prediction: dspy.Prediction, field: str) -> list[Any]:
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


def is_better(candidate: ThoughtNode, current: ThoughtNode) -> bool:
    """Whether `candidate` should replace `current` as the best leaf.

    Higher score wins; ties prefer the SHORTER path -- among equally-promising
    derivations the more concise one wins (parsimony), which also keeps a
    score-saturating evaluator from inflating the winner with redundant filler
    steps.
    """
    if candidate.score != current.score:
        return candidate.score > current.score
    return len(candidate.path) < len(current.path)
