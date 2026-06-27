"""Scoring metrics for reasoning tasks.

Each metric is a callable ``metric(example, prediction) -> float`` returning a
score in ``[0.0, 1.0]`` (``1.0`` means a perfect match). Metrics are registered
by name in :data:`METRIC_REGISTRY` so that tasks and the CLI can refer to them
by string. This combines how ``dspy.Evaluate`` accepts a pluggable ``metric``
callable with how ``mteb`` surfaces a named ``main_score`` per task that is
resolved against a metric implementation.
"""

import string
import unicodedata
from collections import Counter
from collections.abc import Callable

import dspy
from pydantic import BaseModel

from virex_bench.evaluation.judge import LLMJudge
from virex_bench.tasks.base import ReasoningExample

ReasoningMetric = Callable[[ReasoningExample, dspy.Prediction], float]


def _normalize(text: str) -> str:
    """Light normalization: trim, lowercase, drop trailing periods."""
    return text.strip().lower().rstrip(".")


def _normalize_tokens(text: str) -> list[str]:
    """Aggressive normalization for token-level comparison.

    Applies Unicode NFD decomposition, lowercasing, punctuation removal and
    whitespace tokenization.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_punctuation = decomposed.translate(str.maketrans("", "", string.punctuation))
    return without_punctuation.lower().split()


def exact_match(example: ReasoningExample, prediction: dspy.Prediction) -> float:
    """Return ``1.0`` if the predicted answer exactly matches the gold answer after normalization, else ``0.0``."""
    return float(_normalize(str(prediction.answer)) == _normalize(example.answer))


def token_f1(example: ReasoningExample, prediction: dspy.Prediction) -> float:
    """Token-level F1 between the predicted and gold answers after aggressive normalization."""
    prediction_tokens = _normalize_tokens(str(prediction.answer))
    gold_tokens = _normalize_tokens(example.answer)
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return float((2 * precision * recall) / (precision + recall))


def contains(example: ReasoningExample, prediction: dspy.Prediction) -> float:
    """Return ``1.0`` if the normalized gold answer appears anywhere within the normalized prediction, else ``0.0``."""
    return float(_normalize(example.answer) in _normalize(str(prediction.answer)))


class JudgeOutcome(BaseModel):
    """Full outcome of judging a single example with an LLM-as-a-judge."""

    verdict: str
    error_type: str
    feedback: str
    score: float


def judge_example(
    example: ReasoningExample,
    prediction: dspy.Prediction,
    judge_module: LLMJudge,
) -> JudgeOutcome:
    """Run the judge on a single example and return the full outcome.

    This is the single source of truth for invoking the judge: both the
    :func:`llm_judge` metric and the evaluator call this so the result is
    never duplicated.
    """
    judgement = judge_module(
        premises=example.premises,
        question=example.question,
        correct_answer=example.answer,
        predicted_answer=str(prediction.answer),
        predicted_solution=str(getattr(prediction, "reasoning", "") or ""),
    )
    verdict = str(judgement.verdict).strip().upper()
    return JudgeOutcome(
        verdict=verdict,
        error_type=str(judgement.error_type),
        feedback=str(getattr(judgement, "feedback", "") or ""),
        score=float(verdict.startswith("YES")),
    )


def llm_judge(
    example: ReasoningExample,
    prediction: dspy.Prediction,
    *,
    judge: LLMJudge,
) -> float:
    """Score answer correctness with an LLM-as-a-judge across all answer types.

    The ``judge`` is a task-provided :class:`~virex_bench.evaluation.judge.LLMJudge`
    (e.g. ``LogicalReasoningJudge``), bound by the evaluator from the task's
    ``judge`` metadata. Returns ``1.0`` when the judge verdicts the prediction
    equivalent to the gold answer, else ``0.0``.
    """
    return judge_example(example, prediction, judge).score


METRIC_REGISTRY: dict[str, ReasoningMetric] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "contains": contains,
}


def get_metric(name: str) -> ReasoningMetric:
    """Resolve a metric name to its callable, raising ``KeyError`` if unknown."""
    try:
        return METRIC_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(METRIC_REGISTRY))
        raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}") from None


def list_metrics() -> list[str]:
    """Return the sorted names of all registered metrics."""
    return sorted(METRIC_REGISTRY)
