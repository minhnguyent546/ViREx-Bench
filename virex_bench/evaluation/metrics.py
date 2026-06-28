"""Scoring metrics for reasoning tasks.

Each metric is a callable ``metric(example, prediction) -> float`` returning a
score in ``[0.0, 1.0]`` (``1.0`` means a perfect match). Metrics are registered
by name in :data:`METRIC_REGISTRY` so that tasks and the CLI can refer to them
by string. This combines how ``dspy.Evaluate`` accepts a pluggable ``metric``
callable with how ``mteb`` surfaces a named ``main_score`` per task that is
resolved against a metric implementation.
"""

import difflib
import json
import re
import string
import unicodedata
from collections import Counter
from typing import Any

import dspy

from virex_bench.evaluation.judge import LLMJudge
from virex_bench.types import JudgeOutcome, ReasoningExample, ReasoningMetric

# Minimum difflib SequenceMatcher ratio to accept a fuzzy premise-text rematch when
# falling back from the primary 1-based indices to the relevant_premises text field.
MINIMUM_PREMISE_MATCH_SCORE = 0.80


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


_PREMISE_PREFIX_PATTERN = re.compile(r"^\s*(premise\s*)?\d+[:.)]?\s*", re.IGNORECASE)


def _parse_premise_indices(value: object) -> list[int]:
    """Best-effort parse of a model-emitted premise-index field into a list of ints.

    Handles the typed ``list[int]`` case as well as JSON-array strings and loose
    comma/space separated digit strings produced by less compliant models.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items: list[Any] = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        try:
            parsed = json.loads(text)
            raw_items = parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, TypeError):
            raw_items = list(re.findall(r"-?\d+", text))
    else:
        raw_items = [value]
    indices: list[int] = []
    for item in raw_items:
        try:
            indices.append(int(item))
        except (TypeError, ValueError):
            continue
    return indices


def _parse_premise_texts(value: object) -> list[str]:
    """Best-effort parse of a model-emitted premise-text field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return [text]
        return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]
    return [str(value)]


def _normalize_premise_text(text: str) -> str:
    """Strip any leading 'Premise N:' marker and normalize for fuzzy comparison."""
    return _PREMISE_PREFIX_PATTERN.sub("", text.strip()).strip().lower()


def _rematch_premise_indices(premises: list[str], texts: list[str]) -> list[int]:
    """Recover 0-based premise indices by fuzzy-matching free text to the premise list.

    Used as a fallback when the model's primary ``supporting_premise_indices`` field
    is missing or unusable. Each text is matched to its most similar premise; the
    match is accepted only above :data:`MINIMUM_PREMISE_MATCH_SCORE`.
    """
    normalized_premises = [_normalize_premise_text(premise) for premise in premises]
    indices: set[int] = set()
    for text in texts:
        normalized_text = _normalize_premise_text(text)
        if normalized_text == "":
            continue
        best_index = -1
        best_score = 0.0
        for position, premise in enumerate(normalized_premises):
            if premise == "":
                continue
            score = difflib.SequenceMatcher(None, normalized_text, premise).ratio()
            if score > best_score:
                best_score = score
                best_index = position
        if best_index != -1 and best_score >= MINIMUM_PREMISE_MATCH_SCORE:
            indices.add(best_index)
    return sorted(indices)


def predicted_premise_indices(example: ReasoningExample, prediction: dspy.Prediction) -> set[int]:
    """Resolve the set of 0-based premise indices the model claims to have used.

    The model emits 1-based ``supporting_premise_indices`` (validated against the
    premise count); they are converted to 0-based here so the rest of the pipeline
    is uniformly 0-based, matching the dataset's gold ``premises_used``. Only when
    the primary field yields no in-range index do we fall back to fuzzy-matching the
    ``relevant_premises`` text against the original premises.
    """
    num_premises = len(example.premises)
    primary = _parse_premise_indices(getattr(prediction, "supporting_premise_indices", None))
    valid = {index - 1 for index in primary if 1 <= index <= num_premises}
    if valid:
        return valid
    texts = _parse_premise_texts(getattr(prediction, "relevant_premises", None))
    return set(_rematch_premise_indices(example.premises, texts))


def premises_f1(example: ReasoningExample, prediction: dspy.Prediction) -> float:
    """F1 between the model's predicted premises-used and the gold ``premises_used`` set."""
    gold = {int(index) for index in example.premises_used}
    predicted = predicted_premise_indices(example, prediction)
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    overlap = len(predicted & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return float((2 * precision * recall) / (precision + recall))


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
    "premises_f1": premises_f1,
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
