"""LLM-as-a-judge framework for reasoning-task evaluation.

Provides a generic :class:`LLMJudge` base class (owning the judge LM, predict
module, and retry logic), a registry for task-specific judge subclasses, and a
fixed DeepSeek reasoning LM builder so that the judge is always independent of the
model under test — avoiding self-judging bias. Task-specific judges (e.g.
:class:`LogicalReasoningJudge`) live alongside their signatures and implement the
``forward`` / normalization logic for their answer types.
"""

import time
from collections.abc import Sequence

import dspy

from virex_bench import envs
from virex_bench.logger import init_logger
from virex_bench.models.base import BaseLM
from virex_bench.strategies.modules import ChainOfThought
from virex_bench.utils import premises_to_text

logger = init_logger(__name__)

LOGIC_ANSWER_ERROR_TYPES: set[str] = {
    "correct",
    "missing_answer",
    "wrong_language",
    "format_error",
    "multiple_choice_error",
    "yes_no_uncertain_error",
    "numeric_error",
    "open_ended_error",
    "incomplete_open_ended",
    "unsupported_extra_claim",
    "conceptual_error",
    "ambiguous_or_unextractable",
    "unknown",
}


class LogicalReasoningJudgeSignature(dspy.Signature):
    """You are an expert in formal and logical reasoning. Judge whether the predicted
    answer is equivalent to the correct answer for a Vietnamese logical-reasoning
    question.

    LANGUAGE REQUIREMENT: This is a Vietnamese-language benchmark, so the predicted
    answer MUST be in Vietnamese — the same language as the gold answer. A semantically
    equivalent answer expressed in a DIFFERENT language (e.g. English "Yes" for
    Vietnamese "Có") is INCORRECT; responding in the correct language is itself part of
    correctness. The ONLY exception is universal symbols: option labels (A-H), Arabic
    numerals, and mathematical notation are language-neutral and need no translation.

    Auto-detect the answer type from the format of the correct answer and apply the
    matching rules below. Use the premises and question ONLY as context to resolve
    ambiguity — DO NOT solve the problem from scratch. The predicted_solution (when
    present) is only used to diagnose error sources; do not penalize a correct answer
    for an incomplete or imperfect explanation.

    TYPE DETECTION (check in order):

    1. MULTIPLE_CHOICE: The correct answer is one or more option labels (A-H). Extract
       the option label(s) the predicted answer actually chose and compare to the
       correct label(s). Labels are universal symbols, so the language of any
       surrounding text does NOT matter — "Đáp án là A", "The answer is A", and "A" are
       all equivalent to correct answer "A". For multi-select, the predicted answer
       must contain exactly the same SET of labels as the correct answer (order,
       separators, and surrounding text are irrelevant).
       correct="C", predicted="C" or "chọn C" => YES.
       correct="A,C", predicted="C và A" => YES (same set).
       correct="A, C", predicted="A,C" => YES (spaces are no matter here).
       correct="A,C", predicted="A" => NO (missing label — incomplete).

    2. YES_NO_UNCERTAIN: The correct answer is a yes/no/uncertain judgement. The
       predicted answer must express the SAME stance AND do so in Vietnamese. An
       English (or other-language) equivalent is INCORRECT even when the stance
       matches. Within Vietnamese, affirmatives are interchangeable ("Có", "Đúng",
       "Vâng"), as are negatives ("Không", "Sai") and uncertainty ("Không xác định",
       "Không rõ"). Capitalization and surrounding text do not matter.
       correct="Có", predicted="Đúng" or "Vâng" => YES (Vietnamese affirmatives).
       correct="Có", predicted="Yes" => NO (wrong language).
       correct="Không", predicted="No" => NO (wrong language).
       correct="Không xác định", predicted="Uncertain" => NO (wrong language).

    3. NUMERIC: The correct answer is a number (possibly with units, thousands
       separators, scientific notation, fractions, or Vietnamese quantity words such
       as "triệu"/"nghìn"). Extract the numeric value (and its unit) from the predicted
       answer and compare it to the gold — surrounding Vietnamese text is irrelevant, so
       the number may be embedded in a full sentence. Allow different formats, equivalent
       units, and rounding to the same precision as the correct answer. Arabic numerals
       are language-neutral; however any quantity words or unit names must be in
       Vietnamese — English words like "million"/"thousand"/"hours" make the answer
       INCORRECT.
       correct="27000000", predicted="2.7 × 10^7" => YES.
       correct="27 triệu", predicted="27000000" => YES (pure numeral, no language).
       correct="27 triệu", predicted="27 million" => NO (English quantity word).
       correct="2 giờ", predicted="Đáp án là 2 giờ" => YES (number embedded in text).
       correct="7", predicted="Nam đã mua được 7 quyển sách" => YES.
       correct="Cần thêm 3 ngày", predicted="3 ngày" => YES.
       correct="Cần thêm 3 ngày", predicted="5" => NO.

    4. OPEN_ENDED: Otherwise, judge semantic equivalence of the logical conclusions.
       Two answers are equivalent if they express the SAME set of logical conclusions
       derived from the premises, even with different wording, ordering, or granularity
       — BUT the predicted answer MUST be in Vietnamese. An answer in any other language
       is INCORRECT regardless of semantic equivalence. Decompose each answer into
       individual claims; the predicted answer is correct only if it contains all the
       key claims of the correct answer, no factually incorrect or unsupported extra
       claims, AND is in Vietnamese. Do NOT penalize paraphrase, reordering, or
       extra-but-true details. Use the question and premises to disambiguate what
       constitutes a valid answer.

    ERROR TYPE (choose the most specific from this list):
    correct, missing_answer, wrong_language, format_error, multiple_choice_error,
    yes_no_uncertain_error, numeric_error, open_ended_error, incomplete_open_ended,
    unsupported_extra_claim, conceptual_error, ambiguous_or_unextractable, unknown.

    Use `wrong_language` when the answer is semantically equivalent to the gold but
    expressed in a non-Vietnamese language. Use `format_error` only when the content
    cannot be extracted at all (e.g. a totally unparseable response); do NOT use it
    merely because surrounding text differs from the gold.

    FEEDBACK: When the answer is correct, leave feedback empty. When incorrect, explain
    the likely issue and the correction in at most 2 sentences. For language errors,
    state the expected language explicitly. Do not invent claims unsupported by the
    inputs.

    Output verdict as exactly "YES" or "NO". Diagnostics go in error_type and feedback.
    """

    premises: str = dspy.InputField(
        desc=(
            "The premises of the problem as numbered text. Use these as the only source "
            "of truth to verify claims in the answers. DO NOT solve the question from "
            "scratch."
        )
    )
    question: str = dspy.InputField(
        desc=(
            "The original logical-reasoning question. Use it as context to resolve "
            "ambiguity (e.g. which conclusions an open-ended answer is asking for). "
            "DO NOT solve the question from scratch."
        )
    )
    correct_answer: str = dspy.InputField(
        desc="The gold answer (option label(s), a yes/no/uncertain stance, a number, or free text)."
    )
    predicted_answer: str = dspy.InputField(
        desc="The answer predicted by the model under evaluation."
    )
    predicted_solution: str = dspy.InputField(
        desc=(
            "The model's explanation/reasoning, if available. Use it only to diagnose "
            "the error source; the verdict must still be based on answer equivalence."
        )
    )
    verdict: str = dspy.OutputField(
        desc="'YES' if the predicted answer is equivalent to the correct answer, 'NO' otherwise."
    )
    error_type: str = dspy.OutputField(
        desc="Exactly one diagnostic label from the allowed error type list."
    )
    feedback: str = dspy.OutputField(
        desc="Concise actionable feedback, or empty when the answer is correct."
    )


def build_judge_lm() -> BaseLM:
    """Build the fixed DeepSeek reasoning LM used by the judge.

    The model (``deepseek-v4-flash`` by default, or ``deepseek-v4-pro``) is chosen via
    the ``VIREX_BENCH_JUDGE_MODEL`` env var; the API key via
    ``VIREX_BENCH_JUDGE_API_KEY``.
    """
    judge_model_name = envs.VIREX_BENCH_JUDGE_MODEL
    api_key = envs.VIREX_BENCH_JUDGE_API_KEY
    if api_key is None:
        raise RuntimeError(
            "VIREX_BENCH_JUDGE_API_KEY is not set. The llm_judge metric requires "
            f"a judge model API key (current judge model set to `{judge_model_name}`). "
            "Set it via the VIREX_BENCH_JUDGE_API_KEY environment variable. "
            "If the provider does not require an API key, set it to any non-empty string, "
            "e.g., `export VIREX_BENCH_JUDGE_API_KEY=empty`."
        )
    if judge_model_name.lower().startswith("deepseek/"):
        # DeepSeek API
        return BaseLM(
            model=f"{judge_model_name}",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            extra_body={
                "reasoning_effort": "high",
                "thinking": {"type": "enabled"},
            },
            cache=False,
        )
    elif judge_model_name.lower().startswith("opencode-go/"):
        # OpenCode Go
        return BaseLM(
            model=f"openai/{judge_model_name.removeprefix('opencode-go/')}",
            base_url="https://opencode.ai/zen/go/v1",
            api_key=api_key,
            extra_body={
                "reasoning_effort": "high",
                "thinking": {"type": "enabled"},
            },
            cache=False,
        )
    else:
        raise NotImplementedError(f"Judge model {judge_model_name} is not supported.")


class LLMJudge(dspy.Module):
    """Base class for LLM-as-a-judge modules.

    The base class is fully agnostic to a task's input and output shape: it owns
    only the generic machinery shared by every judge — the judge language model,
    the ``dspy.Predict`` module, and a retry wrapper around it. Subclasses
    implement :meth:`_build_predict` (their signature) and :meth:`forward`
    (mapping task-specific inputs to that signature and normalizing its outputs).
    """

    def __init__(
        self,
        judge_lm: BaseLM | None = None,
        max_attempts: int = 3,
        retry_wait_seconds: float = 0.5,
    ) -> None:
        super().__init__()
        self.judge_lm = judge_lm if judge_lm is not None else build_judge_lm()
        self.max_attempts = max_attempts
        self.retry_wait_seconds = retry_wait_seconds
        self.predict = self._build_predict_module()

    def _build_predict_module(self) -> dspy.Module:
        raise NotImplementedError

    def _judge(self, **inputs: object) -> dspy.Prediction:
        """Run the predict module under the judge LM, retrying on failure."""
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with dspy.context(lm=self.judge_lm):
                    return self.predict(**inputs)
            except Exception as error:  # noqa: BLE001
                last_error = error
                logger.debug(f"Judge attempt {attempt + 1}/{self.max_attempts} failed: {error!r}")
                if attempt < self.max_attempts - 1:
                    time.sleep(self.retry_wait_seconds)

        raise RuntimeError(
            f"Judge failed after {self.max_attempts} attempts: {last_error!r}"
        ) from last_error


class LogicalReasoningJudge(LLMJudge):
    """LLM judge for Vietnamese logical-reasoning answers.

    Handles four answer types, auto-detected from the gold answer's format:
    multiple-choice, yes/no/uncertain, numeric, and open-ended. A fixed DeepSeek
    reasoning model performs the judgement, so the judge is independent of the model
    under test.
    """

    error_types = LOGIC_ANSWER_ERROR_TYPES

    def _build_predict_module(self) -> dspy.Module:
        return ChainOfThought(
            signature=LogicalReasoningJudgeSignature,
            rationale_field=dspy.OutputField(desc="The reasoning behind the verdict."),
        )

    def forward(
        self,
        premises: Sequence[str],
        question: str,
        correct_answer: str,
        predicted_answer: str,
        predicted_solution: str = "",
    ) -> dspy.Prediction:
        if predicted_answer.strip() == "":
            return dspy.Prediction(
                verdict="NO",
                error_type="missing_answer",
                feedback="No usable answer was provided.",
                reasoning="",
            )
        result = self._judge(
            premises=premises_to_text(premises),
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            predicted_solution=predicted_solution,
        )
        return self._normalize_result(result)

    def _normalize_result(self, result: dspy.Prediction) -> dspy.Prediction:
        verdict = str(getattr(result, "verdict", "")).strip().upper()
        if verdict not in {"YES", "NO"}:
            logger.warning(
                f"Unexpected verdict '{result.verdict}' from judge. Defaulting to 'NO'."
            )
            verdict = "NO"

        error_type = str(getattr(result, "error_type", "")).strip().lower().replace("-", "_")
        if error_type not in self.error_types:
            logger.warning(
                f"Unexpected error_type '{error_type}' from judge. Defaulting to 'unknown'."
            )
            error_type = "unknown"

        if verdict == "YES":
            error_type = "correct"
        elif error_type == "correct":
            error_type = "unknown"

        feedback = str(getattr(result, "feedback", "")).strip()
        reasoning = str(getattr(result, "reasoning", "") or "")
        return dspy.Prediction(
            verdict=verdict,
            error_type=error_type,
            feedback=feedback,
            reasoning=reasoning,
        )


JUDGE_REGISTRY: dict[str, type[LLMJudge]] = {
    "logical_reasoning": LogicalReasoningJudge,
}


def build_judge(name: str) -> LLMJudge:
    """Instantiate a judge by name from the registry."""
    try:
        judge_cls = JUDGE_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(JUDGE_REGISTRY))
        raise KeyError(f"Unknown judge '{name}'. Available judges: {available}") from None
    return judge_cls()


def list_judges() -> list[str]:
    """Return the sorted names of all registered judges."""
    return sorted(JUDGE_REGISTRY)
