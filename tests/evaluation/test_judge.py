from typing import Any

import dspy
import pytest

from virex_bench.evaluation import judge as judge_module
from virex_bench.evaluation.judge import (
    LogicalReasoningJudge,
    build_judge,
    compare_closed_answer,
    list_judges,
)
from virex_bench.models.base import BaseLM


def test_build_judge_lm_uses_generic_judge_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _FakeBaseLM:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.delenv("VIREX_BENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("VIREX_BENCH_JUDGE_API_KEY", "test-judge-key")
    monkeypatch.setattr(judge_module, "BaseLM", _FakeBaseLM)

    judge_module.build_judge_lm()

    assert captured_kwargs["model"] == "deepseek/deepseek-v4-flash"
    assert captured_kwargs["api_key"] == "test-judge-key"


def test_build_judge_lm_requires_generic_judge_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIREX_BENCH_JUDGE_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.delenv("VIREX_BENCH_JUDGE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="VIREX_BENCH_JUDGE_API_KEY"):
        judge_module.build_judge_lm()


def _closed(category: str | None, correct: str, predicted: str) -> str | None:
    result = compare_closed_answer(category, correct, predicted)
    if result is None:
        return None
    return str(result.verdict)


# MCQ: predicted must be option labels only — extra text/punctuation is wrong.


def test_compare_mcq_single_label_match() -> None:
    assert _closed("mcq", "A", "A") == "YES"
    assert _closed("mcq", "C", "C") == "YES"


def test_compare_mcq_case_insensitive_label() -> None:
    assert _closed("mcq", "A", "a") == "YES"


def test_compare_mcq_multi_label_set_equality_ignores_order() -> None:
    assert _closed("mcq", "B, C", "B, C") == "YES"
    assert _closed("mcq", "A, C", "C, A") == "YES"


def test_compare_mcq_comma_without_space_still_accepted() -> None:
    # The separator allows optional whitespace; "A,C" is labels-only.
    assert _closed("mcq", "A, C", "A,C") == "YES"


def test_compare_mcq_missing_label_is_wrong() -> None:
    assert _closed("mcq", "A, C", "A") == "NO"


def test_compare_mcq_extra_label_is_wrong() -> None:
    assert _closed("mcq", "A", "A, B") == "NO"


def test_compare_mcq_surrounding_text_is_wrong() -> None:
    # The label is correct but extra text makes it a format violation.
    assert _closed("mcq", "C", "Đáp án là C") == "NO"
    assert _closed("mcq", "A, C", "C và A") == "NO"
    assert _closed("mcq", "B", "chọn B") == "NO"


def test_compare_mcq_trailing_punctuation_is_wrong() -> None:
    assert _closed("mcq", "C", "C.") == "NO"
    assert _closed("mcq", "C", "C,") == "NO"


def test_compare_mcq_predicted_not_labels_is_wrong() -> None:
    assert _closed("mcq", "A", "Không chắc chắn") == "NO"


def test_compare_mcq_gold_without_labels_falls_back_to_llm() -> None:
    # Dataset anomaly: an mcq row whose gold is not labels — defer to the LM
    # judge so both "A" (wrong) and a matching "Không chắc chắn" (right) score
    # correctly. Checked before the predicted-format rule, so a model that
    # matches an anomalous gold is not falsely rejected.
    assert _closed("mcq", "Không chắc chắn", "A") is None
    assert _closed("mcq", "Không chắc chắn", "Không chắc chắn") is None


# ynu: predicted must be EXACTLY one of the three canonical labels — no
# variants, no other languages, no LM fallback on the predicted side.


def test_compare_ynu_exact_match() -> None:
    assert _closed("yes_no_uncertain", "Có", "Có") == "YES"
    assert _closed("yes_no_uncertain", "Không", "Không") == "YES"
    assert _closed("yes_no_uncertain", "Không chắc chắn", "Không chắc chắn") == "YES"


def test_compare_ynu_case_and_whitespace_insensitive() -> None:
    assert _closed("yes_no_uncertain", "Có", "có") == "YES"
    assert _closed("yes_no_uncertain", "Không chắc chắn", "không  chắc chắn") == "YES"


def test_compare_ynu_variant_affirmatives_are_wrong() -> None:
    # "Đúng"/"Vâng" are semantically equivalent but not the canonical label.
    assert _closed("yes_no_uncertain", "Có", "Đúng") == "NO"
    assert _closed("yes_no_uncertain", "Có", "Vâng") == "NO"


def test_compare_ynu_variant_negatives_are_wrong() -> None:
    assert _closed("yes_no_uncertain", "Không", "Sai") == "NO"


def test_compare_ynu_variant_uncertainty_is_wrong() -> None:
    assert _closed("yes_no_uncertain", "Không chắc chắn", "Không rõ") == "NO"
    assert _closed("yes_no_uncertain", "Không chắc chắn", "Không xác định") == "NO"


def test_compare_ynu_other_language_is_wrong_no_fallback() -> None:
    # "Yes" is semantically equivalent but wrong language — strict mode marks it
    # wrong directly rather than falling back to the LM judge.
    assert _closed("yes_no_uncertain", "Có", "Yes") == "NO"
    assert _closed("yes_no_uncertain", "Không", "No") == "NO"
    assert _closed("yes_no_uncertain", "Không chắc chắn", "Uncertain") == "NO"


def test_compare_ynu_stance_mismatch_is_wrong() -> None:
    assert _closed("yes_no_uncertain", "Có", "Không") == "NO"
    assert _closed("yes_no_uncertain", "Không", "Có") == "NO"


def test_compare_ynu_uncertainty_not_confused_with_negative() -> None:
    # "Không chắc chắn" must be treated as the uncertain label, not as the
    # negative "không" — so it matches an uncertain gold and mismatches a no.
    assert _closed("yes_no_uncertain", "Không chắc chắn", "Không chắc chắn") == "YES"
    assert _closed("yes_no_uncertain", "Không", "Không chắc chắn") == "NO"


def test_compare_ynu_extra_text_is_wrong() -> None:
    assert _closed("yes_no_uncertain", "Có", "Đáp án là Có") == "NO"
    assert _closed("yes_no_uncertain", "Không", "Không.") == "NO"


def test_compare_ynu_gold_not_a_stance_falls_back_to_llm() -> None:
    # Safety net: a mislabeled ynu row whose gold is not a stance defers to LM.
    assert _closed("yes_no_uncertain", "A", "Có") is None


# Non-closed categories always defer to the LM judge.


def test_compare_closed_unknown_category_falls_back_to_llm() -> None:
    assert _closed("number", "27 triệu", "27000000") is None
    assert _closed("text", "some answer", "some answer") is None
    assert _closed(None, "A", "A") is None


def test_compare_closed_result_carries_method_and_error_type() -> None:
    wrong = compare_closed_answer("mcq", "A", "B")
    assert wrong is not None
    assert wrong.method == "deterministic"
    assert wrong.error_type == "multiple_choice_error"

    ok = compare_closed_answer("yes_no_uncertain", "Có", "Có")
    assert ok is not None
    assert ok.method == "deterministic"
    assert ok.error_type == "correct"

    format_error = compare_closed_answer("mcq", "A", "Đáp án là A")
    assert format_error is not None
    assert format_error.method == "deterministic"
    assert format_error.error_type == "multiple_choice_error"


def _make_judge() -> LogicalReasoningJudge:
    """Build a judge with a stub judge LM so no network/config is needed."""
    return LogicalReasoningJudge(judge_lm=BaseLM(model="openai/test-judge", api_key="k"))


def test_build_judge_lm_opencode_go_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _FakeBaseLM:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setenv("VIREX_BENCH_JUDGE_MODEL", "opencode-go/deepseek-v4-flash")
    monkeypatch.setenv("VIREX_BENCH_JUDGE_API_KEY", "go-key")
    monkeypatch.setattr(judge_module, "BaseLM", _FakeBaseLM)

    judge_module.build_judge_lm()

    assert captured_kwargs["model"] == "openai/deepseek-v4-flash"
    assert captured_kwargs["base_url"] == "https://opencode.ai/zen/go/v1"
    assert captured_kwargs["api_key"] == "go-key"


def test_build_judge_lm_unknown_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIREX_BENCH_JUDGE_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setenv("VIREX_BENCH_JUDGE_API_KEY", "k")

    class _FakeBaseLM:
        def __init__(self, **kwargs: Any) -> None:
            raise NotImplementedError("unsupported model")

    monkeypatch.setattr(judge_module, "BaseLM", _FakeBaseLM)
    with pytest.raises(NotImplementedError, match="unsupported model"):
        judge_module.build_judge_lm()


def test_build_judge_returns_logical_reasoning_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        judge_module, "build_judge_lm", lambda: BaseLM(model="openai/stub", api_key="k")
    )
    judge = build_judge("logical_reasoning")
    assert isinstance(judge, LogicalReasoningJudge)


def test_build_judge_unknown_name_raises_with_available(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(KeyError, match="Unknown judge 'nope'"):
        build_judge("nope")


def test_list_judges_returns_sorted_registered_names() -> None:
    assert list_judges() == ["logical_reasoning"]


def test_normalize_result_yes_forces_correct_error_type() -> None:
    judge = _make_judge()
    result = judge._normalize_result(  # pyright: ignore[reportPrivateUsage]
        dspy.Prediction(verdict="YES", error_type="wrong_language", feedback="", reasoning="r")
    )
    assert result.verdict == "YES"
    assert result.error_type == "correct"
    assert result.method == "llm"


def test_normalize_result_no_with_correct_becomes_unknown() -> None:
    judge = _make_judge()
    result = judge._normalize_result(  # pyright: ignore[reportPrivateUsage]
        dspy.Prediction(verdict="NO", error_type="correct", feedback="f", reasoning="r")
    )
    assert result.verdict == "NO"
    assert result.error_type == "unknown"


def test_normalize_result_no_with_specific_error_keeps_it() -> None:
    judge = _make_judge()
    result = judge._normalize_result(  # pyright: ignore[reportPrivateUsage]
        dspy.Prediction(
            verdict="NO", error_type="multiple_choice_error", feedback="f", reasoning="r"
        )
    )
    assert result.verdict == "NO"
    assert result.error_type == "multiple_choice_error"


def test_normalize_result_unknown_verdict_defaults_to_no_and_unknown_error() -> None:
    judge = _make_judge()
    result = judge._normalize_result(  # pyright: ignore[reportPrivateUsage]
        dspy.Prediction(verdict="MAYBE", error_type="bogus", feedback="f", reasoning="r")
    )
    assert result.verdict == "NO"
    assert result.error_type == "unknown"


def test_normalize_result_lowercases_and_underscores_error_type() -> None:
    judge = _make_judge()
    result = judge._normalize_result(  # pyright: ignore[reportPrivateUsage]
        dspy.Prediction(verdict="NO", error_type="Open-Ended-Error", feedback="f", reasoning="r")
    )
    assert result.error_type == "open_ended_error"


class _FakePredict:
    """Stand-in for the judge's predict module: returns canned verdicts in sequence."""

    def __init__(self, predictions: list[dspy.Prediction]) -> None:
        self._predictions = list(predictions)
        self._index = 0
        self.call_count = 0

    def __call__(self, **_kwargs: object) -> dspy.Prediction:
        prediction = self._predictions[self._index]
        self._index += 1
        self.call_count += 1
        return prediction


class _RetryPredict:
    """Raises on the first N calls, then returns a canned prediction."""

    def __init__(self, fail_times: int, prediction: dspy.Prediction) -> None:
        self._fail_times = fail_times
        self._prediction = prediction
        self.call_count = 0

    def __call__(self, **_kwargs: object) -> dspy.Prediction:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise RuntimeError("transient judge failure")
        return self._prediction


_PREMISES = ["Nếu trời mưa thì đường trơn.", "Đường đang trơn."]


def test_forward_empty_predicted_answer_is_missing_answer_deterministic() -> None:
    judge = _make_judge()
    result = judge.forward(
        premises=_PREMISES,
        question="Đường có trơn không?",
        correct_answer="Có",
        predicted_answer="   ",
        category="yes_no_uncertain",
    )
    assert result.verdict == "NO"
    assert result.error_type == "missing_answer"
    assert result.method == "deterministic"


def test_forward_closed_match_skips_llm() -> None:
    judge = _make_judge()
    judge.predict = _FakePredict([])  # type: ignore[assignment]
    result = judge.forward(
        premises=_PREMISES,
        question="Chọn đáp án đúng.",
        correct_answer="A",
        predicted_answer="A",
        category="mcq",
    )
    assert result.verdict == "YES"
    assert result.method == "deterministic"
    assert judge.predict.call_count == 0  # type: ignore[attr-defined]


def test_forward_open_ended_invokes_llm_and_normalizes() -> None:
    judge = _make_judge()
    judge.predict = _FakePredict(  # type: ignore[assignment]
        [dspy.Prediction(verdict="YES", error_type="correct", feedback="", reasoning="r")]
    )
    result = judge.forward(
        premises=_PREMISES,
        question="Suy luận gì từ các tiền đề?",
        correct_answer="Đường trơn vì trời mưa.",
        predicted_answer="Đường trơn do trời mưa.",
        category="open_ended",
    )
    assert result.verdict == "YES"
    assert result.method == "llm"
    assert judge.predict.call_count == 1  # type: ignore[attr-defined]


def test_judge_retries_then_succeeds() -> None:
    judge = LogicalReasoningJudge(
        judge_lm=BaseLM(model="openai/test-judge", api_key="k"),
        max_attempts=3,
        retry_wait_seconds=0.0,
    )
    judge.predict = _RetryPredict(  # type: ignore[assignment]
        fail_times=2,
        prediction=dspy.Prediction(
            verdict="YES", error_type="correct", feedback="", reasoning="r"
        ),
    )
    outcome = judge._judge(  # pyright: ignore[reportPrivateUsage]
        premises="p",
        question="q",
        correct_answer="A",
        predicted_answer="A",
        predicted_solution="",
    )
    assert outcome.verdict == "YES"
    assert judge.predict.call_count == 3  # type: ignore[attr-defined]


def test_judge_raises_after_exhausting_attempts() -> None:
    judge = LogicalReasoningJudge(
        judge_lm=BaseLM(model="openai/test-judge", api_key="k"),
        max_attempts=2,
        retry_wait_seconds=0.0,
    )
    judge.predict = _RetryPredict(  # type: ignore[assignment]
        fail_times=10,
        prediction=dspy.Prediction(
            verdict="YES", error_type="correct", feedback="", reasoning="r"
        ),
    )
    with pytest.raises(RuntimeError, match="Judge failed after 2 attempts"):
        judge._judge(  # pyright: ignore[reportPrivateUsage]
            premises="p",
            question="q",
            correct_answer="A",
            predicted_answer="A",
            predicted_solution="",
        )
    assert judge.predict.call_count == 2  # type: ignore[attr-defined]
