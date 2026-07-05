from typing import Any

import pytest

from virex_bench.evaluation import judge as judge_module
from virex_bench.evaluation.judge import compare_closed_answer


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


# --- compare_closed_answer (deterministic strict fast path) -----------------


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
