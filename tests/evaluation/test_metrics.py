# pyright: reportPrivateUsage=false

import dspy
import pytest

from virex_bench.evaluation.metrics import (
    _debug_value,
    _parse_premise_indices,
    _parse_premise_texts,
    _parse_reasoning_premise_references,
    _rematch_premise_indices,
    contains,
    exact_match,
    get_metric,
    judge_example,
    list_metrics,
    llm_judge,
    premise_selection_debug,
    premises_f1,
    token_f1,
)
from virex_bench.types import ReasoningExample


def test_premise_selection_debug_records_primary_and_reasoning_references() -> None:
    example = ReasoningExample(
        example_id="example-1",
        premises=["alpha", "beta", "gamma"],
        question="question",
        answer="answer",
        premises_used=[0, 2],
    )
    prediction = dspy.Prediction(
        answer="answer",
        reasoning="Theo ti\u1ec1n \u0111\u1ec1 1 va Premise 3.",
        supporting_premise_indices="[1, 3, 99]",
        relevant_premises=["beta"],
    )

    debug = premise_selection_debug(example, prediction)

    assert debug["raw_supporting_premise_indices"] == "[1, 3, 99]"
    assert debug["parsed_supporting_premise_indices_1_based"] == [1, 3, 99]
    assert debug["valid_supporting_premise_indices_0_based"] == [0, 2]
    assert debug["raw_relevant_premises"] == ["beta"]
    assert debug["parsed_relevant_premises"] == ["beta"]
    assert debug["rematched_relevant_premises_0_based"] == [1]
    assert debug["selected_source"] == "supporting_premise_indices"
    assert debug["selected_premises_used_0_based"] == [0, 2]
    assert debug["reasoning_referenced_premises_0_based"] == [0, 2]


def test_premise_selection_debug_records_relevant_premises_fallback() -> None:
    example = ReasoningExample(
        example_id="example-1",
        premises=["alpha", "beta", "gamma"],
        question="question",
        answer="answer",
        premises_used=[1],
    )
    prediction = dspy.Prediction(
        answer="answer",
        supporting_premise_indices="[]",
        relevant_premises=["Premise 2. beta"],
    )

    debug = premise_selection_debug(example, prediction)

    assert debug["valid_supporting_premise_indices_0_based"] == []
    assert debug["rematched_relevant_premises_0_based"] == [1]
    assert debug["selected_source"] == "relevant_premises"
    assert debug["selected_premises_used_0_based"] == [1]


def test_parse_premise_texts_string_single_line() -> None:
    assert _parse_premise_texts("alpha") == ["alpha"]


def test_parse_premise_texts_string_multi_line() -> None:
    assert _parse_premise_texts("alpha\nbeta") == ["alpha", "beta"]


def test_parse_premise_texts_string_bracket_per_line() -> None:
    assert _parse_premise_texts("[alpha]\n[beta]") == ["alpha", "beta"]


def test_parse_premise_texts_string_json_array() -> None:
    assert _parse_premise_texts('["alpha", "beta"]') == ["alpha", "beta"]


def test_parse_premise_texts_string_empty() -> None:
    assert _parse_premise_texts("") == []


def test_parse_premise_texts_list_still_works() -> None:
    assert _parse_premise_texts(["alpha", "beta"]) == ["alpha", "beta"]


def test_premise_selection_debug_fallback_with_string_relevant_premises() -> None:
    example = ReasoningExample(
        example_id="example-1",
        premises=["alpha", "beta", "gamma"],
        question="question",
        answer="answer",
        premises_used=[0, 2],
    )
    prediction = dspy.Prediction(
        answer="answer",
        supporting_premise_indices="[]",
        relevant_premises="alpha\nbeta\ngamma",
    )

    debug = premise_selection_debug(example, prediction)

    assert debug["valid_supporting_premise_indices_0_based"] == []
    assert debug["parsed_relevant_premises"] == ["alpha", "beta", "gamma"]
    assert debug["rematched_relevant_premises_0_based"] == [0, 1, 2]
    assert debug["selected_source"] == "relevant_premises"
    assert debug["selected_premises_used_0_based"] == [0, 1, 2]


def _example(answer: str = "answer", premises_used: list[int] | None = None) -> ReasoningExample:
    return ReasoningExample(
        example_id="example-1",
        premises=["alpha", "beta", "gamma"],
        question="question",
        answer=answer,
        premises_used=premises_used if premises_used is not None else [],
    )


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        ("Có", "Có", 1.0),
        ("Có", "có", 1.0),
        ("Có.", "có", 1.0),
        ("Có", "Không", 0.0),
        ("A", "a", 1.0),
    ],
)
def test_exact_match(gold: str, predicted: str, expected: float) -> None:
    assert exact_match(_example(gold), dspy.Prediction(answer=predicted)) == expected


def test_token_f1_perfect_match() -> None:
    assert token_f1(_example("the cat sat"), dspy.Prediction(answer="the cat sat")) == 1.0


def test_token_f1_partial_overlap() -> None:
    # gold: [the, cat, sat] pred: [the, dog, sat] -> common 2; p=2/3 r=2/3 f1=2/3
    assert token_f1(
        _example("the cat sat"), dspy.Prediction(answer="the dog sat")
    ) == pytest.approx(2 / 3)


def test_token_f1_empty_both_returns_one() -> None:
    assert token_f1(_example(""), dspy.Prediction(answer="...")) == 1.0


def test_token_f1_one_empty_returns_zero() -> None:
    assert token_f1(_example("the cat"), dspy.Prediction(answer="...")) == 0.0


def test_token_f1_no_overlap_returns_zero() -> None:
    assert token_f1(_example("alpha"), dspy.Prediction(answer="beta")) == 0.0


def test_contains_present() -> None:
    assert contains(_example("cat"), dspy.Prediction(answer="the cat sat")) == 1.0


def test_contains_absent() -> None:
    assert contains(_example("dog"), dspy.Prediction(answer="the cat sat")) == 0.0


def test_premises_f1_perfect_match_via_primary_indices() -> None:
    example = _example(premises_used=[0, 2])
    prediction = dspy.Prediction(answer="answer", supporting_premise_indices=[1, 3])
    assert premises_f1(example, prediction) == 1.0


def test_premises_f1_partial_overlap() -> None:
    example = _example(premises_used=[0, 1])
    prediction = dspy.Prediction(answer="answer", supporting_premise_indices=[1, 3])
    # predicted 0-based {0, 2}; gold {0, 1}; overlap {0} -> p=1/2 r=1/2 f1=1/2
    assert premises_f1(example, prediction) == pytest.approx(1 / 2)


def test_premises_f1_empty_both_returns_one() -> None:
    example = _example(premises_used=[])
    prediction = dspy.Prediction(answer="answer", supporting_premise_indices="[]")
    assert premises_f1(example, prediction) == 1.0


def test_premises_f1_empty_predicted_returns_zero() -> None:
    example = _example(premises_used=[0])
    prediction = dspy.Prediction(answer="answer", supporting_premise_indices="[]")
    assert premises_f1(example, prediction) == 0.0


def test_premises_f1_falls_back_to_relevant_premises_text() -> None:
    example = _example(premises_used=[1])
    prediction = dspy.Prediction(
        answer="answer",
        supporting_premise_indices="[]",
        relevant_premises="beta",
    )
    assert premises_f1(example, prediction) == 1.0


def test_get_metric_returns_registered_callable() -> None:
    assert get_metric("exact_match") is exact_match
    assert get_metric("token_f1") is token_f1


def test_get_metric_unknown_raises_with_available_list() -> None:
    with pytest.raises(KeyError, match="Unknown metric 'nope'"):
        get_metric("nope")


def test_list_metrics_is_sorted_registered_names() -> None:
    assert list_metrics() == ["contains", "exact_match", "premises_f1", "token_f1"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ([1, 3], [1, 3]),
        ("[]", []),
        ("[1, 3]", [1, 3]),
        ("1 3 99", [1, 3, 99]),
        ("5", [5]),
        (5, [5]),
    ],
)
def test_parse_premise_indices(value: object, expected: list[int]) -> None:
    assert _parse_premise_indices(value) == expected


def test_parse_premise_indices_skips_non_int_items() -> None:
    assert _parse_premise_indices(["1", "x", 3]) == [1, 3]


def test_rematch_premise_indices_accepts_close_match_rejects_distant() -> None:
    premises = ["the cat sat on the mat", "a completely different topic"]
    # A near-exact match to premise 0 is accepted; a distant string is rejected.
    assert _rematch_premise_indices(premises, ["the cat sat on the mat"]) == [0]
    assert _rematch_premise_indices(premises, ["zzzzz unrelated gibberish"]) == []


def test_parse_reasoning_premise_references_extracts_in_range_only() -> None:
    # "tiền đề 1" and "Premise 3" -> 0-based [0, 2]; "Premise 9" is out of range.
    text = "Theo tiền đề 1 và Premise 3, bỏ qua Premise 9."
    assert _parse_reasoning_premise_references(text, num_premises=3) == [0, 2]


def test_parse_reasoning_premise_references_none_returns_empty() -> None:
    assert _parse_reasoning_premise_references(None, num_premises=3) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("text", "text"),
        (42, 42),
        ([1, "a"], [1, "a"]),
        ({"k": 1}, {"k": 1}),
    ],
)
def test_debug_value_primitives(value: object, expected: object) -> None:
    assert _debug_value(value) == expected


def test_debug_value_falls_back_to_str_for_unknown_types() -> None:
    class _Custom:
        def __repr__(self) -> str:
            return "<custom>"

    assert _debug_value(_Custom()) == "<custom>"


class _StubJudge:
    """Stand-in for an LLMJudge: returns a canned verdict on __call__."""

    def __init__(self, verdict: str = "YES", error_type: str = "correct") -> None:
        self._verdict = verdict
        self._error_type = error_type
        self.call_count = 0

    def __call__(self, **_kwargs: object) -> dspy.Prediction:
        self.call_count += 1
        return dspy.Prediction(
            verdict=self._verdict,
            error_type=self._error_type,
            feedback="",
            method="llm",
        )


def test_judge_example_yes_scores_one_and_records_method() -> None:
    outcome = judge_example(
        _example("Có"),
        dspy.Prediction(answer="Có"),
        _StubJudge("YES"),  # pyright: ignore[reportArgumentType]
    )
    assert outcome.score == 1.0
    assert outcome.verdict == "YES"
    assert outcome.method == "llm"


def test_judge_example_no_scores_zero() -> None:
    outcome = judge_example(
        _example("Có"),
        dspy.Prediction(answer="Không"),
        _StubJudge("NO"),  # pyright: ignore[reportArgumentType]
    )
    assert outcome.score == 0.0


def test_llm_judge_returns_score() -> None:
    assert (
        llm_judge(_example("Có"), dspy.Prediction(answer="Có"), judge=_StubJudge("YES"))  # pyright: ignore[reportArgumentType]
        == 1.0
    )
    assert (
        llm_judge(_example("Có"), dspy.Prediction(answer="Không"), judge=_StubJudge("NO"))  # pyright: ignore[reportArgumentType]
        == 0.0
    )
