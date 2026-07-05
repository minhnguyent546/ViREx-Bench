import dspy

from virex_bench.evaluation.metrics import premise_selection_debug
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
