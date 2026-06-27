from collections.abc import Mapping
from typing import Any

import dspy

from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import DatasetConfig, ReasoningExample, TaskMetadata


class VietnameseLogicalReasoningSignature(dspy.Signature):
    """Solve a Vietnamese logical-reasoning problem.

    Given a set of premises and a question, reason over the premises to produce
    the correct answer to the question. Use only the information stated in the
    given premises. The premises and question are written in Vietnamese; the
    answer is one of the options offered in the question, or a short value such
    as a label, a number, or a name.
    """

    premises: list[str] = dspy.InputField(desc="The list of premises.")
    question: str = dspy.InputField(desc="The question to answer based on the premises.")
    answer: str = dspy.OutputField(desc="The final answer to the question.")


class VietnameseLogicalReasoning(ReasoningTask):
    metadata = TaskMetadata(
        name="vietnamese-logical-reasoning",
        description=(
            "Vietnamese logical-reasoning task: given a set of premises and a "
            "question, derive the answer that is logically supported by the "
            "premises. Covers multiple-choice, yes/no/uncertain, numeric, and "
            "short free-text answers."
        ),
        language="vie",
        dataset=DatasetConfig(
            path="minhnguyent546/virex-bench-datasets",
            name="logical-reasoning",
            split="test",
            revision="b55afc3316bd905b33523e4c9af2e36d9dd012b3",
            num_proc=2,
        ),
        main_metric="llm_judge",
        judge="logical_reasoning",
    )
    signatures = {"default": VietnameseLogicalReasoningSignature}

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        return ReasoningExample(
            example_id=str(row["query_id"]),
            premises=[str(premise) for premise in row["premises"]],
            question=str(row["query"]),
            answer=str(row["answer"]),
        )
