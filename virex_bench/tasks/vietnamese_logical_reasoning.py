from collections.abc import Mapping
from typing import Any, Literal

import dspy

from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import DatasetConfig, ReasoningExample, TaskMetadata


class VietnameseLogicalReasoningSignature(dspy.Signature):
    """Solve a Vietnamese logical-reasoning problem.

    Given a set of premises and a question, reason strictly over the premises to
    produce the correct answer to the question. The premises and question are
    written in Vietnamese.

    Source of truth:
    - Use ONLY the information stated in the given premises. The premises are the
      single source of truth.
    - Do NOT rely on outside knowledge, commonsense reasoning, or assumptions not
      entailed by the premises.
    - Do NOT use closed-world reasoning: a claim is not false merely because it is
      not stated. If the premises entail neither a claim nor its negation, the
      answer is uncertain.

    Answer type:
    - First infer `answer_type` from the question. It must be exactly one of:
      "multiple_choice", "yes_no_uncertain", "numeric", "open_ended".
      - multiple_choice: the question lists explicit answer choices in the text
        (e.g. "A.", "(B)", "C)", or "Đáp án nào sau đây ...").
      - yes_no_uncertain: the question asks whether a specific claim holds
        (e.g. "Có phải ...", "Liệu ...", "Có thể kết luận ... không").
      - numeric: the question asks for a count or numeric value
        (e.g. "Bao nhiêu ...", "Có mấy ...").
      - open_ended: the question asks for a derived fact, entity, name, list, or
        short conclusion that is not one of the above.
    - Then format `answer` according to the inferred `answer_type` as described
      below.

    Answer formatting by type:
    - multiple_choice: return only the option label letter(s). For a single
      answer, return one uppercase letter (e.g. "A"). For multiple answers,
      return the letters separated by ", " in alphabetical order (e.g. "B, C").
      Do NOT include the option text, punctuation, or any explanation.
    - yes_no_uncertain: return exactly one of the Vietnamese labels "Có",
      "Không", or "Không chắc chắn". Use "Có" when the premises entail the claim,
      "Không" when the premises entail its negation, and "Không chắc chắn" when
      the premises entail neither. Answers in any other language or variant
      (e.g. "Yes", "No", "Uncertain") are considered wrong.
    - numeric: return only the number whenever possible (e.g. "3"). Include
      accompanying text only when it is genuinely required to make the answer
      meaningful (e.g. a unit such as "3 người").
    - open_ended: return a concise answer in Vietnamese (a short phrase or
      sentence) stating the entity, fact, or conclusion entailed by the premises.
      If no requested fact is entailed, return "Không chắc chắn".
    """

    premises: list[str] = dspy.InputField(desc="The list of premises. The only source of truth.")
    question: str = dspy.InputField(desc="The question to answer based on the premises.")

    answer_type: Literal["multiple_choice", "yes_no_uncertain", "numeric", "open_ended"] = (
        dspy.OutputField(
            desc=(
                "The type of answer inferred from the question. Exactly one of: "
                "multiple_choice, yes_no_uncertain, numeric, open_ended."
            )
        )
    )
    answer: str = dspy.OutputField(
        desc=(
            "The final answer, formatted according to `answer_type`:\n"
            "- multiple_choice: option label letter(s) only, e.g. 'A' or 'B, C' "
            "(uppercase, alphabetical, comma-separated, no option text).\n"
            "- yes_no_uncertain: exactly one of 'Có', 'Không', 'Không chắc chắn' "
            "(Vietnamese only; other languages/variants are wrong).\n"
            "- numeric: the number only when possible, e.g. '3'; add text only "
            "when truly needed, e.g. '3 người'.\n"
            "- open_ended: a concise Vietnamese phrase or sentence; "
            "'Không chắc chắn' if nothing is entailed."
        )
    )


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
            revision="712522ef946b72b6d1d7a34d5fbab98696feac54",
            num_proc=2,
        ),
        main_metric="llm_judge",
        judge="logical_reasoning",
    )
    signatures = {
        "default": VietnameseLogicalReasoningSignature,
        "direct": VietnameseLogicalReasoningSignature,
        "cot": VietnameseLogicalReasoningSignature,
        # 'tot': TODO,
        # 'pot_z3': TODO,
    }
    rationale_fields = {
        "default": dspy.OutputField(
            desc=(
                "Step-by-step reasoning over the premises that leads to the answer. "
                "Cite only what the premises state and avoid outside knowledge. "
                "IMPORTANT: write this reasoning in Vietnamese."
            )
        ),
    }

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        return ReasoningExample(
            example_id=str(row["query_id"]),
            premises=[str(premise) for premise in row["premises"]],
            question=str(row["query"]),
            answer=str(row["answer"]),
        )
