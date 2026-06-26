import dspy

from virex_bench.tasks.base import ReasoningExample, ReasoningTask, TaskMetadata


class VietnameseLogicalReasoningSignature(dspy.Signature):
    """Solve a Vietnamese logical-reasoning problem.

    Given a set of premises and a question, reason over the premises to produce
    the correct answer to the question. Use only the information stated in the
    given premises. The premises, question, and answer are written in Vietnamese.
    """

    premises: list[str] = dspy.InputField(desc="The list of premises.")
    question: str = dspy.InputField(desc="The question to answer based on the premises.")
    answer: str = dspy.OutputField(desc="The final answer to the question.")


# A handful of toy Vietnamese logical-reasoning examples so the CLI is runnable
# end-to-end before the real dataset lands under `data/`.
_TOY_EXAMPLES: list[ReasoningExample] = [
    ReasoningExample(
        example_id="toy-001",
        premises=[
            "Tất cả các con mèo đều là động vật.",
            "Kitty là một con mèo.",
        ],
        question="Kitty có phải là động vật không?",
        answer="Có",
    ),
    ReasoningExample(
        example_id="toy-002",
        premises=[
            "Nếu trời mưa thì đường ướt.",
            "Trời đang mưa.",
        ],
        question="Đường có ướt không?",
        answer="Có",
    ),
    ReasoningExample(
        example_id="toy-003",
        premises=[
            "An cao hơn Bình.",
            "Bình cao hơn Cường.",
        ],
        question="An có cao hơn Cường không?",
        answer="Có",
    ),
]


class VietnameseLogicalReasoning(ReasoningTask):
    metadata = TaskMetadata(
        name="vietnamese-logical-reasoning",
        description="Vietnamese logical-reasoning task: premises + a question with a gold answer.",
        language="vie",
    )
    signatures = {"default": VietnameseLogicalReasoningSignature}

    def load_examples(self) -> list[ReasoningExample]:
        return list(_TOY_EXAMPLES)
