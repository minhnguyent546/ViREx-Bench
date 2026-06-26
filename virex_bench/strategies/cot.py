import dspy

from virex_bench.strategies.base import ReasoningStrategy


class CoTStrategy(ReasoningStrategy):
    """Chain-of-Thought: prompt the model to reason step by step before answering."""

    name = "cot"

    def __init__(self, signature: type[dspy.Signature]) -> None:
        super().__init__(signature)
        self.predict = dspy.ChainOfThought(signature)

    def forward(self, **inputs: object) -> dspy.Prediction:
        return self.predict(**inputs)
