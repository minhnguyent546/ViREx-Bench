import dspy
from pydantic.fields import FieldInfo

from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.modules import ChainOfThought


class CoTStrategy(ReasoningStrategy):
    """Chain-of-Thought: prompt the model to reason step by step before answering."""

    name = "cot"

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
    ) -> None:
        super().__init__(signature, rationale_field, rationale_field_type)
        self.predict = ChainOfThought(
            signature,
            rationale_field=self.rationale_field,
            rationale_field_type=self.rationale_field_type,
        )

    def forward(self, **inputs: object) -> dspy.Prediction:
        return self.predict(**inputs)

    async def aforward(self, **input: object) -> dspy.Prediction:
        return await self.predict.acall(**input)
