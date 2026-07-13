import dspy
from pydantic.fields import FieldInfo

from virex_bench.strategies.base import ReasoningStrategy


class DirectStrategy(ReasoningStrategy):
    """Baseline: ask for the answer directly, no intermediate reasoning."""

    name = "direct"
    description = "Baseline: answer directly with no intermediate reasoning."

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
    ) -> None:
        super().__init__(signature, rationale_field, rationale_field_type)
        self.predict = dspy.Predict(signature)

    def forward(self, **inputs: object) -> dspy.Prediction:
        return self.predict(**inputs)

    async def aforward(self, **input: object) -> dspy.Prediction:
        return await self.predict.acall(**input)
