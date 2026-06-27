import dspy

from virex_bench.strategies.base import ReasoningStrategy


class DirectStrategy(ReasoningStrategy):
    """Baseline: ask for the answer directly, no intermediate reasoning."""

    name = "direct"

    def __init__(self, signature: type[dspy.Signature]) -> None:
        super().__init__(signature)
        self.predict = dspy.Predict(signature)

    def forward(self, **inputs: object) -> dspy.Prediction:
        return self.predict(**inputs)

    async def aforward(self, **input: object) -> dspy.Prediction:
        return await self.predict.acall(**input)
