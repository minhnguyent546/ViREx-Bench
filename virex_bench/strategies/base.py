import dspy
from pydantic.fields import FieldInfo


class ReasoningStrategy(dspy.Module):
    """Base class for prompting / inference-time-scaling strategies.

    A strategy is a generic `dspy.Module` (the reasoning *algorithm*); the
    task-specific instructions live in the `dspy.Signature` handed to it by the
    task. Subclasses wire the signature into a concrete DSPy module in `__init__`
    and delegate to it in `forward`.
    """

    name: str = "base"

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
    ) -> None:
        super().__init__()
        self.signature = signature

    def forward(self, **inputs: object) -> dspy.Prediction:
        raise NotImplementedError()

    async def aforward(self, **input: object) -> dspy.Prediction:
        raise NotImplementedError()
