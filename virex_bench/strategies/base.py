import dspy
from pydantic.fields import FieldInfo


class ReasoningStrategy(dspy.Module):
    """Base class for prompting / inference-time-scaling strategies.

    A strategy is a generic `dspy.Module` (the reasoning *algorithm*); the
    task-specific instructions live in the `dspy.Signature` handed to it by the
    task. Subclasses wire the signature into a concrete DSPy module in `__init__`
    and delegate to it in `forward`.

    Reasoning strategies (CoT, ToT, ...) prepend a reasoning field to the
    signature. The task can supply a custom `rationale_field` (and its type) to
    control that field's instruction — e.g. to require the reasoning be written
    in the task language. Strategies without a reasoning step (e.g. the direct
    baseline) accept these arguments for a uniform constructor signature and
    ignore them.
    """

    name: str = "base"
    description: str = ""
    # Strategies that expose a CLI variant axis (e.g. ``tot-beam``) override this.
    accepts_variant: bool = False

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
    ) -> None:
        super().__init__()
        self.signature = signature
        self.rationale_field = rationale_field
        self.rationale_field_type = rationale_field_type

    def forward(self, **inputs: object) -> dspy.Prediction:
        raise NotImplementedError()

    async def aforward(self, **input: object) -> dspy.Prediction:
        raise NotImplementedError()

    @property
    def report_config(self) -> dict[str, object]:
        """Configuration parameters to record in the evaluation report."""
        return {}
