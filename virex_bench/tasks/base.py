import dspy
from pydantic import BaseModel, Field


class ReasoningExample(BaseModel):
    """A single logical-reasoning instance: premises + a question and its gold answer."""

    premises: list[str]
    question: str
    answer: str
    example_id: str = ""


class TaskMetadata(BaseModel):
    name: str
    description: str
    language: str = "vie"


class ReasoningTask:
    """Base class for reasoning tasks. Subclasses provide metadata and load examples."""

    metadata: TaskMetadata
    signatures: dict[str, type[dspy.Signature]]

    def load_examples(self) -> list[ReasoningExample]:
        raise NotImplementedError

    def get_signature(self, strategy_name: str) -> type[dspy.Signature]:
        """Return the signature for a given strategy, falling back to the default."""
        return self.signatures.get(strategy_name, self.signatures["default"])

    def example_to_inputs(self, example: ReasoningExample) -> dict[str, object]:
        """Convert an example into the dict of input kwargs for the signature."""
        return {"premises": example.premises, "question": example.question}

    @property
    def name(self) -> str:
        return self.metadata.name


class TaskResult(BaseModel):
    """Per-example prediction record produced during evaluation."""

    example_id: str
    predicted: str
    gold: str
    is_correct: bool
    extra: dict[str, object] = Field(default_factory=dict)
