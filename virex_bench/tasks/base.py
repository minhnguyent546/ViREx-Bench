from collections.abc import Mapping
from typing import Any, cast

import datasets
import dspy
from pydantic.fields import FieldInfo

from virex_bench import envs
from virex_bench.logger import init_logger
from virex_bench.types import ReasoningExample, TaskMetadata

logger = init_logger(__name__)


class ReasoningTask:
    """Base class for reasoning tasks. Subclasses provide metadata and load examples."""

    metadata: TaskMetadata
    signatures: dict[str, type[dspy.Signature]]
    rationale_fields: dict[str, FieldInfo] = {}
    # Name of the dataset column to derive `ReasoningExample.category` from. When
    # set, per-category score breakdowns are produced automatically for any task;
    # set to None to opt out, or override per task to point at a different column.
    category_column: str | None = "category"

    def load_examples(self) -> list[ReasoningExample]:
        """Load every row of the configured HuggingFace dataset as a reasoning example.

        Each row is mapped to a ``ReasoningExample`` via ``_row_to_example``. After
        the mapping, if the example has no ``category`` and ``category_column`` is
        set, the category is filled from that dataset column. This lets any task
        whose dataset exposes a category-like column get per-category score
        breakdowns for free, without task-specific code; a task may still set
        ``category`` itself in ``_row_to_example`` to override the column.
        """

        dataset_config = self.metadata.dataset
        logger.info(
            f"Loading dataset {dataset_config.path}"
            f"{f' [{dataset_config.name}]' if dataset_config.name else ''}"
            f" split={dataset_config.split}"
        )
        loaded = datasets.load_dataset(
            dataset_config.path,
            dataset_config.name,
            split=dataset_config.split,
            revision=dataset_config.revision,
            num_proc=dataset_config.num_proc,
            token=envs.HF_TOKEN,
        )
        examples: list[ReasoningExample] = []
        for row in loaded:
            row_mapping = cast(Mapping[str, Any], row)
            example = self._row_to_example(row_mapping)
            if example.category is None and self.category_column is not None:
                raw_category = row_mapping.get(self.category_column)
                example.category = str(raw_category) if raw_category is not None else None
            examples.append(example)
        return examples

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        """Map a single HuggingFace dataset row to a ReasoningExample. Override per task."""
        raise NotImplementedError

    def get_signature(self, strategy_name: str) -> type[dspy.Signature]:
        """Return the signature for a given strategy, falling back to the default."""
        return self.signatures.get(strategy_name, self.signatures["default"])

    def get_rationale_field(self, strategy_name: str) -> FieldInfo | None:
        """Return the reasoning-field override for a reasoning strategy, if any.

        Falls back to the ``"default"`` entry, then to ``None`` (which lets the
        strategy use the DSPy module's built-in reasoning field).
        """
        return self.rationale_fields.get(strategy_name, self.rationale_fields.get("default"))

    def example_to_inputs(self, example: ReasoningExample) -> dict[str, object]:
        """Convert an example into the dict of input kwargs for the signature."""
        return {"premises": example.premises, "question": example.question}

    @property
    def name(self) -> str:
        return self.metadata.name
