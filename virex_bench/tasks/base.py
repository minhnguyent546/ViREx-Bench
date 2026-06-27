from collections.abc import Mapping
from typing import Any, cast

import dspy

from virex_bench.logger import init_logger
from virex_bench.types import ReasoningExample, TaskMetadata

logger = init_logger(__name__)


class ReasoningTask:
    """Base class for reasoning tasks. Subclasses provide metadata and load examples."""

    metadata: TaskMetadata
    signatures: dict[str, type[dspy.Signature]]

    def load_examples(self) -> list[ReasoningExample]:
        """Load every row of the configured HuggingFace dataset as a reasoning example."""
        import datasets

        from virex_bench import envs

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
        return [self._row_to_example(cast(Mapping[str, Any], row)) for row in loaded]

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        """Map a single HuggingFace dataset row to a ReasoningExample. Override per task."""
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
