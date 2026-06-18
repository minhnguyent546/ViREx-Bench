from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.tasks.base import ReasoningExample


class DirectStrategy(ReasoningStrategy):
    """Baseline: ask for the answer directly, no intermediate reasoning."""

    name = "direct"

    def build_prompt(self, example: ReasoningExample) -> str:
        return super().build_prompt(example) + "\nTrả lời ngắn gọn. Câu trả lời:"
