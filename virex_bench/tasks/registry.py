from virex_bench.tasks.base import ReasoningTask
from virex_bench.tasks.vietnamese_logical_reasoning import VietnameseLogicalReasoning

_TASK_REGISTRY: dict[str, type[ReasoningTask]] = {
    VietnameseLogicalReasoning.metadata.name: VietnameseLogicalReasoning,
}


def get_task(name: str) -> ReasoningTask:
    if name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY))
        raise KeyError(f"Unknown task {name!r}. Available tasks: {available}")
    return _TASK_REGISTRY[name]()


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)
