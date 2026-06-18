from importlib.metadata import version

from virex_bench.models import get_model
from virex_bench.strategies import get_strategy, list_strategies
from virex_bench.tasks import get_task, list_tasks

__version__ = version("virex-bench")

__all__ = [
    "__version__",
    "get_model",
    "get_strategy",
    "get_task",
    "list_strategies",
    "list_tasks",
]
