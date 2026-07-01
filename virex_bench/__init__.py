from importlib.metadata import version

from virex_bench.decoding import get_decoding, list_decoding
from virex_bench.evaluation import evaluate, save_report
from virex_bench.models import get_model
from virex_bench.strategies import get_strategy, list_strategies, parse_strategy_name
from virex_bench.tasks import get_task, list_tasks

__version__ = version("virex-bench")

__all__ = [
    "get_decoding",
    "list_decoding",
    "evaluate",
    "save_report",
    "get_model",
    "get_strategy",
    "list_strategies",
    "parse_strategy_name",
    "list_tasks",
    "get_task",
]
