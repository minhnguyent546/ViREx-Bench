from importlib.metadata import metadata, version

from virex_bench.decoding import get_decoding, list_decoding
from virex_bench.evaluation import evaluate, save_report
from virex_bench.models import get_model
from virex_bench.strategies import get_strategy, list_strategies, parse_strategy_name
from virex_bench.tasks import get_task, list_tasks


def _get_git_revision() -> str:
    project_metadata = metadata("virex-bench")
    project_urls = project_metadata.get_all("Project-URL") or []
    for project_url in project_urls:
        label, separator, value = project_url.partition(",")
        if label.strip() != "Source commit" or separator == "":
            continue
        revision = value.rstrip("/").rsplit("/", maxsplit=1)[-1].strip()
        if revision != "":
            return revision[:7]
    return "unknown"


__version__ = version("virex-bench")
__git_revision__ = _get_git_revision()

__all__ = [
    "__git_revision__",
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
