import argparse

from virex_bench import envs
from virex_bench.evaluation import evaluate, save_report
from virex_bench.logger import init_logger, set_level
from virex_bench.models import get_model
from virex_bench.strategies import get_strategy, list_strategies
from virex_bench.tasks import get_task, list_tasks

logger = init_logger(__name__)


def _run(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    model = get_model(args.model, backend=args.backend)
    strategy = get_strategy(args.strategy)

    report = evaluate(task, model, strategy, backend=args.backend)

    print(
        f"task={report.task} model={report.model} backend={report.backend} "
        f"strategy={report.strategy}"
    )
    print(f"accuracy={report.accuracy:.4f} ({report.num_examples} examples)")

    save_report(report, args.output_dir)


def _tasks(args: argparse.Namespace) -> None:
    if args.list:
        for name in list_tasks():
            print(name)


def _strategies(args: argparse.Namespace) -> None:
    if args.list:
        for name in list_strategies():
            print(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="virex-bench",
        description="Vietnamese Reasoning Exploration Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default=envs.VIREX_BENCH_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for the virex_bench logger.",
    )
    subparsers = parser.add_subparsers(title="subcommands", required=True)

    run_parser = subparsers.add_parser("run", help="Run a model on a task with a strategy")
    run_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    run_parser.add_argument(
        "--task", type=str, default="vietnamese-logical-reasoning", help="Task name"
    )
    run_parser.add_argument(
        "--strategy", type=str, default="direct", help="Prompting strategy name"
    )
    run_parser.add_argument(
        "--backend",
        type=str,
        default="echo",
        help="Model backend (echo, vllm, hf, api). Only 'echo' is implemented for now.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=envs.VIREX_BENCH_OUTPUT_DIR,
        help="Directory to write results into",
    )
    run_parser.set_defaults(func=_run)

    tasks_parser = subparsers.add_parser("tasks", help="Inspect available tasks")
    tasks_parser.add_argument("--list", action="store_true", help="List available tasks")
    tasks_parser.set_defaults(func=_tasks)

    strategies_parser = subparsers.add_parser("strategies", help="Inspect available strategies")
    strategies_parser.add_argument("--list", action="store_true", help="List available strategies")
    strategies_parser.set_defaults(func=_strategies)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_level(args.log_level)
    args.func(args)
