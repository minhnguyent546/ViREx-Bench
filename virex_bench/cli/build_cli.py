import argparse
import json

from virex_bench import envs
from virex_bench.decoding import get_decoding, list_decoding
from virex_bench.evaluation import evaluate, save_report
from virex_bench.logger import init_logger, set_level
from virex_bench.models import get_model
from virex_bench.strategies import get_strategy, list_strategies
from virex_bench.strategies.registry import parse_strategy_name
from virex_bench.tasks import get_task, list_tasks

logger = init_logger(__name__)


def _parse_model_kwargs(raw: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"--model-kwargs must be valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--model-kwargs must be a JSON object")
    return parsed


def _parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return value


def _run(args: argparse.Namespace) -> None:
    set_level(args.log_level)
    task = get_task(args.task)
    lm = get_model(
        args.model,
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key,
        **args.model_kwargs,
    )
    logger.debug(
        f"Loaded model {args.model} with backend {args.backend} and kwargs {args.model_kwargs}"
    )
    base_strategy, _variant = parse_strategy_name(args.strategy)
    signature = task.get_signature(base_strategy)
    rationale_field = task.get_rationale_field(base_strategy)
    strategy = get_strategy(args.strategy, signature, rationale_field=rationale_field)

    decoding_kwargs: dict[str, object] = {}
    if args.decoding_num_samples is not None:
        decoding_kwargs["num_samples"] = args.decoding_num_samples
    decoding_strategy = get_decoding(args.decoding, strategy, **decoding_kwargs)

    report = evaluate(
        task,
        lm,
        strategy,
        model_name=args.model,
        backend=args.backend,
        num_threads=args.num_threads,
        decoding=decoding_strategy,
        max_examples=args.max_examples,
    )

    print(
        f"task={report.task} model={report.model} backend={report.backend} "
        f"strategy={report.strategy} decoding={report.decoding}"
    )
    num_evaluated_examples = getattr(report, "num_evaluated_examples", None) or report.num_examples
    if num_evaluated_examples == report.num_examples:
        example_summary = f"{report.num_examples} examples"
    else:
        example_summary = f"{num_evaluated_examples}/{report.num_examples} examples"
    print(f"{report.metric}={report.score:.4f} ({example_summary})")

    save_report(report, args.output_dir)


def _tasks(args: argparse.Namespace) -> None:
    if args.list:
        for name in list_tasks():
            print(name)


def _strategies(args: argparse.Namespace) -> None:
    if args.list:
        for name in list_strategies():
            print(name)


def _add_run_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vietnamese-logical-reasoning",
        help="Task name",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="direct",
        choices=list_strategies(),
        help="Prompting strategy name",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="litellm provider prefix for the model (e.g. 'openai' for any OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL of the OpenAI-compatible endpoint. Falls back to $OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the endpoint. Falls back to $OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=_parse_model_kwargs,
        default={},
        help=(
            "Extra LM options as a JSON object, forwarded to dspy.LM / litellm "
            '(e.g. \'{"temperature":0.6,"top_p":0.95,"max_tokens":32768,'
            '"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}\'). '
            "Vendor params like chat_template_kwargs must go under extra_body."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=envs.VIREX_BENCH_OUTPUT_DIR,
        help="Directory to write results into",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of worker threads for concurrent example evaluation",
    )
    parser.add_argument(
        "--max-examples",
        type=_parse_positive_int,
        default=None,
        help="Maximum number of dataset examples to evaluate for quick test runs",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="single-pass",
        choices=list_decoding(),
        help="Decoding strategy name (single-pass, self-consistency)",
    )
    parser.add_argument(
        "--self-consistency-num-samples",
        dest="decoding_num_samples",
        metavar="SELF_CONSISTENCY_NUM_SAMPLES",
        type=_parse_positive_int,
        default=None,
        help=(
            "Number of candidate generations for self-consistency decoding "
            "(overrides VIREX_BENCH_SC_NUM_SAMPLES)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default=envs.VIREX_BENCH_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for the virex_bench logger.",
    )


def _add_tasks_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tasks",
    )


def _add_strategies_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available strategies",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="virex-bench",
        description="Vietnamese Reasoning Exploration Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="subcommands", required=True)

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run a model on a task with a strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_run_opts(run_parser)
    parser.set_defaults(func=_run)

    # tasks subcommand
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="Inspect available tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_tasks_opts(tasks_parser)
    tasks_parser.set_defaults(func=_tasks)

    # strategies subcommand
    strategies_parser = subparsers.add_parser(
        "strategies",
        help="Inspect available strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_strategies_opts(strategies_parser)
    strategies_parser.set_defaults(func=_strategies)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
