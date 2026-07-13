import argparse
import json
from typing import Any

from tabulate import tabulate

from virex_bench import __git_revision__, __version__, envs
from virex_bench.decoding import decoding_descriptions, get_decoding, list_decoding
from virex_bench.evaluation import evaluate, save_report
from virex_bench.logger import init_logger, set_level
from virex_bench.models import get_model
from virex_bench.strategies import list_strategies, parse_strategy_name, strategy_descriptions
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


def _format_version() -> str:
    return f"virex-bench: v{__version__}\ngit revision: {__git_revision__}"


class _VersionAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        print(_format_version())
        parser.exit()


def _run(args: argparse.Namespace) -> None:
    set_level(args.log_level)
    task = get_task(args.task)
    lm = get_model(
        args.model,
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key
        or "<empty>",  # sometimes providing empty string api key can causing issues [?]
        **args.model_kwargs,
    )
    logger.debug(
        f"Loaded model {args.model} with backend {args.backend} and kwargs {args.model_kwargs}"
    )
    strategy = task.get_strategy(args.strategy)

    decoding_kwargs: dict[str, object] = {}
    if args.decoding_num_samples is not None:
        decoding_kwargs["num_samples"] = args.decoding_num_samples
    if args.self_certainty_borda_power is not None:
        decoding_kwargs["borda_power"] = args.self_certainty_borda_power
    decoding_strategy = get_decoding(args.decoding, strategy, **decoding_kwargs)

    report = evaluate(
        task=task,
        lm=lm,
        strategy=strategy,
        num_threads=args.num_threads,
        decoding=decoding_strategy,
        max_examples=args.max_examples,
    )

    print(
        f"task={report.task} model={report.model}"
        f"strategy={report.strategy} decoding={report.decoding}"
    )
    num_evaluated_examples = getattr(report, "num_evaluated_examples", None) or report.num_examples
    if num_evaluated_examples == report.num_examples:
        example_summary = f"{report.num_examples} examples"
    else:
        example_summary = f"{num_evaluated_examples}/{report.num_examples} examples"
    print(f"{report.metric}={report.score:.4f} ({example_summary})")

    save_report(report, args.output_dir)


_DESCRIPTION_COLUMN_WIDTH = 56


def _print_registry_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a registry listing as an aligned table via ``tabulate``.

    The last column (the description) is word-wrapped so long summaries stay
    readable without blowing out the terminal width.
    """
    print(title)
    print()
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            maxcolwidths=[None] * (len(headers) - 1) + [_DESCRIPTION_COLUMN_WIDTH],
        )
    )


def _tasks(args: argparse.Namespace) -> None:
    if not args.list:
        return
    rows = [[name, get_task(name).metadata.description] for name in list_tasks()]
    _print_registry_table(f"Available tasks ({len(rows)})", ["NAME", "DESCRIPTION"], rows)


def _strategies(args: argparse.Namespace) -> None:
    if not args.list:
        return
    descriptions = strategy_descriptions()
    grouped_variants: dict[str, list[str | None]] = {}
    for name in list_strategies():
        base_name, variant = parse_strategy_name(name)
        grouped_variants.setdefault(base_name, []).append(variant)
    rows: list[list[str]] = []
    for base_name in sorted(grouped_variants):
        variants = [entry for entry in grouped_variants[base_name] if entry is not None]
        rows.append(
            [
                base_name,
                descriptions.get(base_name, ""),
                ", ".join(variants),
            ]
        )
    _print_registry_table(
        f"Available strategies ({len(rows)})",
        ["NAME", "DESCRIPTION", "VARIANTS"],
        rows,
    )


def _decodings(args: argparse.Namespace) -> None:
    if not args.list:
        return
    descriptions = decoding_descriptions()
    rows = [[name, descriptions.get(name, "")] for name in list_decoding()]
    _print_registry_table(
        f"Available decoding strategies ({len(rows)})",
        ["NAME", "DESCRIPTION"],
        rows,
    )


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
        help="Decoding strategy name",
    )
    parser.add_argument(
        "--self-consistency-num-samples",
        dest="decoding_num_samples",
        metavar="NUM_SAMPLES",
        type=_parse_positive_int,
        default=None,
        help=(
            "Number of candidate generations for self-consistency / "
            "self-certainty decoding (overrides VIREX_BENCH_SC_NUM_SAMPLES / "
            "VIREX_BENCH_SELFC_NUM_SAMPLES)"
        ),
    )
    parser.add_argument(
        "--self-certainty-num-samples",
        dest="decoding_num_samples",
        metavar="NUM_SAMPLES",
        type=_parse_positive_int,
        default=None,
        help="Alias for --self-consistency-num-samples (self-certainty decoding).",
    )
    parser.add_argument(
        "--self-certainty-borda-power",
        metavar="BORDA_POWER",
        type=float,
        default=None,
        help=(
            "Borda voting exponent for self-certainty decoding "
            "(0 = majority vote, larger = pure certainty selection; overrides "
            "VIREX_BENCH_SELFC_BORDA_POWER)"
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
        "-l",
        "--list",
        action="store_true",
        help="List available tasks",
    )


def _add_strategies_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available strategies",
    )


def _add_decodings_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available decoding strategies",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="virex-bench",
        description="Vietnamese Reasoning Exploration Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action=_VersionAction,
        nargs=0,
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

    # decodings subcommand
    decodings_parser = subparsers.add_parser(
        "decodings",
        help="Inspect available decoding strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_decodings_opts(decodings_parser)
    decodings_parser.set_defaults(func=_decodings)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
