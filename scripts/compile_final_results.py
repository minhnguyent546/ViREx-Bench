#!/usr/bin/env python3
"""Compile ViREx-Bench final-result JSONs into one comprehensive CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from typing import Any

# Canonical ordering for stable, paper-friendly row output.
STRATEGY_ORDER = ["direct", "cot", "cr", "pot_z3", "tot-beam", "tot-dfs", "tot-mcts"]
DECODING_ORDER = ["single-pass", "self-consistency"]
CATEGORY_ORDER = ["yes_no_uncertain", "text", "mcq", "number"]

ROUND_N = 4

# Ordered column groups — the union of every stat we may encounter. Cells are
# left empty when a strategy/decoding does not report that stat.
COLUMNS: list[str] = [
    # identifiers
    "model",
    "model_org",
    "model_short",
    "seed",
    "strategy",
    "decoding",
    "k",
    # counts
    "num_examples",
    "num_evaluated",
    "num_failed",
    "failure_rate",
    "num_answered",
    # scores & components
    "score",
    "llm_judge_score",
    "llm_judge_accuracy",
    "llm_judge_correct",
    "llm_judge_ci95_low",
    "llm_judge_ci95_high",
    "premises_f1",
    "cat_yes_no_uncertain",
    "cat_text",
    "cat_mcq",
    "cat_number",
    # efficiency
    "total_time_s",
    "avg_time_per_example_s",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_total_tokens",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_total_tokens",
    # search stats (CR / ToT-*)
    "avg_nodes_visited",
    "avg_llm_calls",
    "avg_completions",
    "avg_propose_calls",
    "avg_evaluate_calls",
    "avg_depth_reached",
    "avg_best_score",
    # CR-specific
    "avg_validity_calls",
    "avg_meaningfulness_calls",
    "avg_entailed_count",
    "avg_contradicted_count",
    "avg_undetermined_count",
    "context_overflow_rate",
    # PoT-specific
    "avg_generate_calls",
    "avg_regenerate_calls",
    "avg_execute_calls",
    "execution_success_rate",
    # self-consistency-specific
    "avg_vote_count",
    "avg_total_samples",
    "avg_confidence",
    "aggregator_applied_rate",
    # model / judge config
    "temperature",
    "top_p",
    "top_k",
    "enable_thinking",
    "judge_model",
    # provenance
    "strategy_kwargs",
    "decoding_kwargs",
    "source_file",
]

_FILENAME_TS_RE = re.compile(r"results-(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})\.json$")


def wilson_ci(correct: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (accuracy)."""
    if total <= 0:
        return float("nan"), float("nan")
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence]
    p_hat = correct / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    half = (z / denom) * (p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5
    return center - half, center + half


def _num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def avg_field(results: list[dict[str, Any]], stats_key: str, field: str) -> float | None:
    """Mean of ``results[i]['extra'][stats_key][field]`` over examples that
    report it as a number. Returns ``None`` if no example reports it."""
    vals: list[float] = []
    for row in results:
        extra = row.get("extra")
        if not isinstance(extra, dict):
            continue
        stats = extra.get(stats_key)
        if not isinstance(stats, dict):
            continue
        value = stats.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vals.append(float(value))
    if not vals:
        return None
    return statistics.fmean(vals)


def rate_field(results: list[dict[str, Any]], stats_key: str, field: str) -> float | None:
    """Fraction of ``True`` for a boolean field across reporting examples."""
    vals = []
    for row in results:
        extra = row.get("extra")
        if not isinstance(extra, dict):
            continue
        stats = extra.get(stats_key)
        if not isinstance(stats, dict):
            continue
        if field in stats and isinstance(stats[field], bool):
            vals.append(1.0 if stats[field] else 0.0)
    if not vals:
        return None
    return statistics.fmean(vals)


def fmt(x: float | int | bool | None) -> str:
    """Render a value for CSV: round floats, drop ``None`` to empty cell."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:  # NaN
            return ""
        return f"{round(x, ROUND_N):.{ROUND_N}f}"

    return str(x)


def clean_model(model: str) -> tuple[str, str]:
    """Split a backend model id into (org, short_name).

    ``hosted_vllm/Qwen/Qwen3-8B`` -> (``Qwen``, ``Qwen3-8B``).
    Falls back to (``unknown``, last segment) for unexpected shapes.
    """
    parts = model.split("/")
    if len(parts) >= 3 and parts[0] == "hosted_vllm":
        return parts[1], parts[2]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "unknown", parts[-1]


def extract_seed(path: str) -> str:
    """Pull the ``seed-N`` segment out of a results path."""
    for segment in path.split(os.sep):
        if re.fullmatch(r"seed-\d+", segment):
            return segment
    return ""


def file_sort_key(path: str) -> str:
    """Chronological sort key from the ``results-YYYY_MM_DD-HH_MM_SS`` stamp."""
    match = _FILENAME_TS_RE.search(os.path.basename(path))
    return match.group(1) if match else ""


def load_row(path: str, model_dir: str, strategy: str, decoding: str, seed: str) -> dict[str, Any]:
    """Build one CSV row from a single result JSON file."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    results: list[dict[str, Any]] = data.get("results", [])
    score_components: dict[str, Any] = data.get("score_components", {}) or {}
    category_scores: dict[str, Any] = data.get("category_scores", {}) or {}
    token_usage: dict[str, Any] = data.get("token_usage", {}) or {}
    total_token_usage: dict[str, Any] = data.get("total_token_usage", {}) or {}
    model_kwargs: dict[str, Any] = data.get("model_kwargs", {}) or {}
    decoding_kwargs: dict[str, Any] = data.get("decoding_kwargs", {}) or {}
    strategy_kwargs: dict[str, Any] = data.get("strategy_kwargs", {}) or {}
    chat_template_kwargs: dict[str, Any] = model_kwargs.get("chat_template_kwargs", {}) or {}

    num_evaluated: int = int(data.get("num_evaluated_examples", len(results)))
    num_failed: int = int(data.get("num_failed", 0))
    num_examples: int = int(data.get("num_examples", num_evaluated))

    # llm_judge_score is binary per-example (1.0 correct / 0.0 wrong / None when
    # the strategy failed to produce an answer). The reported top-level
    # ``llm_judge_score`` is therefore *conditional on answering* (correct /
    # num_answered). For the paper we also want the *effective* accuracy that
    # counts failures as wrong (correct / num_evaluated), which is consistent
    # with the failure-penalized ``score`` and ``category_scores``.
    num_answered: int = sum(1 for row in results if row.get("llm_judge_score") is not None)
    llm_judge_correct: int = sum(1 for row in results if row.get("llm_judge_score") == 1.0)
    llm_judge_accuracy: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    if num_evaluated > 0:
        llm_judge_accuracy = llm_judge_correct / num_evaluated
        ci_low, ci_high = wilson_ci(llm_judge_correct, num_evaluated)

    total_time = data.get("total_time")
    avg_time_per_ex: float | None = None
    if _num(total_time) and num_evaluated > 0:
        avg_time_per_ex = float(total_time) / num_evaluated

    row: dict[str, Any] = dict.fromkeys(COLUMNS)

    model_full = data.get("model", model_dir)
    org, short = clean_model(str(model_full))
    row["model"] = model_full
    row["model_org"] = org
    row["model_short"] = short
    row["seed"] = seed
    row["strategy"] = data.get("strategy", strategy)
    row["decoding"] = data.get("decoding", decoding)
    row["k"] = decoding_kwargs.get("num_samples", 1) if decoding == "self-consistency" else 1

    row["num_examples"] = num_examples
    row["num_evaluated"] = num_evaluated
    row["num_failed"] = num_failed
    row["failure_rate"] = (num_failed / num_evaluated) if num_evaluated else None
    row["num_answered"] = num_answered

    row["score"] = data.get("score")
    row["llm_judge_score"] = score_components.get("llm_judge_score")
    row["llm_judge_accuracy"] = llm_judge_accuracy
    row["llm_judge_correct"] = llm_judge_correct
    row["llm_judge_ci95_low"] = ci_low
    row["llm_judge_ci95_high"] = ci_high
    row["premises_f1"] = score_components.get("premises_f1")
    for cat in CATEGORY_ORDER:
        row[f"cat_{cat}"] = (category_scores.get(cat, {}) or {}).get("score")

    row["total_time_s"] = total_time
    row["avg_time_per_example_s"] = avg_time_per_ex
    row["avg_prompt_tokens"] = token_usage.get("prompt_tokens")
    row["avg_completion_tokens"] = token_usage.get("completion_tokens")
    row["avg_total_tokens"] = token_usage.get("total_tokens")
    row["total_prompt_tokens"] = total_token_usage.get("prompt_tokens")
    row["total_completion_tokens"] = total_token_usage.get("completion_tokens")
    row["total_total_tokens"] = total_token_usage.get("total_tokens")

    # Search stats (CR + ToT-*). avg_llm_calls is unified across search & PoT.
    row["avg_nodes_visited"] = avg_field(results, "search_stats", "nodes_visited")
    search_llm_calls = avg_field(results, "search_stats", "total_llm_calls")
    row["avg_llm_calls"] = (
        search_llm_calls
        if search_llm_calls is not None
        else avg_field(results, "pot_info", "total_llm_calls")
    )
    row["avg_completions"] = avg_field(results, "search_stats", "total_completions")
    row["avg_propose_calls"] = avg_field(results, "search_stats", "propose_calls")
    row["avg_evaluate_calls"] = avg_field(results, "search_stats", "evaluate_calls")
    row["avg_depth_reached"] = avg_field(results, "search_stats", "depth_reached")
    row["avg_best_score"] = avg_field(results, "search_stats", "best_score")

    # CR-specific verifier bookkeeping.
    row["avg_validity_calls"] = avg_field(results, "search_stats", "validity_calls")
    row["avg_meaningfulness_calls"] = avg_field(results, "search_stats", "meaningfulness_calls")
    row["avg_entailed_count"] = avg_field(results, "search_stats", "entailed_count")
    row["avg_contradicted_count"] = avg_field(results, "search_stats", "contradicted_count")
    row["avg_undetermined_count"] = avg_field(results, "search_stats", "undetermined_count")
    row["context_overflow_rate"] = rate_field(results, "search_stats", "context_overflow")

    # PoT-specific.
    row["avg_generate_calls"] = avg_field(results, "pot_info", "generate_calls")
    row["avg_regenerate_calls"] = avg_field(results, "pot_info", "regenerate_calls")
    row["avg_execute_calls"] = avg_field(results, "pot_info", "execute_calls")
    row["execution_success_rate"] = rate_field(results, "pot_info", "execution_success")

    # Self-consistency-specific.
    row["avg_vote_count"] = avg_field(results, "decoding_stats", "vote_count")
    row["avg_total_samples"] = avg_field(results, "decoding_stats", "total_samples")
    row["avg_confidence"] = avg_field(results, "decoding_stats", "confidence")
    row["aggregator_applied_rate"] = rate_field(results, "decoding_stats", "aggregator_applied")

    # Model / judge config.
    row["temperature"] = model_kwargs.get("temperature")
    row["top_p"] = model_kwargs.get("top_p")
    row["top_k"] = model_kwargs.get("top_k")
    row["enable_thinking"] = chat_template_kwargs.get("enable_thinking")
    row["judge_model"] = data.get("judge_model")

    # Provenance (compact JSON for full traceability).
    row["strategy_kwargs"] = json.dumps(strategy_kwargs, ensure_ascii=False, sort_keys=True)
    row["decoding_kwargs"] = json.dumps(decoding_kwargs, ensure_ascii=False, sort_keys=True)
    row["source_file"] = os.path.relpath(path)
    return row


def discover_files(root: str) -> list[tuple[str, str, str, str]]:
    """Find result JSON files grouped by (model_dir, strategy, decoding).

    Returns a flat list after deduplication:
        (path, model_dir, strategy, decoding)

    * ``self-consistency`` directories hold two files (k=3 and k=5) — both kept.
    * every other directory is expected to hold a single file; if several
      reruns exist, the newest (by filename timestamp) is kept and a warning
      is printed.
    """
    found: list[tuple[str, str, str, str]] = []
    warnings: list[str] = []

    for model_dir in sorted(os.listdir(root)):
        model_path = os.path.join(root, model_dir)
        if not os.path.isdir(model_path):
            continue
        for strategy in sorted(os.listdir(model_path)):
            strat_path = os.path.join(model_path, strategy)
            if not os.path.isdir(strat_path):
                continue
            for decoding in sorted(os.listdir(strat_path)):
                dec_path = os.path.join(strat_path, decoding)
                if not os.path.isdir(dec_path):
                    continue
                files = [
                    os.path.join(dec_path, name)
                    for name in sorted(os.listdir(dec_path))
                    if name.endswith(".json") and not name.endswith(".bak")
                ]
                if not files:
                    continue

                if decoding == "self-consistency":
                    # Keep the newest file per num_samples bucket (k=3, k=5, ...).
                    buckets: dict[int, str] = {}
                    raw: list[tuple[int, str, str]] = []
                    for path in files:
                        try:
                            with open(path, encoding="utf-8") as handle:
                                data = json.load(handle)
                            num_samples = int(
                                (data.get("decoding_kwargs") or {}).get("num_samples", 0)
                            )
                        except (OSError, json.JSONDecodeError):
                            num_samples = 0
                        raw.append((num_samples, file_sort_key(path), path))
                    for num_samples, _key, path in sorted(
                        raw, key=lambda item: (item[0], item[1])
                    ):
                        buckets[num_samples] = path
                    if len(raw) != len(buckets):
                        warnings.append(
                            f"{model_dir}/{strategy}/{decoding}: multiple files share a "
                            f"k value; kept newest per k."
                        )
                    for path in buckets.values():
                        found.append((path, model_dir, strategy, decoding))
                else:
                    files.sort(key=file_sort_key)
                    if len(files) > 1:
                        warnings.append(
                            f"{model_dir}/{strategy}/{decoding}: expected 1 file, "
                            f"found {len(files)}; kept newest ({os.path.basename(files[-1])})."
                        )
                    found.append((files[-1], model_dir, strategy, decoding))

    for warning in warnings:
        print(f"[warn] {warning}")
    return found


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    """Stable ordering: model -> strategy (canonical) -> decoding -> k."""
    strategy_rank = {name: i for i, name in enumerate(STRATEGY_ORDER)}
    decoding_rank = {name: i for i, name in enumerate(DECODING_ORDER)}
    return (
        str(row.get("model_org", "")),
        str(row.get("model_short", "")),
        strategy_rank.get(str(row.get("strategy", "")), len(STRATEGY_ORDER)),
        decoding_rank.get(str(row.get("decoding", "")), len(DECODING_ORDER)),
        int(row.get("k") or 1),
    )


def write_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt(row.get(col)) for col in COLUMNS})


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a compact, human-readable table of the headline numbers."""
    headers = [
        "model_short",
        "strategy",
        "decoding",
        "k",
        "score",
        "llm_judge",
        "prem_f1",
        "avg_tok",
        "avg_llm",
        "failed",
    ]
    widths = {h: len(h) for h in headers}
    lines = []
    for row in rows:
        line = {
            "model_short": str(row.get("model_short", "")),
            "strategy": str(row.get("strategy", "")),
            "decoding": str(row.get("decoding", "")),
            "k": str(row.get("k", "")),
            "score": fmt(row.get("score")),
            "llm_judge": fmt(row.get("llm_judge_score")),
            "prem_f1": fmt(row.get("premises_f1")),
            "avg_tok": fmt(row.get("avg_total_tokens")),
            "avg_llm": fmt(row.get("avg_llm_calls")),
            "failed": fmt(row.get("num_failed")),
        }
        for h in headers:
            widths[h] = max(widths[h], len(line[h]))
        lines.append(line)

    sep = "  "
    header_line = sep.join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for line in lines:
        print(sep.join(line[h].ljust(widths[h]) for h in headers))


def compile_final_results(args: argparse.Namespace) -> None:
    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"results directory not found: {results_dir}")

    output_path = args.output or os.path.join(results_dir, "compiled_results.csv")

    discovered = discover_files(results_dir)
    if not discovered:
        raise SystemExit(f"no result JSON files found under {results_dir}")

    rows: list[dict[str, Any]] = []
    for path, model_dir, strategy, decoding in discovered:
        seed = extract_seed(path) or os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(path)))
        )
        try:
            rows.append(load_row(path, model_dir, strategy, decoding, seed))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] failed to parse {path}: {exc}")

    rows.sort(key=row_sort_key)
    write_csv(rows, output_path)

    print(f"\nCompiled {len(rows)} rows from {len(discovered)} files.")
    print(f"CSV written to: {output_path}\n")
    print_summary(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile ViREx-Bench final results into a comprehensive CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "final_results",
            "seed-0",
            "vietnamese-logical-reasoning",
        ),
        help="Root results directory (model_dirs underneath).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <results-dir>/compiled_results.csv.",
    )
    args = parser.parse_args()

    compile_final_results(args)


if __name__ == "__main__":
    main()
