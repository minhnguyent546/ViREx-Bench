#!/usr/bin/env python
"""Translate the ViREx-Bench logical-reasoning dataset from English to Vietnamese.

The dataset was originally authored in Vietnamese, translated to English for a contest,
and is now translated back to Vietnamese with TranslateGemma served through vLLM.

For every row the `query`, `premises`, and `answer` are bundled into a single piece of text
so the translation model has the full row context — this is especially helpful for answers
like "Yes"/"No"/"Uncertain" that need the question for correct translation. Each segment is
prefixed with a role tag ([Q], [A], [P1], [P2], ...) so the model can track the number of
premises and is less likely to drop one. A stable delimiter (`|||`) separates the bundled
items; because the premise count is known upstream, the translated bundle can be split back
into the three fields afterwards. The `answer_aliases` column is dropped from the output
entirely.

The output JSON file keeps every original column (minus `answer_aliases`) and adds
`query_vi`, `premises_vi` and `answer_vi`, ready to be loaded back and pushed to the
Hugging Face Hub.

Usage:
    uv run python scripts/run_translate_dataset_translategemma.py \\
        --model Infomaniak-AI/vllm-translategemma-4b-it --output_dir data
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel, ConfigDict
from torch.version import __version__ as torch_version
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm import __version__ as vllm_version

BUNDLE_DELIMITER = " ||| "

# Regex to strip the role tags ([Q], [A], [P1], [P2], ...) that `build_bundle` prepends to
# each segment. Applied during parsing so the translated text is clean.
_SEGMENT_LABEL_PATTERN = re.compile(r"^\[(?:Q|A|P\d+)\]\s*")

# Minimal language-code -> human name map; falls back to the raw code. TranslateGemma's
# standard template resolves full names, but the custom prompt below bypasses it, so we
# resolve the names ourselves to keep the instruction close to the model's training format.
_LANGUAGE_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
}

# Domain-aware translation instruction, sent through TranslateGemma's `<<<custom>>>` chat
# template mode. It tells the model the text is logic-based educational content (so domain
# terminology is translated consistently) and asks it to leave the segment labels, option
# letters, and `|||` bundle separators untouched so the fields can be recovered afterwards.
_DOMAIN_TRANSLATION_PROMPT = """\
You are a professional {source_name} ({source_lang}) to {target_name} ({target_lang}) translator.
The text below is from a logic-based educational benchmark. It bundles, in order, a question labeled [Q] (which may include multiple-choice options labeled A, B, C, D), the answer labeled [A], and a list of logical premises labeled [P1], [P2], ... (rules and facts), all separated by " ||| ".
Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to {target_name} grammar, vocabulary, and cultural sensitivities. Keep the logical terminology consistent across the whole bundle.
Leave the segment labels ([Q], [A], [P1], [P2], ...), the option letters (A, B, C, D), and every " ||| " separator unchanged; do not remove, merge, or add any.
Produce only the {target_name} translation, without any additional explanations or commentary. Please translate the following {source_name} text into {target_name}:

{text}""".strip("\n")


class LogicalReasoningRow(BaseModel):
    """Schema of a single row in the logical-reasoning dataset."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    category: str
    query: str
    premises: list[str]
    options: list[str] | None = None
    answer: str
    answer_aliases: list[str]
    unit: str = ""
    premises_used: list[int]


def build_bundle(query: str, premises: list[str], answer: str) -> str:
    """Bundle the query, answer, and premises of a row into one piece of text.

    Each segment is prefixed with a role tag ([Q], [A], [P1], [P2], ...) so the model can
    clearly see how many premises there are and is less likely to drop one during
    translation. The answer is placed right after the question (and before the lengthy
    premises) so the model has the question as direct context when translating it —
    especially helpful for "Yes"/"No"/"Uncertain" answers that need the question for correct
    meaning. Items are joined with `BUNDLE_DELIMITER` so the individual fields can still be
    recovered after translation.
    """
    items: list[str] = [f"[Q] {query}", f"[A] {answer}"]
    for index, premise in enumerate(premises, start=1):
        items.append(f"[P{index}] {premise}")
    return BUNDLE_DELIMITER.join(items)


def parse_bundle(translated: str, num_premises: int) -> tuple[str, list[str], str] | None:
    """Split a translated bundle back into (query, premises, answer).

    Strips the `[Q]`/`[A]`/`[PN]` segment labels added by `build_bundle`. Returns None when
    the recovered item count does not match the expectation, which happens if the
    translation model altered the delimiters.
    """
    parts = [part.strip() for part in translated.split("|||")]
    expected_count = 2 + num_premises
    if len(parts) != expected_count:
        return None
    query = _SEGMENT_LABEL_PATTERN.sub("", parts[0])
    answer = _SEGMENT_LABEL_PATTERN.sub("", parts[1])
    premises = [_SEGMENT_LABEL_PATTERN.sub("", part) for part in parts[2:]]
    return query, premises, answer


def translate_dataset(args: argparse.Namespace) -> None:
    run_start_time = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(
        args.output_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    init_logger(level="DEBUG", log_file_path=log_file_path)

    logger.info(f"Args: {vars(args)}")
    logger.info(f"Torch version: {torch_version} | vLLM version: {vllm_version}")
    nvidia_smi = subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    logger.info(nvidia_smi.stdout)

    logger.info(
        f"Loading dataset {args.dataset_path} "
        f"(config={args.dataset_name}, split={args.dataset_split})"
    )
    hf_dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
    rows: list[dict[str, Any]] = [dict(hf_row) for hf_row in hf_dataset]
    if args.limit is not None:
        rows = rows[: args.limit]
    logger.info(f"Loaded {len(rows)} rows")

    validated_rows = [LogicalReasoningRow.model_validate(row) for row in rows]

    llm = LLM(
        model=args.model,
        dtype="bfloat16" if args.use_bfloat16 else "float32",
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(seed=args.seed, max_tokens=args.max_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    bundles = [build_bundle(row.query, row.premises, row.answer) for row in validated_rows]
    bundle_messages = [
        build_messages(source_lang=args.source_lang, target_lang=args.target_lang, text=bundle)
        for bundle in bundles
    ]
    bundle_prompts = tokenizer.apply_chat_template(
        bundle_messages, tokenize=False, add_generation_prompt=True
    )
    logger.debug(f"Example bundle prompt: {bundle_prompts[0]}")

    start_time = time.perf_counter()
    bundle_outputs = llm.generate(prompts=bundle_prompts, sampling_params=sampling_params)
    bundle_inference_time = time.perf_counter() - start_time
    translated_bundles = [output.outputs[0].text.strip() for output in bundle_outputs]
    logger.info(
        f"Translated {len(translated_bundles)} query+premises+answer bundles "
        f"in {to_hms(bundle_inference_time)}"
    )

    # Build output rows, parsing each translated bundle back into (query, premises, answer).
    output_rows: list[dict[str, Any]] = []
    failed_query_ids: list[str] = []
    for row, translated_bundle in zip(rows, translated_bundles, strict=True):
        parsed = parse_bundle(translated=translated_bundle, num_premises=len(row["premises"]))
        output_row = {key: value for key, value in row.items() if key != "answer_aliases"}
        if parsed is None:
            logger.warning(
                f"Could not reconstruct query+premises+answer for {row['query_id']}; "
                f"keeping the original (English) text for this row"
            )
            logger.debug(f"{translated_bundle = }")
            output_row["query_vi"] = row["query"]
            output_row["premises_vi"] = list(row["premises"])
            output_row["answer_vi"] = row["answer"]
            failed_query_ids.append(row["query_id"])
        else:
            query_vi, premises_vi, answer_vi = parsed
            output_row["query_vi"] = query_vi
            output_row["premises_vi"] = premises_vi
            output_row["answer_vi"] = answer_vi
        output_rows.append(output_row)

    output_file_path = os.path.join(
        args.output_dir, f"{args.dataset_name}_{args.target_lang}.json"
    )
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(output_rows, output_file, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - run_start_time
    num_failed = len(failed_query_ids)
    num_success = len(output_rows) - num_failed
    logger.info("─" * 60)
    logger.info("Translation summary")
    logger.info("─" * 60)
    logger.info(f"  Rows processed          : {len(output_rows)}")
    logger.info(f"  Successfully translated : {num_success}/{len(output_rows)}")
    logger.info(f"  Failed (kept original)  : {num_failed}/{len(output_rows)}")
    if failed_query_ids:
        logger.info(f"  Failed query IDs        : {', '.join(failed_query_ids)}")
    logger.info(f"  Inference time          : {to_hms(bundle_inference_time)}")
    logger.info(f"  Total elapsed time      : {to_hms(total_elapsed)}")
    logger.info(f"  Output file             : {output_file_path}")
    logger.info(f"  Log file                : {log_file_path}")
    logger.info("─" * 60)


def build_messages(source_lang: str, target_lang: str, text: str) -> list[dict[str, Any]]:
    """Build messages for the TranslateGemma chat template using its `<<<custom>>>` mode.

    The custom mode lets us inject a domain-aware instruction (logic-based educational
    content) while keeping the bundle text and the `|||` separators recoverable. Everything
    after `<<<custom>>>` becomes the full user-turn content.
    """
    source_name = _LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang, target_lang)
    prompt = _DOMAIN_TRANSLATION_PROMPT.format(
        source_name=source_name,
        source_lang=source_lang,
        target_name=target_name,
        target_lang=target_lang,
        text=text,
    )
    return [{"role": "user", "content": f"<<<custom>>>{prompt}"}]


def to_hms(seconds: float) -> str:
    """Convert seconds to hours, minutes, seconds format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"


def init_logger(level: str = "DEBUG", log_file_path: str | None = None) -> None:
    logger.remove()
    logger.add(sys.stdout, level=level)
    if log_file_path is not None:
        logger.add(log_file_path, level="DEBUG")


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Hugging Face path of the dataset repository",
        default="minhnguyent546/virex-bench-datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset configuration name",
        default="logical-reasoning",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="Dataset split to translate",
        default="test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where the translated JSON file and log are written",
        default="data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Translate only the first N rows (unset = all rows)",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for random number generators",
        default=1061109567,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to use for translation",
        choices=[
            "Infomaniak-AI/vllm-translategemma-4b-it",
            "Infomaniak-AI/vllm-translategemma-12b-it",
            "Infomaniak-AI/vllm-translategemma-27b-it",
        ],
        default="Infomaniak-AI/vllm-translategemma-4b-it",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        help="Use bfloat16 for inference",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        help="Maximum model length",
        default=8192,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of new tokens to generate for each bundle",
        default=4096,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        help="Target GPU memory utilization for vLLM (between 0 and 1)",
        default=0.85,
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        help="Tensor parallel size for vLLM (number of GPUs to use for tensor parallelism)",
        default=1,
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        help="Language ID of the source text (full list: https://huggingface.co/google/translategemma-4b-it/blob/main/chat_template.jinja)",
        default="en",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        help="Language ID of the target text (full list: https://huggingface.co/google/translategemma-4b-it/blob/main/chat_template.jinja)",
        default="vi",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Translate the ViREx-Bench logical-reasoning dataset using TranslateGemma via vLLM"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)

    args = parser.parse_args()

    translate_dataset(args)


if __name__ == "__main__":
    main()
