"""Shared signatures + helpers for the Cumulative Reasoning (CR) strategy.

CR (Zhang et al. 2023, arXiv:2308.04371) accumulates *verified* intermediate
propositions into a growing "cumulative context," then feeds that context to a
Solver that commits the final answer. We implement the FOLIO/logic variant --
the natural fit for a logical-reasoning dataset and CR's flagship result.

This module hosts the strategy-internal dspy.Signature contracts (proposer,
verifier) and a few pure-Python helpers. It mirrors :mod:`virex_bench.strategies.tot.common`
in spirit: propositions are near-identical in role to ToT thoughts, so the
dedupe primitives (``dedupe_thoughts`` / ``jaccard_similarity``) are imported
from there rather than reimplemented.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import dspy

from virex_bench.logger import init_logger

logger = init_logger(__name__)

# Verifier modes: ``single`` = one validity check; ``multi`` = a meaningfulness
# check followed by a validity check (default -- the more robust gate).
VerifierMode = Literal["single", "multi"]

# Three-valued verdict derived from the two-boolean verifier
# (:class:`PropositionValiditySignature`). ``undetermined`` = neither entailed
# nor contradicted (or both, when premises are inconsistent).
PropositionVerdict = Literal["entailed", "contradicted", "undetermined"]


class PropositionProposerSignature(dspy.Signature):
    """Propose ONE new proposition deduced from the premises.

    You are given the premises (the only source of truth), the question, and the
    propositions already accumulated. Produce a single new proposition that:
      - is deduced STRICTLY from the premises (and any already-accumulated
        propositions) -- no outside knowledge, no closed-world assumptions,
      - makes explicit forward progress toward answering the question,
      - is self-contained enough to be read on its own,
      - cites the 1-based premise index/indices it relies on.

    Be concise: at most three short sentences. Do not include hidden thinking
    traces or alternative branches.

    Do NOT restate a proposition already present in `accumulated_context`. Write
    the proposition in Vietnamese to match the task language.

    ALWAYS emit the `proposition` output field, even when you cannot derive
    anything new -- in that case set `proposition` to the literal string
    "KHÔNG CÓ MỆNH ĐỀ MỚI" (Vietnamese for "no new proposition"). Never omit the
    `proposition` field label or write the sentinel as a bare line.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block (Premise 1, Premise 2, ...); the only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer.")
    accumulated_context: str = dspy.InputField(
        desc=(
            "Propositions already accumulated and verified, organized by verdict "
            "bucket. Empty on the first step."
        )
    )
    proposition: str = dspy.OutputField(
        desc=(
            "A single new proposition in Vietnamese, citing the premise indices it uses "
            "(e.g. 'Theo tiền đề 3, ...'). If nothing new can be derived, output "
            "'KHÔNG CÓ MỆNH ĐỀ MỚI'."
        )
    )


class PropositionValiditySignature(dspy.Signature):
    """Judge the logical relationship between the premises and a candidate proposition.

    You will write a counterexample search into the `reasoning` field FIRST
    (see its description), then commit two booleans anchored in that analysis.
    "Undetermined" is never a button you actively press -- it is the natural
    result when BOTH booleans are "no".

    ### CONTRADICTION (is_contradicted)
    Is the `proposition` NECESSARILY FALSE given the premises? Set
    is_contradicted = true ONLY when the premises logically force the
    proposition's negation. Do NOT set it merely because the proposition seems
    unlikely or improbable.

    ### ENTAILMENT (is_entailed)
    Is the `proposition` NECESSARILY TRUE given ONLY the stated premises and
    any already-accepted propositions in `accumulated_context`? Set
    is_entailed = true ONLY when every scenario consistent with the premises
    also satisfies the proposition. If you found even ONE falsifying scenario
    in your `reasoning`, is_entailed MUST be false.

    The four possible answer combinations and their meaning:
    - is_entailed=false, is_contradicted=false => UNDETERMINED.
          The premises neither prove nor disprove the proposition. This is a
          common and valid outcome -- it is the correct signal that the question
          may be unanswerable ("Không chắc chắn"). Set both to false whenever
          you cannot prove or disprove the proposition from the premises alone.
    - is_entailed=true, is_contradicted=false => ENTAILED.
    - is_entailed=false, is_contradicted=true => CONTRADICTED.
    - is_entailed=true, is_contradicted=true => Inconsistent premises (rare).

    When in doubt, default to false for both -- undetermined is always a safe
    and honest verdict. The bar for each flag is "necessarily", not "probably".
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block; the only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer.")
    proposition: str = dspy.InputField(desc="The candidate proposition to verify.")
    accumulated_context: str = dspy.InputField(
        desc=(
            "Propositions already accumulated, organized by verdict bucket. "
            "Empty on the first step."
        ),
        default="",
    )
    is_contradicted: bool = dspy.OutputField(
        desc=(
            "True if the proposition is necessarily false given the premises "
            "(strict logical contradiction); false otherwise. Consistent with "
            "your `reasoning` analysis."
        )
    )
    is_entailed: bool = dspy.OutputField(
        desc=(
            "True if the proposition is necessarily true given the premises "
            "(strict logical entailment); false if you found any counterexample "
            "in `reasoning`."
        )
    )


class PropositionMeaningfulnessSignature(dspy.Signature):
    """Cheap pre-filter: is this a meaningful, substantive proposition?

    Used only in ``multi`` verifier mode, ahead of the (heavier) validity check.
    ACCEPT (``is_meaningful = true``) when `proposition` is a substantive new
    claim that could in principle advance the reasoning.

    REJECT (``is_meaningful = false``) when `proposition` is any of:
      - a non-answer / filler ("there is no proposition", "unknown", "unclear");
      - empty or whitespace;
      - a bare tautology that adds no information ("a thing is itself");
      - a restatement of the question rather than a deduced fact.
    """

    proposition: str = dspy.InputField(desc="The candidate proposition to screen.")
    is_meaningful: bool = dspy.OutputField(
        desc="True if the proposition is a substantive, non-filler claim worth verifying."
    )


@dataclass
class CRConfig:
    """Knobs for the CR accumulation loop.

    All fields have a 1:1 ``VIREX_BENCH_CR_*`` env var (see :mod:`virex_bench.envs`).
    Validation in :meth:`__post_init__` keeps the bounds explicit so a bad env
    value fails fast at strategy construction, not mid-loop.
    """

    target_propositions: int
    """Cap on accepted propositions accumulated before the Solver runs."""

    max_failed_attempts: int
    """Cap on consecutively-rejected (or duplicate) proposals before giving up."""

    verifier_mode: VerifierMode
    """``single`` = one validity check; ``multi`` = meaningfulness then validity."""

    propose_config: Mapping[str, Any]
    """Per-call sampling override for the proposer. Empty mapping -> dspy
    inherits the LM's ``--model-kwargs`` profile verbatim (the default; modern
    cards like Qwen3's forbid greedy decoding and rely on a tuned temp+penalty
    system). A populated dict forces only the keys it lists -- dspy merges it on
    top of the LM kwargs, so other params (top_p, top_k, penalties) persist."""

    verify_config: Mapping[str, Any]
    """Per-call sampling override for the verifiers. Same inheritance semantics
    as :attr:`propose_config`. Not forced to ``temperature=0`` -- the multi-check
    verifier ensemble handles robustness instead."""

    dedupe_similarity_threshold: float
    """Jaccard threshold above which a new proposition is treated as a duplicate."""

    def __post_init__(self) -> None:
        if self.target_propositions < 1:
            raise ValueError(f"target_propositions must be >= 1, got {self.target_propositions}")
        if self.max_failed_attempts < 1:
            raise ValueError(f"max_failed_attempts must be >= 1, got {self.max_failed_attempts}")
        if self.verifier_mode not in ("single", "multi"):
            raise ValueError(
                f"verifier_mode must be 'single' or 'multi', got {self.verifier_mode!r}"
            )
        if not (0.0 <= self.dedupe_similarity_threshold <= 1.0):
            raise ValueError(
                f"dedupe_similarity_threshold must be between 0 and 1, "
                f"got {self.dedupe_similarity_threshold}"
            )
        # Validate a ``temperature`` key if either config carries one. Kept loose
        # (only the well-known sampling key is bounds-checked) so the configs can
        # freely hold other overrides (top_p, top_k, ...) without special-casing.
        for role, override in (("propose", self.propose_config), ("verify", self.verify_config)):
            if "temperature" in override:
                temperature = override["temperature"]
                if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
                    raise ValueError(
                        f"{role}_config['temperature'] must be a number in [0, 2], "
                        f"got {temperature!r}"
                    )


# --- pure-Python helpers ---


_TRUTHY = {"true", "yes", "1", "t", "y"}
_FALSY = {"false", "no", "0", "f", "n", ""}


def parse_bool(raw: object) -> bool:
    """Parse a verifier's ``is_entailed`` / ``is_contradicted`` / ``is_meaningful`` output tolerantly.

    DSPy-typed ``bool`` outputs usually arrive as Python ``bool`` already, but
    models occasionally emit strings ("True", "yes", "False", "0"). We accept
    the common truthy/falsy spellings (case-insensitive); anything unparseable
    is treated as ``False`` (the conservative verdict for a verifier gate -- a
    rejected proposition never poisons the cumulative context).
    """
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    # "is_entailed: true" or similar -- peel off a leading label if present.
    match = re.search(r"\b(true|yes|1|false|no|0)\b", text)
    if match is None:
        logger.debug(f"Could not parse verifier bool from {raw!r}; defaulting to False")
        return False
    return match.group() in _TRUTHY


# Vietnamese + English sentinel phrases the proposer emits when it has nothing
# new to add. Matched as a cheap Python pre-filter before any LLM verifier call
# (mirrors FOLIO's ``is_something`` idea). Kept lowercase; matched by substring.
_EMPTY_PROPOSITION_MARKERS = (
    "không có mệnh đề",  # "no proposition" (Vietnamese)
    "không có mệnh đề mới",
    "không thể suy luận",  # "cannot deduce"
    "không thể rút ra",
    "no proposition",
    "no new proposition",
    "nothing new",
    "cannot derive",
    "không có thông tin",
)


def is_empty_or_none_proposition(text: object) -> bool:
    """Heuristic sentinel check for a "no further proposition" proposal.

    Returns True when ``text`` is empty/whitespace or matches one of the
    Vietnamese/English filler markers the proposer is instructed to emit when
    nothing new can be deduced. Used to short-circuit the verifier (a filler
    proposal should not cost an LLM verify call).
    """
    if text is None:
        return True
    normalized = str(text).strip().lower()
    if normalized == "":
        return True
    return any(marker in normalized for marker in _EMPTY_PROPOSITION_MARKERS)


def render_verdict_buckets(
    entailed: Sequence[str],
    contradicted: Sequence[str],
    undetermined: Sequence[str],
) -> str:
    """Render the three verdict buckets into a section-labeled block.

    Each non-empty bucket is rendered with a Vietnamese header naming the
    verdict, followed by 1-indexed propositions. Empty buckets are omitted.
    Returns an empty string when all three buckets are empty (e.g. the first
    loop iteration).

    The headers double as semantic cues: "được xác nhận" = entailed (necessarily
    true), "bị bác bỏ" = contradicted (necessarily false), "không xác định" =
    undetermined (the premises neither prove nor disprove).
    """
    sections: list[str] = []
    for label, items in (
        ("Mệnh đề được xác nhận", entailed),
        ("Mệnh đề bị bác bỏ", contradicted),
        ("Mệnh đề không xác định", undetermined),
    ):
        if items:
            body = "\n".join(
                f"[{label} {index}] {proposition}"
                for index, proposition in enumerate(items, start=1)
            )
            sections.append(body)
    return "\n\n".join(sections)
