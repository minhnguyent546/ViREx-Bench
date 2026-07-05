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

# ``single`` = one validity check; ``multi`` = meaningfulness then validity (default).
VerifierMode = Literal["single", "multi"]

# Three-valued verdict derived from the verifier's two booleans;
# ``undetermined`` = neither entailed nor contradicted.
PropositionVerdict = Literal["entailed", "contradicted", "undetermined"]


class PropositionProposerSignature(dspy.Signature):
    """Propose ONE new proposition deduced from the premises.

    You are given the premises (the only source of truth), the question, and the
    propositions already accumulated. Produce a single new proposition that:
      - is deduced STRICTLY from the premises -- no outside knowledge,
      - makes explicit forward progress toward answering the question,
      - is self-contained enough to be read on its own,
      - cites the 1-based premise index/indices it relies on.

    Write in Vietnamese. Do NOT restate a proposition already present in `accumulated_context`.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block; the only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer.")
    accumulated_context: str = dspy.InputField(
        desc="Propositions already accumulated and verified. Empty on the first step."
    )

    next_proposition: str = dspy.OutputField(
        desc=(
            "A single new proposition in Vietnamese, citing the premise indices it uses "
            "(e.g. 'Theo tiền đề 3, ...'). If nothing new can be derived, output '[[ ## next_proposition ## ]]\n\nKHÔNG CÓ MỆNH ĐỀ MỚI' instead."
        )
    )


class PropositionValiditySignature(dspy.Signature):
    """Judge the logical relationship between the premises and a candidate proposition.

    Set is_contradicted = true ONLY when the premises logically force the
    proposition's negation -- not merely because it seems unlikely. Set
    is_entailed = true ONLY when every scenario consistent with the premises
    also satisfies the proposition -- if you found even one counterexample,
    is_entailed MUST be false.

    When both are false, the verdict is UNDETERMINED: the premises neither
    prove nor disprove the proposition. This is a common and valid outcome.
    When in doubt, default to false for both. The bar is "necessarily", not
    "probably".
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block; the only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer.")
    proposition: str = dspy.InputField(desc="The candidate proposition to verify.")
    accumulated_context: str = dspy.InputField(
        desc="Propositions already accumulated. Empty on the first step.",
        default="",
    )

    is_contradicted: bool = dspy.OutputField(
        desc="True if the proposition is necessarily false given the premises (strict logical contradiction)."
    )
    is_entailed: bool = dspy.OutputField(
        desc="True if the proposition is necessarily true given the premises (strict logical entailment)."
    )


class PropositionMeaningfulnessSignature(dspy.Signature):
    """Cheap pre-filter: is this a meaningful, substantive proposition?

    Set is_useful = false when the proposition is:
      - a non-answer or filler ("unknown", "unclear", "no proposition");
      - empty or whitespace;
      - a bare tautology that adds no information;
      - a restatement of the question rather than a deduced fact.

    Otherwise set is_useful = true.
    """

    proposition: str = dspy.InputField(desc="The candidate proposition to screen.")

    is_useful: bool = dspy.OutputField(
        desc="True if the proposition is a substantive, non-filler claim worth verifying; false otherwise."
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
    """Per-call sampling override for the proposer. Empty -> dspy inherits the
    LM's ``--model-kwargs`` profile verbatim (the default)."""

    verify_config: Mapping[str, Any]
    """Per-call sampling override for the verifiers. Same semantics as
    :attr:`propose_config`."""

    n_propose_samples: int
    """Number of candidate propositions sampled per propose call (``n`` parameter).
    > 1 gives the loop multiple diverse candidates to try in order (deduped,
    verified one by one until one is accepted), combating proposer diversity
    exhaustion. Mirrors ToT's branching factor."""

    dedupe_similarity_threshold: float
    """Jaccard threshold above which a new proposition is treated as a duplicate."""

    def __post_init__(self) -> None:
        if self.target_propositions < 1:
            raise ValueError(f"target_propositions must be >= 1, got {self.target_propositions}")
        if self.max_failed_attempts < 1:
            raise ValueError(f"max_failed_attempts must be >= 1, got {self.max_failed_attempts}")
        if self.n_propose_samples < 1:
            raise ValueError(f"n_propose_samples must be >= 1, got {self.n_propose_samples}")
        if self.verifier_mode not in ("single", "multi"):
            raise ValueError(
                f"verifier_mode must be 'single' or 'multi', got {self.verifier_mode!r}"
            )
        if not (0.0 <= self.dedupe_similarity_threshold <= 1.0):
            raise ValueError(
                f"dedupe_similarity_threshold must be between 0 and 1, "
                f"got {self.dedupe_similarity_threshold}"
            )
        # Only the well-known ``temperature`` key is bounds-checked, so other
        # overrides (top_p, top_k, ...) need no special-casing.
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

    Each non-empty bucket gets a Vietnamese header naming the verdict, followed
    by 1-indexed propositions. Empty buckets are omitted; returns "" when all
    three are empty (e.g. the first loop iteration). The headers double as
    semantic cues: "được xác nhận" = entailed, "bị bác bỏ" = contradicted,
    "không xác định" = undetermined.
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
