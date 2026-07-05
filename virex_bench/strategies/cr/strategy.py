"""Cumulative Reasoning (CR) prompting strategy (Zhang et al. 2023).

CR accumulates *verified* intermediate propositions into a growing "cumulative
context," then feeds that context to a Solver that commits the final answer.
We implement the FOLIO/logic variant -- the natural fit for a logical-reasoning
dataset.

The loop is a single linear accumulation (unlike ToT's branching search), so
there is no pluggable ``search/`` subpackage:

1. **Propose** -- sample ``n_propose_samples`` candidate propositions deduced
   from the premises (+ already-accumulated propositions). Candidates are
   deduped within the batch; each is tried in order (filler / duplicate
   candidates are skipped before any verify call) until one passes verification.
2. **Verify** -- three-valued verdict gate: the verifier answers
   ``is_entailed`` / ``is_contradicted`` in a single call, yielding one of
   *entailed*, *contradicted*, *undetermined*. Every verdict is accumulated into
   its bucket; only filler / duplicate / parse-error proposals count as
   failures (parse errors still consume LLM calls -- they just avoid
   incrementing the failed-proposal counter). ``multi`` mode (default) adds a
   meaningfulness pre-filter.
3. Repeat until ``target_propositions`` propositions are accumulated or
   ``max_failed_attempts`` failed propose calls are rejected.
4. **Solve** -- feed premises + the three verdict buckets into the task
   signature (with an added ``accumulated_context`` input). The *undetermined*
   bucket is the positive signal for "Không chắc chắn" (Uncertain).

The recorded ``reasoning`` on the returned prediction is the rendered bucket
block (the same convention ToT uses for its thought path). Config knobs come
from the ``VIREX_BENCH_CR_*`` env vars; per-call ``config`` overrides
temperature but leaves the LM's other sampling params (top_p, top_k, penalties)
inherited from ``--model-kwargs``, so no global state is mutated -- the strategy
stays safe under the evaluator's thread pool.
"""

import dspy
from dspy.adapters.base import AdapterParseError
from pydantic.fields import FieldInfo

from virex_bench import envs
from virex_bench.logger import init_logger
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.cr.common import (
    CRConfig,
    PropositionMeaningfulnessSignature,
    PropositionProposerSignature,
    PropositionValiditySignature,
    PropositionVerdict,
    is_empty_or_none_proposition,
    parse_bool,
    render_verdict_buckets,
)
from virex_bench.strategies.modules import ChainOfThought
from virex_bench.strategies.tot.common import completion_values, dedupe_thoughts

logger = init_logger(__name__)


def _build_cr_config() -> CRConfig:
    """Populate a :class:`CRConfig` from the ``VIREX_BENCH_CR_*`` env vars.

    Bounds are validated in ``CRConfig.__post_init__`` so a bad value fails fast
    at strategy construction.
    """
    propose_temperature = envs.VIREX_BENCH_CR_PROPOSE_TEMPERATURE
    verify_temperature = envs.VIREX_BENCH_CR_VERIFY_TEMPERATURE
    return CRConfig(
        target_propositions=envs.VIREX_BENCH_CR_TARGET_PROPOSITIONS,
        max_failed_attempts=envs.VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS,
        verifier_mode=envs.VIREX_BENCH_CR_VERIFIER_MODE,
        # None -> empty config -> dspy inherits the LM's --model-kwargs profile.
        propose_config={} if propose_temperature is None else {"temperature": propose_temperature},
        verify_config={} if verify_temperature is None else {"temperature": verify_temperature},
        n_propose_samples=envs.VIREX_BENCH_CR_N_PROPOSE_SAMPLES,
        dedupe_similarity_threshold=envs.VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD,
    )


class CRStrategy(ReasoningStrategy):
    """Cumulative Reasoning: accumulate verified propositions, then solve.

    The accumulation loop is single-threaded (no search tree): each proposition
    is proposed, verified, and routed into one of three verdict buckets
    (entailed / contradicted / undetermined). After the loop, a final Solver
    call consumes the bucketed context and commits the answer via the task
    signature. All knobs default from the ``VIREX_BENCH_CR_*`` env vars.

    Cost varies with ``target_propositions``, ``n_propose_samples``, and
    ``verifier_mode`` -- in default ``multi`` mode with ``target_propositions=7``
    and ``n_propose_samples=4``, best case is ~22 LLM calls and ~46 completions
    per example (7 propose + 7 meaningfulness + 7 validity + 1 solve), far
    higher than CoT's single call, which is the tradeoff CR exists to study.
    """

    name = "cr"

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
    ) -> None:
        super().__init__(signature, rationale_field, rationale_field_type)

        self.config = _build_cr_config()
        logger.info(f"CR config for {self.name}: {self.config}")

        self.propose = dspy.Predict(PropositionProposerSignature)
        self.verify_validity = ChainOfThought(
            PropositionValiditySignature,
            rationale_field=dspy.OutputField(
                desc=(
                    "Check for counterexamples: try to find a scenario where the proposition "
                    "is FALSE despite all premises holding. If none exists, it may be entailed. "
                    "Keep it concise. Vietnamese or English."
                )
            ),
        )
        # Built always but only invoked when verifier_mode == "multi".
        self.verify_meaningful = dspy.Predict(PropositionMeaningfulnessSignature)

        aggregator_signature = self.signature.prepend(
            name="accumulated_context",
            field=dspy.InputField(
                desc=(
                    "Propositions verified against the premises, sorted into "
                    "three buckets:\n"
                    "- [Mệnh đề được xác nhận] (entailed) — necessarily TRUE.\n"
                    "- [Mệnh đề bị bác bỏ] (contradicted) — necessarily FALSE.\n"
                    "- [Mệnh đề không xác định] (undetermined) — insufficient "
                    "information to decide.\n\n"
                    "Use the entailed and contradicted facts to answer. If they "
                    "settle the question, answer definitively. If not, answer "
                    "'Không chắc chắn' (Uncertain). Note: 'Không' means the answer "
                    "is provably negative — do not answer 'Không' just because "
                    "propositions are undetermined."
                )
            ),
            type_=str,
        )
        self.aggregate = ChainOfThought(
            aggregator_signature,
            rationale_field=self.rationale_field,
            rationale_field_type=self.rationale_field_type,
        )

    @property
    def report_config(self) -> dict[str, object]:
        # Resolved per-call override dicts, surfaced so the run report shows what was applied.
        return {
            "verifier_mode": self.config.verifier_mode,
            "target_propositions": self.config.target_propositions,
            "max_failed_attempts": self.config.max_failed_attempts,
            "n_propose_samples": self.config.n_propose_samples,
            "propose_config": dict(self.config.propose_config),
            "verify_config": dict(self.config.verify_config),
            "dedupe_similarity_threshold": self.config.dedupe_similarity_threshold,
        }

    def forward(self, **inputs: object) -> dspy.Prediction:
        premises = inputs.get("premises")
        question = inputs.get("question")
        if premises is None or question is None:
            raise ValueError(
                f"CRStrategy requires 'premises' and 'question' inputs, got keys: {sorted(inputs)}"
            )

        config = self.config
        entailed: list[str] = []
        contradicted: list[str] = []
        undetermined: list[str] = []
        failed = 0
        propose_calls = 0
        propose_completions = 0
        validity_calls = 0
        meaningfulness_calls = 0
        overflowed = False

        # Merge ``n`` into the propose config once; only set when > 1 so the
        # n=1 path stays identical to the legacy single-proposition call.
        n_samples = config.n_propose_samples
        base_propose_config = dict(config.propose_config)
        if n_samples > 1:
            base_propose_config["n"] = n_samples

        try:
            while (
                len(entailed) + len(contradicted) + len(undetermined) < config.target_propositions
                and failed < config.max_failed_attempts
            ):
                try:
                    proposed = self.propose(
                        premises=premises,
                        question=question,
                        accumulated_context=render_verdict_buckets(
                            entailed, contradicted, undetermined
                        ),
                        config=base_propose_config,
                    )
                except AdapterParseError as error:
                    # Recoverable: count as a failed attempt and retry.
                    propose_calls += 1
                    propose_completions += n_samples
                    failed += 1
                    logger.debug(
                        f"CR: proposer emitted unparseable response "
                        f"(failed={failed}/{config.max_failed_attempts}): {error}"
                    )
                    continue
                propose_calls += 1
                propose_completions += n_samples

                # Collect candidates: when n > 1, dspy populates ``.completions``
                # with a list; when n == 1, fall back to the single field.
                candidates = completion_values(proposed, "next_proposition")
                if not candidates:
                    candidates = [str(proposed.next_proposition).strip()]
                # Dedupe candidates against each other within this batch.
                candidates = dedupe_thoughts(
                    candidates, similarity_threshold=config.dedupe_similarity_threshold
                )

                # Try candidates in order until one is accepted. Filler and
                # near-duplicate rejections are free (no LLM call); only the
                # meaningfulness + validity checks cost.
                accepted = False
                for candidate in candidates:
                    candidate = candidate.strip()
                    if is_empty_or_none_proposition(candidate):
                        continue

                    all_accumulated = [*entailed, *contradicted, *undetermined]
                    if all_accumulated:
                        rededuped = dedupe_thoughts(
                            [*all_accumulated, candidate],
                            similarity_threshold=config.dedupe_similarity_threshold,
                        )
                        if len(rededuped) == len(all_accumulated):
                            continue

                    (
                        delta_validity,
                        delta_meaningfulness,
                        verdict,
                    ) = self._verify(
                        premises=premises,
                        question=question,
                        proposition=candidate,
                        accumulated_context=render_verdict_buckets(
                            entailed, contradicted, undetermined
                        ),
                    )
                    validity_calls += delta_validity
                    meaningfulness_calls += delta_meaningfulness
                    if verdict is None:
                        continue
                    if verdict == "entailed":
                        entailed.append(candidate)
                    elif verdict == "contradicted":
                        contradicted.append(candidate)
                    else:
                        undetermined.append(candidate)
                    accepted = True
                    break

                if not accepted:
                    failed += 1
        except dspy.ContextWindowExceededError:
            # Context grows monotonically; on long examples it may overflow
            # mid-loop. Bail with what we have -- same policy ToT uses.
            overflowed = True
            logger.debug(
                f"CR: context window exceeded after "
                f"{len(entailed) + len(contradicted) + len(undetermined)} propositions; "
                f"solving with accumulated context so far"
            )

        total_accumulated = len(entailed) + len(contradicted) + len(undetermined)
        logger.debug(
            f"CR loop done: accumulated {total_accumulated}/{config.target_propositions} "
            f"propositions (entailed={len(entailed)}, contradicted={len(contradicted)}, "
            f"undetermined={len(undetermined)}; {propose_calls} propose, "
            f"{validity_calls} validity, {meaningfulness_calls} meaningfulness calls, "
            f"overflow={overflowed})"
        )

        accumulated_context = render_verdict_buckets(entailed, contradicted, undetermined)
        prediction = self.aggregate(
            premises=premises,
            question=question,
            accumulated_context=accumulated_context,
        )
        prediction["reasoning"] = accumulated_context
        prediction["search_stats"] = {
            "algorithm": "cr",
            "verifier_mode": config.verifier_mode,
            "nodes_visited": total_accumulated,
            "propose_calls": propose_calls,
            # Map verify calls into evaluate_calls so per-strategy cost tables
            # stay comparable (ToT reports its evaluator calls here too).
            "evaluate_calls": validity_calls + meaningfulness_calls,
            "validity_calls": validity_calls,
            "meaningfulness_calls": meaningfulness_calls,
            "entailed_count": len(entailed),
            "contradicted_count": len(contradicted),
            "undetermined_count": len(undetermined),
            "depth_reached": total_accumulated,
            "best_score": float(total_accumulated >= 1),
            "context_overflow": overflowed,
            # "LLM request" units incl. the final aggregate call (the direct
            # analog of CoT/direct's single call, comparable 1:1).
            "total_llm_calls": propose_calls + validity_calls + meaningfulness_calls + 1,
            # Completions: each propose call yields n_samples completions; verify
            # and aggregate calls yield 1 each.
            "total_completions": propose_completions + validity_calls + meaningfulness_calls + 1,
        }
        return prediction

    def _verify(
        self,
        *,
        premises: object,
        question: object,
        proposition: str,
        accumulated_context: str,
    ) -> tuple[int, int, PropositionVerdict | None]:
        """Run the verifier gate, returning ``(validity_calls, meaningfulness_calls, verdict)``.

        ``verdict`` is ``None`` when the proposition is rejected (failed the
        meaningfulness pre-filter, or a verifier emitted an unparseable
        response); ``forward`` increments ``failed`` in that case. Otherwise
        ``verdict`` is one of ``"entailed"``, ``"contradicted"``, or
        ``"undetermined"`` -- all three are accumulated evidence and do NOT
        count as failures. ``AdapterParseError`` from either verifier is
        recovered (not propagated): the LLM call is counted, then the
        proposition is conservatively rejected -- the same default-false stance
        ``parse_bool`` uses, and the same recovery policy the proposer loop
        applies.
        """
        config = self.config
        meaningfulness_calls = 0
        validity_calls = 0

        if config.verifier_mode == "multi":
            try:
                meaningful = self.verify_meaningful(
                    proposition=proposition,
                    config=config.verify_config,
                )
            except AdapterParseError as error:
                meaningfulness_calls += 1
                logger.debug(
                    f"CR: meaningfulness verifier emitted unparseable response "
                    f"(rejecting proposition): {error}"
                )
                return validity_calls, meaningfulness_calls, None
            meaningfulness_calls += 1

            if not parse_bool(meaningful.is_useful):
                return validity_calls, meaningfulness_calls, None

        try:
            result = self.verify_validity(
                premises=premises,
                question=question,
                proposition=proposition,
                accumulated_context=accumulated_context,
                config=config.verify_config,
            )
        except AdapterParseError as error:
            validity_calls += 1
            logger.debug(
                f"CR: validity verifier emitted unparseable response "
                f"(rejecting proposition): {error}"
            )
            return validity_calls, meaningfulness_calls, None
        validity_calls += 1

        is_entailed = parse_bool(result.is_entailed)
        is_contradicted = parse_bool(result.is_contradicted)
        if is_entailed and not is_contradicted:
            verdict: PropositionVerdict = "entailed"
        elif is_contradicted and not is_entailed:
            verdict = "contradicted"
        else:
            # Neither (undetermined) or both (inconsistent premises).
            verdict = "undetermined"
        return validity_calls, meaningfulness_calls, verdict
