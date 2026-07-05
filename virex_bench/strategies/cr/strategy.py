"""Cumulative Reasoning (CR) prompting strategy (Zhang et al. 2023).

CR accumulates *verified* intermediate propositions into a growing "cumulative
context," then feeds that context to a Solver that commits the final answer.
We implement the FOLIO/logic variant -- the natural fit for a logical-reasoning
dataset and CR's flagship result (~98% on FOLIO).

The loop is a single linear accumulation (unlike ToT's branching search), so
there is no pluggable ``search/`` subpackage:

1. **Propose** -- sample one candidate proposition deduced from the premises
   (+ already-accumulated propositions).
2. **Verify** -- three-valued verdict gate. The verifier answers two independent
   boolean questions (``is_entailed``, ``is_contradicted``) in a single call;
   their combination yields one of three verdicts: *entailed*, *contradicted*,
   or *undetermined* (neither -- the premises don't settle it). Every verdict is
   accumulated into its respective bucket; only filler / duplicate / parse-error
   proposals count as failures. Two modes:
   - ``single``: one validity check.
   - ``multi`` (default): a meaningfulness pre-filter then a validity check --
     more robust, at the cost of one extra LLM call per proposal.
3. Repeat until ``target_propositions`` total propositions are accumulated or
   ``max_failed_attempts`` filler/duplicate proposals are rejected.
4. **Solve** -- feed premises + the three verdict buckets into the task
   signature (with an added ``accumulated_context`` input) so the model commits
   a well-formed answer. The *undetermined* bucket is the positive signal for
   "Không chắc chắn" (Uncertain) -- it tells the solver the premises may not
   settle the question.

The recorded ``reasoning`` on the returned prediction is the rendered bucket
block, so the evaluator and judge see the actual CR trace -- the same convention
ToT uses for its thought path.

Config knobs are read from environment variables (see :mod:`virex_bench.envs`,
prefix ``VIREX_BENCH_CR_*``). The LM is picked up from ``dspy.settings.lm``
(set by the evaluator) -- per-call ``config`` overrides temperature so no global
state is mutated, keeping the strategy safe under the evaluator's thread pool.
Per-role temperatures default to ``None`` (inherit the LM's ``--model-kwargs``
profile); dspy merges per-call config on top of those kwargs, so even when an
override is set the other sampling params (top_p, top_k, penalties) persist --
only ``temperature`` (and ``n``) is forced. See the strategy plan's "Sampling
params" section for the rationale.
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
from virex_bench.strategies.tot.common import dedupe_thoughts

logger = init_logger(__name__)


def _build_cr_config() -> CRConfig:
    """Populate a :class:`CRConfig` from the ``VIREX_BENCH_CR_*`` env vars.

    Bounds are validated in ``CRConfig.__post_init__`` so a bad value fails fast
    at strategy construction rather than mid-loop.
    """
    propose_temperature = envs.VIREX_BENCH_CR_PROPOSE_TEMPERATURE
    verify_temperature = envs.VIREX_BENCH_CR_VERIFY_TEMPERATURE
    return CRConfig(
        target_propositions=envs.VIREX_BENCH_CR_TARGET_PROPOSITIONS,
        max_failed_attempts=envs.VIREX_BENCH_CR_MAX_FAILED_ATTEMPTS,
        verifier_mode=envs.VIREX_BENCH_CR_VERIFIER_MODE,
        # None -> empty config -> dspy inherits the LM's --model-kwargs profile
        # verbatim (see the strategy plan's "Sampling params" section).
        propose_config={} if propose_temperature is None else {"temperature": propose_temperature},
        verify_config={} if verify_temperature is None else {"temperature": verify_temperature},
        dedupe_similarity_threshold=envs.VIREX_BENCH_CR_DEDUPE_SIMILARITY_THRESHOLD,
    )


class CRStrategy(ReasoningStrategy):
    """Cumulative Reasoning: accumulate verified propositions, then solve.

    The accumulation loop is single-threaded (no search tree): one proposition
    is proposed, verified, and routed into one of three verdict buckets
    (entailed / contradicted / undetermined). After the loop, a final Solver
    call consumes the bucketed context and commits the answer via the task
    signature.

    All knobs default from the ``VIREX_BENCH_CR_*`` env vars. Cost varies with
    ``target_propositions`` and ``verifier_mode`` -- in default ``multi`` mode
    with ``target_propositions=7``, best case is ~22 LLM calls per example (7
    propose + 7 meaningfulness + 7 validity + 1 solve), far higher than CoT's
    single call, which is the tradeoff CR exists to study.
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
        # Report the resolved per-call override dicts. An empty dict means
        # "inherit the LM's --model-kwargs profile" -- the inherit-by-default
        # behavior, surfaced explicitly so a run report shows what was applied.
        return {
            "verifier_mode": self.config.verifier_mode,
            "target_propositions": self.config.target_propositions,
            "max_failed_attempts": self.config.max_failed_attempts,
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
        validity_calls = 0
        meaningfulness_calls = 0
        overflowed = False

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
                        config=config.propose_config,
                    )
                except AdapterParseError as error:
                    # Recoverable: count as a failed attempt and retry.
                    propose_calls += 1
                    failed += 1
                    logger.debug(
                        f"CR: proposer emitted unparseable response "
                        f"(failed={failed}/{config.max_failed_attempts}): {error}"
                    )
                    continue
                propose_calls += 1
                proposition = str(proposed.next_proposition).strip()

                # Cheap Python-side rejection: filler / "nothing new" proposals
                # short-circuit before any LLM verifier call.
                if is_empty_or_none_proposition(proposition):
                    failed += 1
                    continue

                # Near-duplicate of an already-accumulated proposition -> reject.
                # ``dedupe_thoughts`` keeps first occurrences, so appending the
                # candidate and re-deduping drops it iff it is a near-duplicate
                # of something already accepted across all three buckets.
                all_accumulated = [*entailed, *contradicted, *undetermined]
                if all_accumulated:
                    rededuped = dedupe_thoughts(
                        [*all_accumulated, proposition],
                        similarity_threshold=config.dedupe_similarity_threshold,
                    )
                    if len(rededuped) == len(all_accumulated):
                        failed += 1
                        continue

                # Verifier gate -> three-valued verdict (None on rejection).
                (
                    delta_validity,
                    delta_meaningfulness,
                    verdict,
                ) = self._verify(
                    premises=premises,
                    question=question,
                    proposition=proposition,
                    accumulated_context=render_verdict_buckets(
                        entailed, contradicted, undetermined
                    ),
                )
                validity_calls += delta_validity
                meaningfulness_calls += delta_meaningfulness
                if verdict is None:
                    failed += 1
                elif verdict == "entailed":
                    entailed.append(proposition)
                elif verdict == "contradicted":
                    contradicted.append(proposition)
                else:
                    undetermined.append(proposition)
        except dspy.ContextWindowExceededError:
            # The accumulated context grows monotonically; on long examples it
            # may overflow the model window mid-loop. Bail with what we have --
            # same graceful-degradation policy ToT's beam search uses.
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
            # Map verify calls into the evaluate_calls slot so per-strategy cost
            # tables stay comparable (ToT reports its evaluator calls here too).
            "evaluate_calls": validity_calls + meaningfulness_calls,
            "validity_calls": validity_calls,
            "meaningfulness_calls": meaningfulness_calls,
            "entailed_count": len(entailed),
            "contradicted_count": len(contradicted),
            "undetermined_count": len(undetermined),
            "depth_reached": total_accumulated,
            "best_score": float(total_accumulated >= 1),
            "context_overflow": overflowed,
            # Cost in "LLM request" units (one dspy.Predict invocation each),
            # incl. the final aggregate call -- the direct analog of a CoT/direct
            # single call, so total_llm_calls is comparable 1:1.
            "total_llm_calls": propose_calls + validity_calls + meaningfulness_calls + 1,
            # CR proposes one completion per call and each verifier yields one
            # completion; the aggregate call yields 1 completion. (No `n`
            # sampling -- unlike ToT's branching proposer/evaluator.)
            "total_completions": propose_calls + validity_calls + meaningfulness_calls + 1,
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
        response). A ``None`` verdict increments ``failed`` in ``forward``.

        Otherwise ``verdict`` is one of ``"entailed"``, ``"contradicted"``, or
        ``"undetermined"`` -- all three are accumulated evidence (into their
        respective buckets) and do NOT count as failures.

        - ``multi`` mode: a meaningfulness pre-filter then the validity check.
        - ``single`` mode: just the validity check.

        An ``AdapterParseError`` from either verifier (e.g. the LM emitting the
        field label ``is_mean`` instead of ``is_meaningful``) is recovered
        rather than propagated: the call still consumed an LLM request so it is
        counted, then the proposition is conservatively rejected -- the same
        default-false stance ``parse_bool`` uses, and the same recovery policy
        the proposer loop applies. A rejection increments ``failed`` in
        ``forward``, so ``max_failed_attempts`` still bounds how long a run of
        unparseable verifier responses can churn.
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
