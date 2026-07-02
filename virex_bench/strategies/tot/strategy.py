"""Tree-of-Thoughts (ToT) prompting strategy (Yao et al., 2023).

ToT deliberates over a problem by maintaining a *tree of partial reasoning
states* and searching over it, instead of committing to a single chain like
CoT. The search algorithm is pluggable (beam, DFS, MCTS) and is selected via
the CLI composite name (``tot-beam``, ``tot-dfs``, ``tot-mcts``) or, for the
bare ``tot`` name, the ``VIREX_BENCH_TOT_SEARCH_ALGORITHM`` env var.

Each search step performs three operations, each backed by an internal
:class:`dspy.Signature` (these are strategy-internal, not defined by the task —
the task only supplies the final answer contract):

1. **Propose** — from each frontier node, sample several distinct next reasoning
   steps (a thought) given the premises, question, and the reasoning so far.
2. **Evaluate** — score how promising each candidate thought-path is on a 1-10
   scale (averaged over ``n_eval_samples`` independent votes).
3. **Select** — the search algorithm's own policy (e.g. beam: keep the
   ``beam_width`` highest-scoring candidates as the next frontier).

After the search, a final **aggregate** step feeds the best explored thought-path
back into the task signature (with an added ``reasoning_path`` input) so the
model commits to a well-formed answer grounded in the exploration. The recorded
``reasoning`` on the returned prediction is the chosen thought-path, so the
evaluator and judge see the actual ToT trace.

Config knobs are read from environment variables (see :mod:`virex_bench.envs`,
prefix ``VIREX_BENCH_TOT_*``). The LM is picked up from ``dspy.settings.lm``
(set by the evaluator) — per-call ``config`` overrides temperature / ``n`` so no
global state is mutated, keeping the strategy safe under the evaluator's thread
pool.
"""

import dspy
from pydantic.fields import FieldInfo

from virex_bench import envs
from virex_bench.logger import init_logger
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.tot.common import (
    ThoughtEvaluatorSignature,
    ThoughtProposerSignature,
    render_thought_path,
)
from virex_bench.strategies.tot.search import SearchConfig, ThoughtSearch, build_search

logger = init_logger(__name__)


def _resolve_threshold(raw: float | None, default: float) -> float | None:
    """Resolve a ToT threshold env var: unset -> default, ``0`` -> disable (None)."""

    if raw == 0:
        return None
    if raw is None:
        return default
    return raw


def _build_search_config(search_algorithm: str) -> SearchConfig:
    """Populate a :class:`SearchConfig` from the ``VIREX_BENCH_TOT_*`` env vars.

    The propose/evaluate operation knobs (depth, branching, eval samples,
    temperatures, dedupe) are shared across all algorithms. The threshold and
    budget cap are variant-specific -- ``early_stop_threshold`` is stop-on-success
    for beam and ToT's ``v_th`` pruning for DFS; ``success_threshold`` is DFS's
    stop-on-success (analogous to beam's ``early_stop_threshold``);
    ``max_iterations`` caps DFS/MCTS and is unused by beam -- so the right env var
    is selected per ``search_algorithm``.

    Threshold env vars are parsed by ``maybe_convert_float`` (pure converter) and
    return ``None`` when unset/empty, or ``0.0`` when set to ``"0"``. Since the
    thresholds live on the evaluator's 1-10 scale, ``0`` is never a useful real
    threshold -- it is the explicit disable sentinel. ``_resolve_threshold``
    applies the algorithm default when the env is unset, and maps ``0`` to
    ``None`` (disabled) so consumers can keep their existing ``if x is not None``
    guard.
    """
    beam_width: int | None = None
    exploration_constant: float | None = None
    success_threshold: float | None = None
    if search_algorithm == "beam":
        early_stop_threshold = _resolve_threshold(
            envs.VIREX_BENCH_TOT_BEAM_EARLY_STOP_THRESHOLD, 9.0
        )
        max_iterations = None  # beam has a fixed budget (max_depth * beam_width).
        beam_width = envs.VIREX_BENCH_TOT_BEAM_WIDTH
    elif search_algorithm == "dfs":
        early_stop_threshold = _resolve_threshold(envs.VIREX_BENCH_TOT_DFS_PRUNE_THRESHOLD, 3.0)
        max_iterations = envs.VIREX_BENCH_TOT_DFS_MAX_ITERATIONS
        success_threshold = _resolve_threshold(envs.VIREX_BENCH_TOT_DFS_STOP_THRESHOLD, 9.0)
    elif search_algorithm == "mcts":
        # MCTS does not use early_stop_threshold (ToT's v_th pruning is DFS-only);
        # its stop-on-success knob is ``success_threshold`` (analogous to DFS).
        early_stop_threshold = None
        max_iterations = envs.VIREX_BENCH_TOT_MCTS_MAX_ITERATIONS
        exploration_constant = envs.VIREX_BENCH_TOT_MCTS_EXPLORATION_CONSTANT
        mcts_stop_threshold = envs.VIREX_BENCH_TOT_MCTS_STOP_THRESHOLD
        success_threshold = (
            None if mcts_stop_threshold is None else _resolve_threshold(mcts_stop_threshold, 9.0)
        )
    else:
        # build_search validates the algorithm before this is called, so an unknown
        # value here is a programming error, not a user-facing one.
        raise KeyError(f"Unknown search algorithm {search_algorithm!r}")

    return SearchConfig(
        max_depth=envs.VIREX_BENCH_TOT_MAX_DEPTH,
        branching_factor=envs.VIREX_BENCH_TOT_BRANCHING_FACTOR,
        n_eval_samples=envs.VIREX_BENCH_TOT_EVAL_SAMPLES,
        propose_temperature=envs.VIREX_BENCH_TOT_PROPOSE_TEMPERATURE,
        evaluate_temperature=envs.VIREX_BENCH_TOT_EVALUATE_TEMPERATURE,
        early_stop_threshold=early_stop_threshold,
        beam_width=beam_width,
        exploration_constant=exploration_constant,
        max_iterations=max_iterations,
        success_threshold=success_threshold,
    )


class ToTStrategy(ReasoningStrategy):
    """Tree of Thoughts: search over reasoning steps, then commit an answer.

    The search algorithm is delegated to a :class:`ThoughtSearch` instance
    (beam, DFS, or MCTS), selected via the ``variant`` kwarg (CLI composite
    name) or the ``VIREX_BENCH_TOT_SEARCH_ALGORITHM`` env var (bare ``tot``).

    All knobs default from the ``VIREX_BENCH_TOT_*`` env vars. Cost varies by
    algorithm — beam is roughly ``max_depth * beam_width`` proposer calls and
    ``max_depth * beam_width * branching_factor`` evaluator calls per example,
    far higher than CoT's single call, which is the tradeoff ToT exists to study.
    """

    name = "tot"
    accepts_variant = True

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        *,
        variant: str | None = None,
    ) -> None:
        super().__init__(signature, rationale_field, rationale_field_type)
        # CLI variant wins; bare `tot` falls back to the env var (backward compatible).
        self.search_algorithm: str = (
            variant if variant is not None else envs.VIREX_BENCH_TOT_SEARCH_ALGORITHM
        ).lower()
        # Distinct name -> distinct report label + output folder.
        self.name = "tot" if variant is None else f"tot-{variant}"

        self.config = _build_search_config(self.search_algorithm)
        logger.info(f"Search config for {self.name}: {self.config}")
        self.search: ThoughtSearch = build_search(self.search_algorithm, self.config)

        self.propose = dspy.Predict(ThoughtProposerSignature)
        self.evaluate = dspy.Predict(ThoughtEvaluatorSignature)
        # Feed the explored path into the task signature as an extra input so the
        # committed answer reuses the task's full output contract (answer,
        # supporting_premise_indices, ...). `prepend` is the same mechanism the
        # CoT module uses to add its `reasoning` output field.
        aggregator_signature = self.signature.prepend(
            name="reasoning_path",
            field=dspy.InputField(
                desc=(
                    "A chain of reasoning steps already explored over the premises. "
                    "Use it as the basis for the final answer; do not contradict it. "
                    "The path may inspect or reject several candidate conclusions "
                    "before settling on one. When citing premises, ignore that "
                    "exploration: return only the minimal chain that justifies the "
                    "FINAL answer, and exclude any premise the path used solely to "
                    "analyze or rule out other options."
                )
            ),
            type_=str,
        )
        self.aggregate = dspy.Predict(aggregator_signature)

    @property
    def report_config(self) -> dict[str, object]:
        return {
            "search_algorithm": self.search_algorithm,
            "max_depth": self.config.max_depth,
            "branching_factor": self.config.branching_factor,
            "n_eval_samples": self.config.n_eval_samples,
            "propose_temperature": self.config.propose_temperature,
            "evaluate_temperature": self.config.evaluate_temperature,
            "early_stop_threshold": self.config.early_stop_threshold,
            "beam_width": self.config.beam_width,
            "exploration_constant": self.config.exploration_constant,
            "max_iterations": self.config.max_iterations,
            "success_threshold": self.config.success_threshold,
        }

    def forward(self, **inputs: object) -> dspy.Prediction:
        premises = inputs.get("premises")
        question = inputs.get("question")
        if premises is None or question is None:
            raise ValueError(
                "ToTStrategy requires 'premises' and 'question' inputs, "
                f"got keys: {sorted(inputs)}"
            )

        result = self.search.search(
            premises=premises,
            question=question,
            propose=self.propose,
            evaluate=self.evaluate,
        )
        reasoning_path = render_thought_path(result.best_path)
        prediction = self.aggregate(
            premises=premises,
            question=question,
            reasoning_path=reasoning_path,
        )
        prediction["reasoning"] = reasoning_path
        prediction["search_stats"] = {
            "algorithm": self.search_algorithm,
            "nodes_visited": result.nodes_visited,
            "propose_calls": result.propose_calls,
            "evaluate_calls": result.evaluate_calls,
            "depth_reached": result.depth_reached,
            "best_score": result.best_score,
            # Cost in "LLM request" units (one dspy.Predict invocation each), incl.
            # the final aggregate call that commits the answer -- the direct analog
            # of a CoT/direct single call, so total_llm_calls is comparable 1:1.
            "total_llm_calls": result.propose_calls + result.evaluate_calls + 1,
            # Cost in "generated completion" units (what drives GPU/token cost): each
            # propose request samples `branching_factor` completions and each evaluate
            # request samples `n_eval_samples`; the aggregate call yields 1 completion.
            "total_completions": (
                result.propose_calls * self.config.branching_factor
                + result.evaluate_calls * self.config.n_eval_samples
                + 1
            ),
        }
        return prediction
