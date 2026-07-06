"""Program-of-Thought + Z3 symbolic reasoning strategy.

PoT-Z3 asks the LM to translate the premises into an executable Python program
against a pre-loaded Z3 symbolic API, runs it in a sandboxed interpreter, then
feeds the program + its output to a commit step (the task signature with a
``solver_result`` input prepended) that renders the final Vietnamese answer.

The pipeline has three LLM-touched stages:

1. **Generate** -- ``ChainOfThought(ProgramGenerationSignature)`` produces an
   executable Python program encoding the premises as Z3 formulas.
2. **Execute** -- :class:`LocalZ3PythonInterpreter.execute` runs the program in a
   spawned subprocess with a wall-clock timeout. On failure (parse error,
   syntax error, timeout, runtime crash), **regenerate** up to ``max_iters``
   times, feeding ``previous_code`` + ``error`` back to the LM.
3. **Commit** -- ``ChainOfThought(task_signature + prepended solver_result)``
   reads the Z3 program output and emits the canonical task fields in the exact
   Vietnamese format the judge expects.

Error-handling boundary (two distinct failure modes):

- **Code-execution failure** (program does not run, or runs but errors):
  ``parse_code`` returns an error, or ``interpreter.execute`` raises
  ``SyntaxError`` / ``TimeoutError`` / ``RuntimeError``. These trigger the
  regenerate loop. This is the core PoT mechanism.
- **LLM parse/transient failure** (dspy can't parse the model output):
  ``self.generate`` / ``self.regenerate`` / ``self.aggregate`` raise
  ``dspy.AdapterParseError`` or ``dspy.ContextWindowExceededError``. These do
  NOT trigger regeneration -- they propagate out of ``forward`` (the eval
  harness handles them like any other strategy failure).

Config knobs come from the ``VIREX_BENCH_POT_*`` env vars.
"""

import dspy
from pydantic.fields import FieldInfo

from virex_bench.logger import init_logger
from virex_bench.strategies.base import ReasoningStrategy
from virex_bench.strategies.modules import ChainOfThought
from virex_bench.strategies.pot.common import (
    POT_GENERATE_RATIONALE,
    ProgramGenerationSignature,
    ProgramRegenerationSignature,
    build_pot_config,
    format_solver_result,
    parse_code,
    strip_ansi,
)
from virex_bench.strategies.pot.interpreter import LocalZ3PythonInterpreter

logger = init_logger(__name__)


class PoTZ3Strategy(ReasoningStrategy):
    """Program-of-Thought with Z3 symbolic execution.

    The strategy generates a Z3 program from the premises, executes it in a
    sandboxed interpreter, and feeds the result to a commit step that formats
    the final Vietnamese answer. On execution failure, the program is
    regenerated up to ``max_iters`` times with the error fed back.

    Cost: best case 2 LLM calls (generate + commit); each regeneration adds one
    generate + one execute. The interpreter execution is local (no LM call).
    """

    name = "pot_z3"
    accepts_variant = False

    def __init__(
        self,
        signature: type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
    ) -> None:
        super().__init__(signature, rationale_field, rationale_field_type)

        self.config = build_pot_config()
        logger.info(f"PoT-Z3 config: {self.config}")

        self.interpreter = LocalZ3PythonInterpreter(timeout=self.config.execution_timeout)

        self.generate = ChainOfThought(
            ProgramGenerationSignature,
            rationale_field=POT_GENERATE_RATIONALE,
            rationale_field_type=str,
        )
        self.regenerate = dspy.Predict(ProgramRegenerationSignature)

        aggregator_signature = self.signature.prepend(
            name="solver_result",
            field=dspy.InputField(
                desc=(
                    "Output of a Z3 symbolic-solver program run over the premises.\n"
                    "It contains: `answer` (the solver's best-guess for the question "
                    "type), `z3_status` (entailed / contradicted / uncertain / mixed), "
                    "`supporting_premises` (1-based premise indices), and `solution` "
                    "(a one-sentence explanation).\n\n"
                    "Map the Z3 verdict to the Vietnamese answer label:\n"
                    "- entailed -> 'Có' (Yes)\n"
                    "- contradicted -> 'Không' (No)\n"
                    "- uncertain -> 'Không chắc chắn' (Uncertain)\n"
                    "For multiple-choice, use the entailed option labels from the "
                    "program output. For numeric/open-ended, use the solver's derived "
                    "value. If the program failed, reason over the premises directly."
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
        return {
            "max_iters": self.config.max_iters,
            "execution_timeout": self.config.execution_timeout,
            "fallback_on_error": self.config.fallback_on_error,
            "generate_config": dict(self.config.generate_config),
            "regenerate_config": dict(self.config.regenerate_config),
        }

    def forward(self, **inputs: object) -> dspy.Prediction:
        premises = inputs.get("premises")
        question = inputs.get("question")
        if premises is None or question is None:
            raise ValueError(
                f"PoTZ3Strategy requires 'premises' and 'question' inputs, "
                f"got keys: {sorted(inputs)}"
            )

        config = self.config

        code = ""
        output: str | None = None
        error: str | None = None
        execute_calls = 0
        regenerate_calls = 0

        gen = self.generate(
            premises=premises,
            question=question,
            config=config.generate_config,
        )
        parsed, parse_error = parse_code(gen.generated_code)
        code = parsed if parsed is not None else gen.generated_code
        error = parse_error
        if error is None:
            execute_calls += 1
            try:
                output = strip_ansi(self.interpreter.execute(code))
            except (SyntaxError, TimeoutError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"

        while error is not None and regenerate_calls < config.max_iters:
            regen = self.regenerate(
                premises=premises,
                question=question,
                previous_code=code,
                error=error,
                config=config.regenerate_config,
            )
            regenerate_calls += 1
            parsed, parse_error = parse_code(regen.generated_code)
            code = parsed if parsed is not None else regen.generated_code
            error = parse_error
            output = None
            if error is None:
                execute_calls += 1
                try:
                    output = strip_ansi(self.interpreter.execute(code))
                except (SyntaxError, TimeoutError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"

        execution_success = error is None
        if not execution_success:
            if not config.fallback_on_error:
                raise RuntimeError(
                    f"PoT-Z3 solver failed after {regenerate_calls} regeneration(s): {error}"
                )
            solver_result = format_solver_result(code, None, error)
        else:
            solver_result = format_solver_result(code, output, None)

        logger.debug(
            f"PoT-Z3 pipeline: regenerate_calls={regenerate_calls}, "
            f"execute_calls={execute_calls}, success={execution_success}"
        )

        prediction = self.aggregate(
            premises=premises,
            question=question,
            solver_result=solver_result,
        )
        prediction["reasoning"] = solver_result

        total_llm_calls = 1 + regenerate_calls + 1
        prediction["search_stats"] = {
            "algorithm": "pot_z3",
            "nodes_visited": execute_calls,
            "propose_calls": 1 + regenerate_calls,
            "evaluate_calls": 0,
            "depth_reached": 1,
            "best_score": float(execution_success),
            "total_llm_calls": total_llm_calls,
            "total_completions": total_llm_calls,
            "generate_calls": 1,
            "regenerate_calls": regenerate_calls,
            "execute_calls": execute_calls,
            "execution_success": execution_success,
        }
        return prediction
