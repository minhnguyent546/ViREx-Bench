"""Shared signatures + helpers for the Program-of-Thought + Z3 (pot_z3) strategy.

PoT-Z3 asks the LM to translate the premises into an executable Python program
against a pre-loaded Z3 symbolic API, runs it in a sandboxed interpreter, then
feeds the program + its output to a commit step (the task signature with a
``solver_result`` input prepended) that renders the final Vietnamese answer.

This module hosts the two generation signatures, the :class:`POTConfig`
dataclass, and the pure-Python helpers (code extraction, ANSI stripping, solver-
result formatting, config builder). The interpreter sandbox lives in
:mod:`virex_bench.strategies.pot.interpreter`; the strategy orchestration lives
in :mod:`virex_bench.strategies.pot.strategy`.
"""

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import dspy

from virex_bench import envs
from virex_bench.logger import init_logger

logger = init_logger(__name__)


class ProgramGenerationSignature(dspy.Signature):
    """Translate the logical structure of the premises into an executable Python program
    using the pre-loaded Z3 symbolic API, then run it to derive a logical verdict.

    You are given the premises (the only source of truth) and the question. Produce
    `generated_code`: a complete, self-contained Python program that encodes the premises
    as Z3 formulas, evaluates the question symbolically, and prints a `result` dict.

    ## PRE-LOADED EXECUTION ENVIRONMENT -- USE ONLY THESE FUNCTIONS/TOKENS:
    Custom wrapper functions already in scope (do NOT import anything):
    - create_domain(name) -> uninterpreted sort (universe of discourse) via z3.DeclareSort.
    - create_constant(name, domain) -> single constant of a domain via z3.Const.
    - create_constants('x y', domain) -> multiple space-separated constants (or one).
    - create_predicate(name, *domains) -> boolean-valued z3.Function (BoolSort return).
    - create_function(name, *domains_and_range) -> mapping function; use IntSort()/RealSort()
      as the LAST argument (the return sort), NOT Real/Int/create_real/create_int.
    - create_int(name) / create_ints('x y') -> Z3 Integer variables (NOT sorts).
    - create_real(name) / create_reals('x y') -> Z3 Real variables (NOT sorts).
    - create_bool(name) / create_bools('a b') -> Z3 Boolean variables.
    - all_different(*args) -> z3.Distinct(*args).
    - check_entailment(premises_list, conclusion) -> 2-step verification returning exactly
      'entailed' (premises + Not(conclusion) is unsat), 'contradicted' (premises + conclusion
      is unsat), or 'uncertain' (neither). Do NOT pass Not(conclusion) manually.
    - find_solution(constraints_list) -> dict of variable assignments if sat, else None.

    Z3 logical tokens and sorts already in scope (use bare, no prefix):
    - And, Or, Not, Implies, ForAll, Exists, Distinct, BoolVal, IntVal, RealVal
    - IntSort(), RealSort(), BoolSort()

    ## KEY CODING RULES:
    - CRITICAL: Do NOT write `import z3`, `import logic_api`, or any `import` statement.
      Everything is pre-loaded; `__import__` is disabled. Use functions bare.
    - CRITICAL: EVERY name used in the code -- including quantifier variables like `x`,
      `y`, `z` in ForAll([x], ...) / Exists([x], ...) -- MUST be declared with an assignment
      BEFORE its first use, or Python raises NameError. Declare a domain first, then the
      variable: `domain = create_domain("Person")` then `x = create_constant("x", domain)`.
      For numeric quantifier vars: `x = create_int("x")` / `create_real("x")`.
      There is NO implicit declaration and NO bare-sort token like `ObjectSort`/`PersonSort` --
      the only sort-producing calls are `create_domain(name)`, `IntSort()`, `RealSort()`,
      `BoolSort()`. Assign the sort to a variable (`person = create_domain("Person")`) and
      reuse that variable; never reference an undeclared sort name.
    - Forbidden operators: NEVER use `->`, `=>`, `&`, `|`, `~` -- not Python/Z3 syntax.
      Use ONLY Implies(...), And(...), Or(...), Not(...).
    - NEVER use Python `and`/`or`/`not` with Z3 expressions -- they short-circuit. Use And/Or/Not.
    - All logical operators MUST be parenthesized: Not(P(x)), And(a, b), Implies(a, b).
    - Predicates are callable: Implies(P(x), Q(x)); never Implies(P, Q(x)).
    - Quantifiers use list syntax: ForAll([x], body), Exists([x], body).
    - Assign formulas as Z3 objects: p1 = ForAll([x], Implies(...)); NEVER use Python `def`.

    ## LOGIC FOUNDATIONS -- strict, no shortcuts:
    ### Condition direction (do NOT reverse unless explicit):
    - "A if B" -> B -> A.  "A only if B" / "A requires B" / "B is required for A" -> A -> B.
    - "A is sufficient for B" / "A grants/guarantees B" -> A -> B.  "A is necessary for B" -> B -> A.
    - "Without A, no B" -> B -> A.  Biconditional ("iff"/"exactly when") requires explicit wording.
    ### "All" vs "Some" (wrong encoding = wrong answer):
    - "All A are B" / "Every A is B" -> ForAll([x], Implies(A(x), B(x))).
      WRONG: ForAll([x], And(A(x), B(x))).
    - "Some A are B" / "There exists an A that is B" -> Exists([x], And(A(x), B(x))).
      WRONG: Exists([x], Implies(A(x), B(x))) -- trivially true for any non-A.
    - "No A are B" -> ForAll([x], Implies(A(x), Not(B(x)))).
    - "Not all A are B" -> Exists([x], And(A(x), Not(B(x)))).
    ### Inference traps:
    - Converse FALLACY: A->B does NOT imply B->A.  Inverse FALLACY: A->B does NOT imply ¬A->¬B.
    - Contrapositive IS valid: A->B entails ¬B->¬A.
    - Existential P(x) does NOT imply universal P(x). Facts about one named individual do
      NOT transfer to another unless a universal rule connects them.
    ### Open-world reasoning:
    - Use ONLY provided premises. No outside knowledge, no invented facts/rules/entities.
    - Absence of evidence != evidence of absence. Not-entailed does NOT mean negated.
    - Yes = claim entailed. No = negation entailed. Uncertain = neither.
    ### Vacuous truth (easily missed):
    - Implies(A, B) is vacuously TRUE when A is impossible under the premises. If any premise
      makes an option's antecedent unsatisfiable, that option is entailed by vacuous truth.
    - A free-variable constraint on `x` does NOT define a universal rule. Wrap universal
      rules explicitly: ForAll([x], Implies(...)).

    ## RELEVANCE FILTER:
    - Encode only premises that connect to the target claim/option/entity/derived fact.
    - Ignore premises about unrelated entities, times, places, exceptions, or quantities
      unless they are logically needed for entailment.
    - Do not encode a premise merely because it shares words with the question.
    - If a claim specifies access FOR a particular purpose/course, do not conclude it is
      entailed unless a premise explicitly links general permission to that specific purpose.

    ## ENTAILMENT WORKFLOW:
    - Encode premises as Z3 formulas and collect them in `premises_list`.
    - Encode the target(s) as Z3 formula(s) and call check_entailment(premises_list, target).
    - Use the returned status directly for `z3_status`. Map to public labels only inside the
      result dict (see RESULT CONTRACT): entailed->Yes/selected, contradicted->No, uncertain->Uncertain.
    - Do NOT hardcode answers; every answer must come from a check_entailment / find_solution result.

    ## QUESTION-TYPE BRANCHING (detect from the question; one program covers all):
    - yes/no question: encode the tested claim; if the question has an "if A, ..." clause,
      add A as a temporary assumption, NOT into premises_list.
      claim_status = check_entailment(premises_list + temporary_assumptions, claim).
    - multiple choice (options A./B./... embedded in the question): encode each option as a
      formula; loop `for label, formula in options.items(): statuses[label] = check_entailment(...)`.
      Collect entailed labels; default single-answer (allows_multiple_answers = False) unless
      the question uses plural wording ("which statements", "select all"). If none entailed
      and no explicit none/uncertain option is satisfied, answer = 'Uncertain'.
      Distinguish object-level negation ("not all X are Y" -> a formula) from meta-level
      non-entailment ("cannot be determined" -> evaluated from option statuses, NOT encoded
      as Not(ForAll(...))).
    - numeric / count: use find_solution or encode the quantity-bearing premise as an
      Int-valued function and read the value.
    - open-ended ("Which ...", "List ...", "What follows"): build candidate_facts, call
      check_entailment on each, keep only 'entailed' ones with non-empty supporting_premises.

    ## RECOMMENDED CODE STRUCTURE (adapt names to the ACTUAL problem -- never copy example names;
    every line below is MANDATORY unless marked optional -- skipping a declaration line yields
    a NameError at runtime):
    # 1. domain = create_domain("<Entity>")            # declare the sort FIRST
    # 2. x = create_constant("x", domain)              # generic quantifier var (REQUIRED before
    #                                                   any ForAll([x], ...) / Exists([x], ...))
    # 3. named constants for individuals: lan = create_constant("lan", domain)
    # 4. predicates with correct arity: IsStudent = create_predicate("IsStudent", domain)
    # 5. (optional) numeric functions: Credits = create_function("Credits", domain, IntSort())
    # 6. encode universal rules: p1 = ForAll([x], Implies(IsStudent(x), Eligible(x)))
    # 7. encode named facts:     p2 = IsStudent(lan)
    # 8. premises_list = [p1, p2, ...]   # only relevant ones
    # 9. encode target claim/option formulas; call check_entailment / find_solution
    # 10. map status -> answer; build result; print(result)

    ## RESULT CONTRACT (uniform for every question type):
    - Build a dict named `result` with EXACTLY these keys:
      `answer` (your best-guess answer for the detected question type -- a label like 'A' or
      'A,C' for MCQ, 'Yes'/'No'/'Uncertain' for yes/no, the entity/count/text otherwise),
      `z3_status` (one of entailed, contradicted, uncertain, mixed),
      `supporting_premises` (non-empty list of 1-based premise indices used in the derivation;
      [] only if the solver genuinely failed),
      `solution` (one concise English sentence; for MCQ summarize which options were
      entailed/contradicted, for OE list the derived facts).
    - Do not add extra keys. Do not omit any key. Assign every value to a variable first.
    - Use the check_entailment / find_solution return values to compute the fields; never
      hardcode answer='Yes' or z3_status='entailed'.
    - End with exactly `print(result)` as the last line.

    ## CODE HYGIENE (audit before finalizing):
    - Variable names: letters/digits/underscores only. WRONG: GPA2.5; RIGHT: gpa_2_5.
    - Every variable used on the RHS must be declared earlier in the code.
    - No placeholder text (`...`, `option_A_formula`) may remain in the final code.
    - Do NOT overwrite a domain variable with a predicate
      (WRONG: Student = create_domain(...); Student = create_predicate(...)).
      Use distinct names: student_domain / IsStudent.
    - Predicates MUST always be called with arguments: IsStudent(lan), never bare IsStudent.
    - Do not invent premises. Encode EXACTLY what is stated -- never add a complementary or
      opposite rule. Every entry in premises_list must correspond to an actual premise sentence.
    - Keep formulas flat: never nest more than 2 quantifier levels. Extract repeated
      sub-formulas into named variables.

    Return only executable Python code in `generated_code`. No markdown fences, no commentary.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block (Premise 1, Premise 2, ...). The only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer based on the premises.")

    generated_code: str = dspy.OutputField(
        desc="Executable Python code only. No markdown fences, no commentary. Must end with print(result)."
    )


class ProgramRegenerationSignature(dspy.Signature):
    """Fix broken Python Z3 code. Given the premises, the question, the previously-generated
    program that failed, and its execution error, produce corrected executable code that
    preserves the same logical goal.

    ## RESULT CONTRACT:
    - The corrected code must build `result` with EXACTLY these keys: `answer`, `z3_status`,
      `supporting_premises`, `solution`. End with `print(result)`.
    - Do not add or omit keys.

    ## REGENERATION RULES:
    - Keep the same logical goal; fix only code/API/formalization errors.
    - Do NOT import anything or redefine pre-loaded functions.
    - "Z3 sort expected in create_function" -> replace Real/Int/create_real/create_int with
      RealSort()/IntSort() as the return sort.
    - "Z3 sort expected in create_predicate" / sort mismatch -> a domain variable was
      overwritten by a predicate, or constants from domain A were passed into a predicate
      for domain B. Use distinct names (student_domain vs IsStudent) and check argument sorts.
    - NameError for x/y/z -> declare with create_constant before use. NameError for a named
      value -> declare it as a constant in the right domain, or omit the irrelevant premise.
    - Quantifier error -> use ForAll([x], body) / Exists([x], body) list syntax.
    - TypeError from Python booleans -> replace and/or/not with And/Or/Not.
    - SyntaxError around a logical operator (e.g. `Not P(x)`) -> call as a function: `Not(P(x))`.
    - SyntaxError on a dotted variable name (e.g. `GPA2.5`) -> rename to letters/digits/underscores.
    - "True/False or Z3 Boolean expression expected ... FuncDeclRef" -> a predicate was used
      bare; apply it to an argument: `EnforcesCompliance(some_const)`.
    - "index out of bounds" / "Z3_get_domain" -> raw z3.Function was used without a return
      sort; replace with create_predicate / create_function.
    - Bool/Int/Real sort mismatch when comparing a predicate numerically -> it was declared
      as create_predicate (returns Bool); redeclare as create_function with a numeric return sort.
    - All options 'uncertain' when a clear answer is expected -> check quantifier direction:
      'some' -> Exists+And, 'all' -> ForAll+Implies (not the reverse).
    - An 'If A then B' option marked 'uncertain' though A is impossible -> it is entailed by
      vacuous truth; check whether a premise contradicts the antecedent.
    - Empty/null output -> the code forgot print(result); add it as the last line.
    - If the error is NOT in this list: inspect the `error` field, reason about the root cause,
      and fix it directly.

    Return only corrected executable Python code in `generated_code`. No markdown fences.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block (Premise 1, Premise 2, ...). The only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer based on the premises.")
    previous_code: str = dspy.InputField(desc="The previously generated program that failed.")
    error: str = dspy.InputField(desc="The execution error or traceback to fix.")

    generated_code: str = dspy.OutputField(
        desc="Corrected executable Python code only. Must end with print(result)."
    )


# Custom rationale for the generation ChainOfThought -- cues the LM to plan the Z3
# encoding (domains, predicates, constraints) before writing code. Passed as
# ``rationale_field`` to ``ChainOfThought(ProgramGenerationSignature, ...)``.
POT_GENERATE_RATIONALE = dspy.OutputField(
    desc=(
        "Explain how the logical structure maps to executable Z3 code: the domains, predicates, "
        "constants, and constraints you will declare, and how the question type determines the "
        "check_entailment / find_solution query. Reason step by step. Do not include this "
        "reasoning in generated_code."
    )
)


@dataclass
class POTConfig:
    """Knobs for the PoT-Z3 generate -> execute -> regenerate loop.

    All fields have a 1:1 ``VIREX_BENCH_POT_*`` env var (see :mod:`virex_bench.envs`).
    Validation in :meth:`__post_init__` keeps the bounds explicit so a bad env
    value fails fast at strategy construction, not mid-loop.
    """

    max_iters: int
    """Cap on code-regeneration attempts on execution failure (>= 1)."""

    execution_timeout: float
    """Wall-clock timeout for the Z3 subprocess in seconds (> 0)."""

    generate_config: Mapping[str, Any]
    """Per-call sampling override for the code generator. Empty -> dspy inherits
    the LM's ``--model-kwargs`` profile verbatim (the default)."""

    regenerate_config: Mapping[str, Any]
    """Per-call sampling override for the regeneration module. Same semantics as
    :attr:`generate_config`."""

    fallback_on_error: bool
    """On unrecoverable execution failure: ``True`` lets the commit LM fall back
    to plain reasoning over the premises; ``False`` raises."""

    def __post_init__(self) -> None:
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if self.execution_timeout <= 0:
            raise ValueError(f"execution_timeout must be > 0, got {self.execution_timeout}")
        # Only the well-known ``temperature`` key is bounds-checked, so other
        # overrides (top_p, top_k, ...) need no special-casing (mirrors CRConfig).
        for role, override in (
            ("generate", self.generate_config),
            ("regenerate", self.regenerate_config),
        ):
            if "temperature" in override:
                temperature = override["temperature"]
                if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
                    raise ValueError(
                        f"{role}_config['temperature'] must be a number in [0, 2], "
                        f"got {temperature!r}"
                    )


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Strip ANSI color/style escape codes from captured terminal output."""
    return _ANSI_RE.sub("", text)


def strip_imports(code: str) -> str:
    """Remove ``import`` and ``from ... import`` statements from generated code.

    LLMs frequently emit ``import json``, ``import re``, etc. by habit, despite
    the instruction not to. All modules the model would want are already
    pre-loaded in the sandbox's global namespace, so import statements are
    unnecessary — and they fail with ``ImportError: __import__ not found``
    because the sandbox blocks ``__import__``.

    Uses :func:`ast.parse` to find import statements (including multi-line
    parenthesized imports) and remove those lines. Returns the code unchanged on
    syntax error so the interpreter can report it.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines_to_remove: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for lineno in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                lines_to_remove.add(lineno)

    if not lines_to_remove:
        return code

    logger.debug(f"Stripping {len(lines_to_remove)} import line(s) from generated code")
    filtered_lines = [
        line
        for lineno, line in enumerate(code.split("\n"), start=1)
        if lineno not in lines_to_remove
    ]
    return "\n".join(filtered_lines)


def parse_code(raw: str) -> tuple[str | None, str | None]:
    """Extract executable Python from an LLM ``generated_code`` string.

    Handles three LLM habits that break direct execution:
    - Wrapping code in ```` ```python ... ``` ```` fences despite the instruction not to.
    - Appending explanatory text after a ``---`` separator.
    - Emitting ``import`` statements despite the instruction not to. These are
      stripped via :func:`strip_imports` because the sandbox pre-loads all needed
      modules and blocks ``__import__``.

    On success returns ``(code_block, None)``. On failure (empty or garbled code)
    returns ``(None, error_message)`` -- the caller (the strategy's regenerate loop)
    treats a non-None error as a code-execution failure.

    If the last non-empty line is a bare assignment (``result = {...}``) without a
    ``print``, a ``print(varname)`` is appended so the interpreter captures the value
    via stdout (its primary output path) rather than relying on the ``result`` global
    fallback.
    """
    # Drop trailing commentary after a "---" separator.
    code = raw.split("---", 1)[0]

    # Primary path: extract code from a complete fenced block.
    code_match = re.search(r"```(?:python)?\s*\n(.*?)\n\s*```", code, re.DOTALL)
    if code_match:
        code_block = code_match.group(1).strip()
    else:
        # Fallback: the LLM output a partial block (e.g. opening ```python but no
        # closing ```), or no markers at all. Strip markers from start/end manually.
        code_block = code
        code_block = re.sub(r"^```(?:python)?\s*\n?", "", code_block, count=1)
        code_block = re.sub(r"\n?\s*```\s*$", "", code_block, count=1)
        code_block = code_block.strip()

    code_block = strip_imports(code_block)

    if not code_block:
        return None, "Error: Empty code after parsing."
    # Heuristic: a single line with multiple "=" is almost certainly a garbled extract
    # (e.g. "answer = result = ..."), not a real one-liner.
    if "\n" not in code_block and code_block.count("=") > 1:
        return None, "Error: Code format is not correct."

    # If the last line is a bare assignment (e.g. "result = {...}") without a print,
    # append print(varname) so the interpreter captures it via stdout.
    lines = code_block.split("\n")
    last_line_match = re.match(r"^(\w+)\s*=", lines[-1].strip())
    if last_line_match and len(lines) > 1:
        code_block += "\nprint(" + last_line_match.group(1) + ")"

    return code_block, None


def format_solver_result(code: str, output: str | None, error: str | None) -> str:
    """Build the ``solver_result`` string fed to the commit step.

    On success (``error is None``): the program and its captured output, so the
    commit LM can read the symbolic verdict.

    On failure (``error is not None``): the failed program and the error message,
    followed by a cue to reason over the premises directly. This graceful fallback
    lets the commit LM still produce an answer when the symbolic solver crashes.
    """
    if error is not None:
        return f"Z3 Program (failed):\n{code}\nError:\n{error}\nReason over the premises directly."
    if output is None:
        raise ValueError("format_solver_result requires output or error to be set")
    return f"Z3 Program:\n{code}\nProgram Output:\n{output}"


def build_pot_config() -> POTConfig:
    """Populate a :class:`POTConfig` from the ``VIREX_BENCH_POT_*`` env vars.

    Bounds are validated in ``POTConfig.__post_init__`` so a bad value fails fast
    at strategy construction.
    """
    generate_temperature = envs.VIREX_BENCH_POT_GENERATE_TEMPERATURE
    regenerate_temperature = envs.VIREX_BENCH_POT_REGENERATE_TEMPERATURE
    return POTConfig(
        max_iters=envs.VIREX_BENCH_POT_MAX_ITERS,
        execution_timeout=envs.VIREX_BENCH_POT_EXECUTION_TIMEOUT,
        # None -> empty config -> dspy inherits the LM's --model-kwargs profile.
        generate_config={}
        if generate_temperature is None
        else {"temperature": generate_temperature},
        regenerate_config=(
            {} if regenerate_temperature is None else {"temperature": regenerate_temperature}
        ),
        fallback_on_error=envs.VIREX_BENCH_POT_FALLBACK_ON_ERROR,
    )


# Re-exported for tests / strategy imports.
__all__ = [
    "POTConfig",
    "POT_GENERATE_RATIONALE",
    "ProgramGenerationSignature",
    "ProgramRegenerationSignature",
    "build_pot_config",
    "format_solver_result",
    "parse_code",
    "strip_ansi",
    "strip_imports",
]
