"""Z3 execution sandbox for the Program-of-Thought (PoT) strategy.

The sandbox runs LLM-generated Python code against a pre-loaded, LLM-friendly
Z3 wrapper API so the answer is backed by a hard entailment/satisfiability
verdict rather than a language-model guess. Execution happens in a spawned
subprocess with a wall-clock timeout to prevent infinite loops.

Two crown jewels are ported near-verbatim from the reference
(``api/app/shared/interpreters.py``):

- :func:`check_entailment` — the classic two-solver unsat test that maps a
  conclusion to *entailed* / *contradicted* / *uncertain*.
- :func:`find_solution` — returns a model's variable assignments when the
  constraints are satisfiable, ``None`` otherwise.

Differences from the reference (see ``.plans/program-of-thoughts-z3.md``):

- ``__import__`` is NOT re-added to the safe builtins — the instruction forbids
  imports, and a violation raises ``ImportError`` (triggering regeneration).
- ``numpy`` / ``sympy`` / ``scipy`` / ``mpmath`` are not injected — only stdlib
  modules (``json``, ``math``, ``collections``, ``itertools``, ``re``,
  ``functools``, ``string``) plus Z3.
- No ``rich`` / ``ruff`` / file-logging dependencies — uses the shared
  :mod:`virex_bench.logger`.

This module has no DSPy dependency and is fully unit-testable in isolation.
"""

import ast
import builtins
import collections
import contextlib
import functools
import io
import itertools
import json
import math
import multiprocessing
import re
import string
import traceback
from typing import Any, Literal

import z3

from virex_bench.logger import init_logger

logger = init_logger(__name__)

__all__ = ["LocalZ3PythonInterpreter", "check_entailment", "find_solution"]


# Safe builtins

_DANGEROUS_BUILTINS = frozenset(
    {
        # Dynamic code execution
        "exec",
        "eval",
        "compile",
        # File / stream I/O
        "open",
        "input",
        # Debugger / raw memory
        "breakpoint",
        "memoryview",
        # Import machinery (the instruction forbids imports)
        "__import__",
        # Namespace / scope introspection — would expose the sandbox internals
        "globals",
        "locals",
        "dir",
        # Interactive-interpreter hooks
        "exit",
        "quit",
        "help",
    }
)


def _make_safe_builtins() -> dict[str, Any]:
    """Build a safe builtins dict from all Python builtins, excluding dangerous functions.

    The blocklist removes code-execution builtins (``exec`` / ``eval`` /
    ``compile``), file I/O (``open`` / ``input``), import machinery
    (``__import__``), and namespace introspection (``globals`` / ``locals`` /
    ``dir``). ``__import__`` is NOT re-added, so an ``import`` statement
    raises ``ImportError``, which the strategy's regenerate loop turns into a
    correction request.
    """
    return {
        key: value
        for key, value in vars(builtins).items()
        if not key.startswith("_") and key not in _DANGEROUS_BUILTINS
    }


_SAFE_BUILTINS = _make_safe_builtins()


# LLM-friendly Z3 wrapper API


def _unwrap(arg: Any) -> Any:
    """Unwrap a single-element list — LLMs sometimes pass ``[domain]`` instead of ``domain``."""
    if isinstance(arg, list) and len(arg) == 1:
        return arg[0]
    return arg


def create_domain(name: str) -> Any:
    """Create an uninterpreted sort (universe of discourse) via ``z3.DeclareSort``."""
    return z3.DeclareSort(name)


def create_constant(name: str, domain: Any) -> Any:
    """Create a single constant/individual of the given domain."""
    return z3.Const(name, _unwrap(domain))


def create_constants(names: str, domain: Any) -> Any:
    """Create one or more space-separated constants of the given domain.

    A single name returns a single constant; multiple names return a list.
    """
    unwrapped_domain = _unwrap(domain)
    parts = names.split()
    if len(parts) == 1:
        return z3.Const(parts[0], unwrapped_domain)
    return z3.Consts(names, unwrapped_domain)


def create_predicate(name: str, *domains: Any) -> Any:
    """Create a boolean-valued predicate function (``z3.Function`` with ``BoolSort`` return)."""
    unwrapped = tuple(_unwrap(domain) for domain in domains)
    return z3.Function(name, *unwrapped, z3.BoolSort())


def create_function(name: str, *domains_and_range: Any, **kwargs: Any) -> Any:
    """Create a mapping function with a specified return sort.

    Absorbs the LLM hallucination ``return_sort=RealSort()`` passed as a keyword arg,
    and unwraps single-element list domain args.
    """
    if "return_sort" in kwargs:
        domains_and_range = (*domains_and_range, kwargs["return_sort"])
    unwrapped = tuple(_unwrap(domain) for domain in domains_and_range)
    return z3.Function(name, *unwrapped)


def create_int(name: str) -> Any:
    """Create a Z3 Integer variable."""
    return z3.Int(name)


def create_ints(names: str) -> Any:
    """Create multiple Z3 Integer variables from space-separated names."""
    return z3.Ints(names)


def create_real(name: str) -> Any:
    """Create a Z3 Real-valued variable."""
    return z3.Real(name)


def create_reals(names: str) -> Any:
    """Create multiple Z3 Real-valued variables from space-separated names."""
    return z3.Reals(names)


def create_bool(name: str) -> Any:
    """Create a Z3 Boolean variable."""
    return z3.Bool(name)


def create_bools(names: str) -> Any:
    """Create multiple Z3 Boolean variables from space-separated names."""
    return z3.Bools(names)


def all_different(*args: Any) -> Any:
    """Assert all provided variables have distinct values (``z3.Distinct``)."""
    return z3.Distinct(*args)


def check_entailment(
    premises: list[Any], conclusion: Any
) -> Literal["entailed", "contradicted", "uncertain"]:
    """Check whether *premises* entail, contradict, or leave *conclusion* open.

    Classic two-solver unsat test:

    - ``premises ∧ ¬conclusion`` is UNSAT  ⟹ ``"entailed"``
    - ``premises ∧  conclusion`` is UNSAT  ⟹ ``"contradicted"``
    - otherwise                              ⟹ ``"uncertain"``
    """
    solver = z3.Solver()
    solver.add(z3.And(*premises))
    solver.add(z3.Not(conclusion))
    if solver.check() == z3.unsat:
        return "entailed"

    solver_neg = z3.Solver()
    solver_neg.add(z3.And(*premises))
    solver_neg.add(conclusion)
    if solver_neg.check() == z3.unsat:
        return "contradicted"
    return "uncertain"


def find_solution(constraints: list[Any]) -> dict[str, int | float | str] | None:
    """Check satisfiability of *constraints* and return variable assignments.

    Returns a dict mapping variable names to their values (``int`` for integer
    constants, ``float`` for rationals, ``str`` for anything else), or ``None``
    if the constraints are unsatisfiable.
    """
    solver = z3.Solver()
    solver.add(*constraints)
    if solver.check() != z3.sat:
        return None
    model = solver.model()
    assignments: dict[str, int | float | str] = {}
    for declaration in model.decls():
        value: Any = model[declaration]
        if z3.is_int_value(value):
            assignments[declaration.name()] = value.as_long()
        elif z3.is_rational_value(value):
            assignments[declaration.name()] = float(value.numerator_as_long()) / float(
                value.denominator_as_long()
            )
        else:
            assignments[declaration.name()] = str(value)
    return assignments


_LOCAL_WRAPPERS: dict[str, Any] = {
    "create_domain": create_domain,
    "create_constant": create_constant,
    "create_constants": create_constants,
    "create_predicate": create_predicate,
    "create_function": create_function,
    "create_int": create_int,
    "create_ints": create_ints,
    "create_real": create_real,
    "create_reals": create_reals,
    "create_bool": create_bool,
    "create_bools": create_bools,
    "all_different": all_different,
    "check_entailment": check_entailment,
    "find_solution": find_solution,
}


# Attribute proxies (forgive z3./logic_api. prefixes the LLM may emit)


class _LogicApiMock:
    """Proxy that routes ``logic_api.foo(...)`` to the wrappers first, then Z3."""

    def __getattr__(self, name: str) -> Any:
        if name in _LOCAL_WRAPPERS:
            return _LOCAL_WRAPPERS[name]
        if hasattr(z3, name):
            return getattr(z3, name)
        raise AttributeError(f"Function '{name}' does not exist in the logic environment.")


class _Z3ApiMock:
    """Proxy that routes ``z3.foo(...)`` to real Z3 first, then falls back to wrappers."""

    def __getattr__(self, name: str) -> Any:
        if hasattr(z3, name):
            return getattr(z3, name)
        if name in _LOCAL_WRAPPERS:
            return _LOCAL_WRAPPERS[name]
        raise AttributeError(f"Library Z3 does not have attribute or function '{name}'.")


# Execution worker (module-scope — picklable for spawn)


def _build_z3_globals() -> dict[str, Any]:
    """Assemble the global namespace injected into the exec'd program.

    Bare Z3 tokens (``And``, ``Or``, ``Not``, ...) and the custom wrappers are
    in scope so the generated code never needs an ``import`` statement. The
    ``z3`` / ``logic_api`` proxies forgive prefixed call styles.
    """
    globals_dict: dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": _SAFE_BUILTINS,
        **_LOCAL_WRAPPERS,
        "z3": _Z3ApiMock(),
        "logic_api": _LogicApiMock(),
        "z3api": _Z3ApiMock(),
        "z3_wrapper": _Z3ApiMock(),
        "json": json,
        "math": math,
        "collections": collections,
        "itertools": itertools,
        "re": re,
        "functools": functools,
        "string": string,
    }
    # Expose all non-underscore Z3 names as bare tokens (And, Or, Not, Implies,
    # ForAll, Exists, Distinct, BoolVal, IntVal, RealVal, IntSort, ...).
    # Skip 'z3' so the _Z3ApiMock proxy is not overwritten by the real submodule.
    for name in dir(z3):
        if not name.startswith("_") and name != "z3":
            globals_dict[name] = getattr(z3, name)
    return globals_dict


def _run_z3_python_code(code: str) -> tuple[str, str]:
    """Exec *code* in the restricted Z3 environment and capture stdout.

    Intended to run in a spawned subprocess for timeout isolation. Returns
    ``("success", output)`` or ``("error", formatted_traceback)``.
    """
    globals_dict = _build_z3_globals()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, globals_dict, globals_dict)
        output = buffer.getvalue()
        # Fallbacks: programs that assign answer/result but forget to print.
        if not output and "answer" in globals_dict:
            output = str(globals_dict["answer"])
        if not output and "result" in globals_dict:
            output = str(globals_dict["result"])
        return "success", output
    except Exception:
        return "error", traceback.format_exc()


# Public interpreter class


class LocalZ3PythonInterpreter:
    """Sandboxed Z3 Python interpreter with subprocess-level timeout isolation.

    Runs LLM-generated Python code in a spawned child process against the
    pre-loaded Z3 wrapper API. The wall-clock timeout prevents infinite loops;
    ``ast.parse`` catches syntax errors before spawning. Dangerous builtins (code
    execution, file I/O, import machinery, namespace introspection) are removed,
    but the subprocess boundary — not the builtins blocklist — is the real
    isolation: bare attribute access can still reach arbitrary objects.

    Raises:
        SyntaxError: if the code fails to parse.
        TimeoutError: if execution exceeds *timeout* seconds.
        RuntimeError: if the worker crashes or the code raises at runtime.
    """

    def __init__(self, timeout: float = 45.0) -> None:
        self.timeout = timeout
        logger.debug(f"Initialized {self.__class__.__name__} with timeout={self.timeout}s")

    def execute(self, code: str) -> str:
        """Parse and execute *code*, returning captured stdout.

        Falls back to the ``answer`` or ``result`` global variable when the
        program produces no stdout.
        """
        try:
            ast.parse(code)
        except SyntaxError as error:
            logger.debug(f"SyntaxError before execution: {error}")
            raise SyntaxError(f"Invalid Python syntax: {error}") from error

        logger.debug("Code syntax valid; spawning execution subprocess...")
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            async_result = pool.apply_async(_run_z3_python_code, (code,))
            try:
                status, output = async_result.get(timeout=self.timeout)
            except multiprocessing.TimeoutError as error:
                pool.terminate()
                message = f"Execution timed out after {self.timeout} seconds."
                logger.debug(message)
                raise TimeoutError(message) from error
            except OSError as error:
                pool.terminate()
                message = (
                    f"Worker process communication error: {error}. "
                    "The worker process may have crashed or been killed."
                )
                logger.debug(message)
                raise RuntimeError(message) from error

        if status == "error":
            logger.debug(f"Execution failed:\n{output}")
            raise RuntimeError(f"Error during execution:\n{output}")

        logger.debug(f"Execution succeeded; output length {len(output)} chars")
        return output

    def __call__(self, code: str) -> str:
        """Alias for :meth:`execute`."""
        return self.execute(code)
