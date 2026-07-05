"""Tests for the Z3 execution sandbox in ``interpreter.py``.

Covers the two crown jewels (``check_entailment`` / ``find_solution``), the
LLM-friendly wrapper API, and the ``LocalZ3PythonInterpreter.execute`` contract
(stdout capture, fallbacks, error modes, timeout, proxy forgiveness).
"""

import pytest

from virex_bench.strategies.pot.interpreter import (
    LocalZ3PythonInterpreter,
    check_entailment,
    create_constant,
    create_domain,
    create_function,
    create_predicate,
    find_solution,
)

# ---------------------------------------------------------------------------
# check_entailment — the two-solver unsat test
# ---------------------------------------------------------------------------


def test_check_entailment_entailed() -> None:
    """Premises force the conclusion ⟹ 'entailed'."""
    person = create_domain("Person")
    a = create_constant("a", person)
    reads = create_predicate("Reads", person)
    knows = create_predicate("Knows", person)
    premise = _forall(person, lambda x: _implies(reads(x), knows(x)))
    assert check_entailment([premise, reads(a)], knows(a)) == "entailed"


def test_check_entailment_contradicted() -> None:
    """Premises force the negation ⟹ 'contradicted'."""
    person = create_domain("Person")
    a = create_constant("a", person)
    knows = create_predicate("Knows", person)
    # Premise: a does NOT know. Conclusion: a knows. → contradicted.
    assert check_entailment([_not(knows(a))], knows(a)) == "contradicted"


def test_check_entailment_uncertain() -> None:
    """Premises neither prove nor disprove ⟹ 'uncertain'."""
    person = create_domain("Person")
    a = create_constant("a", person)
    reads = create_predicate("Reads", person)
    knows = create_predicate("Knows", person)
    # Premise: Reads(a). Conclusion: Knows(a). No rule connecting them.
    assert check_entailment([reads(a)], knows(a)) == "uncertain"


# ---------------------------------------------------------------------------
# find_solution — satisfiability + model extraction
# ---------------------------------------------------------------------------


def test_find_solution_sat_returns_assignments() -> None:
    import z3

    x = z3.Int("x")
    y = z3.Int("y")
    result = find_solution([x >= 1, x <= 5, y == x + 1])
    assert result is not None
    assert result["x"] in {1, 2, 3, 4, 5}
    assert result["y"] == result["x"] + 1


def test_find_solution_unsat_returns_none() -> None:
    import z3

    x = z3.Int("x")
    assert find_solution([x >= 1, x <= 0]) is None


def test_find_solution_rational_returns_float() -> None:
    """A fractional Real model is returned as a Python float."""
    import z3

    x = z3.Real("x")
    result = find_solution([2 * x == 1])
    assert result is not None
    assert result["x"] == 0.5


# ---------------------------------------------------------------------------
# Wrapper forgiveness — _unwrap single-element lists
# ---------------------------------------------------------------------------


def test_create_predicate_unwraps_single_element_list() -> None:
    """LLMs sometimes pass ``[domain]`` instead of ``domain`` — _unwrap handles it."""
    person = create_domain("Person")
    loves = create_predicate("Loves", [person], [person])
    romeo = create_constant("romeo", person)
    claim = loves(romeo, romeo)
    assert check_entailment([claim], claim) == "entailed"


def test_create_function_absorbs_return_sort_kwarg() -> None:
    """LLMs sometimes pass ``return_sort=IntSort()`` — it is appended as the return sort."""
    import z3

    person = create_domain("Person")
    age = create_function("Age", person, return_sort=z3.IntSort())
    lan = create_constant("lan", person)
    assert age.range() == z3.IntSort()
    assert age.domain(0) == person
    assert z3.is_int(age(lan))


# ---------------------------------------------------------------------------
# LocalZ3PythonInterpreter.execute — end-to-end sandbox tests
# ---------------------------------------------------------------------------


def test_execute_captures_printed_result() -> None:
    """A well-formed program that builds ``result`` and prints it → stdout captured."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = (
        "person = create_domain('Person')\n"
        "a = create_constant('a', person)\n"
        "P = create_predicate('P', person)\n"
        "status = check_entailment([P(a)], P(a))\n"
        "result = {'answer': 'Yes', 'z3_status': status, "
        "'supporting_premises': [1], 'solution': 'entailed'}\n"
        "print(result)\n"
    )
    output = interpreter.execute(code)
    assert "entailed" in output
    assert "Yes" in output


def test_execute_falls_back_to_result_variable() -> None:
    """Program assigns ``result`` but forgets to print → fallback to str(result)."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = "result = {'answer': 'Yes', 'z3_status': 'entailed'}"
    output = interpreter.execute(code)
    assert "entailed" in output


def test_execute_falls_back_to_answer_variable() -> None:
    """Program assigns ``answer`` but produces no stdout → fallback to str(answer)."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    output = interpreter.execute("answer = 42")
    assert "42" in output


def test_execute_logic_api_prefix() -> None:
    """The ``logic_api.foo(...)`` call style is routed to the wrappers via _LogicApiMock."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = (
        "person = create_domain('Person')\n"
        "a = create_constant('a', person)\n"
        "P = create_predicate('P', person)\n"
        "status = logic_api.check_entailment([P(a)], P(a))\n"
        "print(status)\n"
    )
    assert "entailed" in interpreter.execute(code)


def test_execute_z3_prefix() -> None:
    """The ``z3.foo(...)`` call style is routed to real Z3 via _Z3ApiMock."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = 'x = z3.Int("x")\ns = z3.Solver()\ns.add(x >= 1, x <= 3)\nprint(s.check())\n'
    assert "sat" in interpreter.execute(code)


def test_execute_z3_prefix_falls_back_to_wrapper() -> None:
    """``z3.check_entailment`` (not a real Z3 name) falls back to the local wrapper."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = (
        "person = create_domain('Person')\n"
        "a = create_constant('a', person)\n"
        "P = create_predicate('P', person)\n"
        "print(z3.check_entailment([P(a)], P(a)))\n"
    )
    assert "entailed" in interpreter.execute(code)


def test_execute_logic_api_unknown_name_raises() -> None:
    """An unknown ``logic_api.foo`` name raises AttributeError inside the sandbox."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    with pytest.raises(RuntimeError, match="does not exist in the logic environment"):
        interpreter.execute("logic_api.no_such_function()")


def test_execute_bare_z3_tokens() -> None:
    """Bare Z3 tokens (And, Or, Not, Implies) are in scope without a prefix."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = (
        "person = create_domain('Person')\n"
        "a = create_constant('a', person)\n"
        "P = create_predicate('P', person)\n"
        "Q = create_predicate('Q', person)\n"
        "claim = Implies(P(a), Q(a))\n"
        "status = check_entailment([P(a), claim], Q(a))\n"
        "print(status)\n"
    )
    assert "entailed" in interpreter.execute(code)


def test_execute_imports_are_blocked() -> None:
    """An ``import`` statement raises ImportError (no __import__ in safe builtins)."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    with pytest.raises(RuntimeError, match="ImportError"):
        interpreter.execute("import os")


def test_execute_dangerous_builtins_are_blocked() -> None:
    """Code-execution, I/O, namespace, and introspection builtins are removed.

    Each name is referenced; if it survives in the sandbox the program prints
    ``LEAK:<name>``. We assert no leaks.
    """
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    dangerous_names = [
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "breakpoint",
        "memoryview",
        "globals",
        "locals",
        "dir",
        "exit",
        "quit",
        "help",
    ]
    checks = "\n".join(
        f"try:\n    {name}\n    print('LEAK:{name}')\nexcept NameError:\n    pass"
        for name in dangerous_names
    )
    output = interpreter.execute(checks)
    assert "LEAK" not in output


def test_execute_syntax_error_raises_syntax_error() -> None:
    interpreter = LocalZ3PythonInterpreter(timeout=5.0)
    with pytest.raises(SyntaxError):
        interpreter.execute("if True\n    print('bad')")


def test_execute_runtime_error_raises_runtime_error() -> None:
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    with pytest.raises(RuntimeError, match="Error during execution"):
        interpreter.execute("x = 1 / 0")


def test_execute_timeout_raises_timeout_error() -> None:
    interpreter = LocalZ3PythonInterpreter(timeout=2.0)
    with pytest.raises(TimeoutError, match="timed out"):
        interpreter.execute("while True:\n    pass")


# ---------------------------------------------------------------------------
# Helpers — thin wrappers to keep check_entailment tests readable without
# importing z3 at the top level
# ---------------------------------------------------------------------------


def _forall(domain: object, fn: object) -> object:
    import z3

    x = z3.Const("x", domain)  # type: ignore[arg-type]
    return z3.ForAll([x], fn(x))  # type: ignore[operator]


def _implies(antecedent: object, consequent: object) -> object:
    import z3

    return z3.Implies(antecedent, consequent)


def _not(formula: object) -> object:
    import z3

    return z3.Not(formula)
