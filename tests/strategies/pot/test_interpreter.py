"""Tests for the Z3 execution sandbox in ``interpreter.py``."""

# pyright: reportPrivateUsage=false

import multiprocessing

import pytest

from virex_bench.strategies.pot import interpreter as interpreter_module
from virex_bench.strategies.pot.interpreter import (
    LocalZ3PythonInterpreter,
    _apply_solver_timeout,
    check_entailment,
    create_constant,
    create_constants,
    create_domain,
    create_function,
    create_predicate,
    find_solution,
)

# check_entailment — the two-solver unsat test


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


# Per-check Z3 soft timeout — _apply_solver_timeout + unknown mapping


def test_apply_solver_timeout_noop_when_unconfigured() -> None:
    """With no soft timeout configured, _apply_solver_timeout touches neither
    the solver nor Z3 -- a fresh solver still works normally."""
    import z3

    saved = interpreter_module._Z3_SOFT_TIMEOUT_MS
    interpreter_module._Z3_SOFT_TIMEOUT_MS = None
    try:
        solver = z3.Solver()
        _apply_solver_timeout(solver)
        solver.add(z3.Int("x") >= 0)
        assert solver.check() == z3.sat
    finally:
        interpreter_module._Z3_SOFT_TIMEOUT_MS = saved


def test_apply_solver_timeout_invokes_solver_set_with_configured_ms() -> None:
    """When configured, _apply_solver_timeout forwards the millisecond budget to
    ``solver.set("timeout", ms)``; when unconfigured it must not touch the solver.

    Asserted via a recording fake so the test does not depend on Z3's own solve
    timing (which is racy at the millisecond scale on a shared CI box)."""

    class _RecordingSolver:
        def __init__(self) -> None:
            self.set_calls: list[tuple[str, object]] = []

        def set(self, key: str, value: object) -> None:
            self.set_calls.append((key, value))

    saved = interpreter_module._Z3_SOFT_TIMEOUT_MS
    try:
        interpreter_module._Z3_SOFT_TIMEOUT_MS = 5000
        configured = _RecordingSolver()
        _apply_solver_timeout(configured)  # pyright: ignore[reportArgumentType]
        assert configured.set_calls == [("timeout", 5000)]

        interpreter_module._Z3_SOFT_TIMEOUT_MS = None
        unconfigured = _RecordingSolver()
        _apply_solver_timeout(unconfigured)  # pyright: ignore[reportArgumentType]
        assert unconfigured.set_calls == []
    finally:
        interpreter_module._Z3_SOFT_TIMEOUT_MS = saved


def test_check_entailment_returns_uncertain_when_solver_unknown(monkeypatch) -> None:
    """When the Z3 soft timeout fires (``unknown``), check_entailment falls
    through to ``"uncertain"`` -- the neutral verdict that lets the commit LM
    reason over the premises instead of pegging a core for the full wall-clock.

    Forced via monkeypatch so the test does not depend on Z3's solve timing."""
    import z3

    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    assert check_entailment([z3.BoolVal(True)], z3.BoolVal(True)) == "uncertain"


def test_find_solution_returns_none_when_solver_unknown(monkeypatch) -> None:
    """When the Z3 soft timeout fires (``unknown``), find_solution returns
    ``None`` -- the conservative same-as-unsat outcome.

    Forced via monkeypatch so the test does not depend on Z3's solve timing."""
    import z3

    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    assert find_solution([z3.Int("x") >= 0]) is None


# find_solution — satisfiability + model extraction


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


def test_find_solution_non_numeric_returns_string() -> None:
    """An uninterpreted-sort model value is returned as a string (the ``else``
    branch of find_solution's value-type switch)."""
    import z3

    person = z3.DeclareSort("Person")
    a = z3.Const("a", person)
    b = z3.Const("b", person)
    result = find_solution([a != b])
    assert result is not None
    assert isinstance(result["a"], str)
    assert isinstance(result["b"], str)


def test_find_solution_empty_constraints_returns_empty_dict() -> None:
    """No constraints are trivially satisfiable; the empty model yields no decls."""
    assert find_solution([]) == {}


# create_constants — single vs multiple names is a real branch (scalar vs list).


def test_create_constants_single_name_returns_scalar() -> None:
    person = create_domain("Person")
    single = create_constants("lan", person)
    # A single name returns one constant, not a one-element list.
    assert not isinstance(single, list)
    assert single.sort() == person


def test_create_constants_multiple_names_returns_list() -> None:
    person = create_domain("Person")
    multi = create_constants("lan minh", person)
    assert isinstance(multi, list)
    assert len(multi) == 2
    assert all(const.sort() == person for const in multi)


def test_create_constants_separate_name_args() -> None:
    """LLMs sometimes pass names as separate positional args instead of a single
    space-separated string — the variadic signature absorbs this."""
    person = create_domain("Person")
    multi = create_constants("a", "b", "c", person)
    assert isinstance(multi, list)
    assert len(multi) == 3
    assert all(const.sort() == person for const in multi)


def test_create_constants_list_of_names() -> None:
    """A list of name strings is flattened into individual names."""
    person = create_domain("Person")
    multi = create_constants(["a", "b"], person)
    assert isinstance(multi, list)
    assert len(multi) == 2


def test_create_constants_duplicated_domain_arg() -> None:
    """Extra positional args that are Z3 sorts (not strings) are treated as
    domains — only the first is used, extras are silently ignored."""
    person = create_domain("Person")
    multi = create_constants("a b", person, person)
    assert isinstance(multi, list)
    assert len(multi) == 2
    assert all(const.sort() == person for const in multi)


def test_create_constants_names_interleaved_with_domain() -> None:
    """Names and domain can be in any order — the domain is identified by type."""
    person = create_domain("Person")
    multi = create_constants("a", person, "b")
    assert isinstance(multi, list)
    assert len(multi) == 2
    assert all(const.sort() == person for const in multi)


def test_create_constants_unwraps_single_element_list_domain() -> None:
    """A domain wrapped in a single-element list (common LLM mistake) is unwrapped."""
    person = create_domain("Person")
    single = create_constants("a", [person])
    assert not isinstance(single, list)
    assert single.sort() == person


def test_create_constants_single_name_still_returns_scalar() -> None:
    """The documented 2-arg call with a single name still returns a scalar."""
    person = create_domain("Person")
    single = create_constants("a", person)
    assert not isinstance(single, list)
    assert single.sort() == person


def test_create_constants_raises_on_no_names() -> None:
    person = create_domain("Person")
    with pytest.raises(TypeError, match="name"):
        create_constants(person)


def test_create_constants_raises_on_no_domain() -> None:
    with pytest.raises(TypeError, match="domain"):
        create_constants("a", "b")


# Wrapper forgiveness — _unwrap single-element lists


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


# LocalZ3PythonInterpreter.execute — end-to-end sandbox tests


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


def test_execute_z3_prefix_unknown_name_raises() -> None:
    """An unknown ``z3.foo`` name (not real Z3, not a local wrapper) raises
    AttributeError inside the sandbox -- the symmetric counterpart to the
    ``logic_api`` unknown-name path."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    with pytest.raises(RuntimeError, match="Library Z3 does not have attribute"):
        interpreter.execute("z3.no_such_function()")


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


def test_execute_numeric_wrappers_and_all_different() -> None:
    """The numeric/boolean wrappers (create_int, create_ints, create_bool,
    create_real) and all_different are wired into the sandbox globals and
    compose with find_solution end-to-end."""
    interpreter = LocalZ3PythonInterpreter(timeout=30.0)
    code = (
        "x = create_int('x')\n"
        "y = create_int('y')\n"
        "_ab = create_ints('a b')\n"
        "b = create_bool('b')\n"
        "r = create_real('r')\n"
        "constraints = [x >= 1, x <= 5, y == x + 1, all_different(x, y), b == True, r == 0.5]\n"
        "sol = find_solution(constraints)\n"
        "result = {'answer': str(sol), 'z3_status': 'entailed'}\n"
        "print(result)\n"
    )
    output = interpreter.execute(code)
    assert "entailed" in output
    assert "'x':" in output
    assert "'r': 0.5" in output


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


def test_each_execute_call_spawns_a_fresh_pool(monkeypatch) -> None:
    """execute() uses a per-call `with ctx.Pool(1) as pool:` -- Pool() is
    constructed on every call and released at the end, so peak memory scales
    with current concurrency, not the whole evaluation run."""
    pool_create_count = 0

    class _FakeAsyncResult:
        def get(self, timeout: float | None = None) -> tuple[str, str]:
            return ("success", "ok")

    class _FakePool:
        def __enter__(self) -> "_FakePool":
            return self

        def __exit__(self, *_exc: object) -> None:
            pass

        def apply_async(self, func: object, args: tuple[object, ...]) -> _FakeAsyncResult:
            return _FakeAsyncResult()

        def terminate(self) -> None:
            pass

        def join(self) -> None:
            pass

    class _FakeContext:
        def Pool(self, processes: int) -> _FakePool:
            nonlocal pool_create_count
            pool_create_count += 1
            return _FakePool()

    monkeypatch.setattr(interpreter_module, "_get_forkserver_context", lambda: _FakeContext())

    interpreter = LocalZ3PythonInterpreter(timeout=5.0)
    interpreter.execute("print('hello')")
    interpreter.execute("print('world')")
    assert pool_create_count == 2


def test_pool_terminated_and_joined_on_timeout(monkeypatch) -> None:
    """A timeout terminates the worker and reaps it via join -- the previous
    persistent-pool design dropped the worker from a dict; the per-call design
    just relies on `with` + explicit terminate/join in the error branch."""

    class _TimingOutAsyncResult:
        def get(self, timeout: float | None = None) -> tuple[str, str]:
            raise multiprocessing.TimeoutError()

    terminated = False
    joined = False

    class _FakePool:
        def __enter__(self) -> "_FakePool":
            return self

        def __exit__(self, *_exc: object) -> None:
            pass

        def apply_async(self, func: object, args: tuple[object, ...]) -> _TimingOutAsyncResult:
            return _TimingOutAsyncResult()

        def terminate(self) -> None:
            nonlocal terminated
            terminated = True

        def join(self) -> None:
            nonlocal joined
            joined = True

    class _FakeContext:
        def Pool(self, processes: int) -> _FakePool:
            return _FakePool()

    monkeypatch.setattr(interpreter_module, "_get_forkserver_context", lambda: _FakeContext())

    interpreter = LocalZ3PythonInterpreter(timeout=5.0)
    with pytest.raises(TimeoutError):
        interpreter.execute("while True: pass")
    assert terminated
    assert joined


def test_default_timeout_is_15_seconds() -> None:
    """The default wall-clock timeout is 15s -- enough for these tasks, since a
    stuck Z3 query now self-bounds via the soft timeout instead of consuming
    the full window."""
    assert LocalZ3PythonInterpreter().timeout == 15.0


def test_execute_threads_computed_z3_soft_timeout_into_worker(monkeypatch) -> None:
    """execute() computes a per-check Z3 soft timeout (80% of the wall-clock,
    capped at 5s) and passes it as the second arg to the worker's
    ``apply_async`` -- the wiring that lets a stuck Z3 query self-bound instead
    of pegging a core for the full wall-clock window.

    Verified via a fake multiprocessing context so the test does not spawn a
    subprocess nor depend on Z3's solve timing. The real subprocess path is
    covered by ``test_execute_timeout_raises_timeout_error`` and the existing
    end-to-end execute tests.
    """
    from virex_bench.strategies.pot.interpreter import _run_z3_python_code

    captured: dict[str, object] = {}

    class _FakeAsyncResult:
        def get(self, timeout: float | None = None) -> tuple[str, str]:
            return ("success", "")

    class _FakePool:
        def __init__(self, processes: int) -> None:
            assert processes == 1

        def __enter__(self) -> "_FakePool":
            return self

        def __exit__(self, *_exc: object) -> None:
            pass

        def apply_async(self, func: object, args: tuple[object, ...]) -> _FakeAsyncResult:
            captured["func"] = func
            captured["args"] = args
            return _FakeAsyncResult()

        def terminate(self) -> None:
            pass

        def join(self) -> None:
            pass

    class _FakeContext:
        def Pool(self, processes: int) -> "_FakePool":
            return _FakePool(processes)

    monkeypatch.setattr(interpreter_module, "_get_forkserver_context", lambda: _FakeContext())

    code = "result = {'answer': 'Yes'}\nprint(result)\n"

    # timeout below the 5s cap: 80% of the wall-clock.
    interpreter = LocalZ3PythonInterpreter(timeout=5.0)
    interpreter.execute(code)
    assert captured["func"] is _run_z3_python_code
    assert captured["args"] == (code, 4000)

    # timeout above the 5s cap: capped at 5000ms.
    interpreter = LocalZ3PythonInterpreter(timeout=20.0)
    interpreter.execute(code)
    assert captured["args"] == (code, 5000)

    # default 15s wall-clock still caps the soft timeout at 5s.
    interpreter = LocalZ3PythonInterpreter()
    interpreter.execute(code)
    assert captured["args"] == (code, 5000)


# Helpers — thin wrappers to keep check_entailment tests readable without
# importing z3 at the top level


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
