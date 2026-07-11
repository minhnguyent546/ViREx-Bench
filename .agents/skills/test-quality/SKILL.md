---
name: test-quality
description: Review existing tests for weakness — missing edge cases, weak assertions, over-mocked internals, implementation-coupled asserts, flaky patterns, and non-BDD-friendly structure — then fix them. Use when asked to check for weak tests, review test quality, or strengthen the test suite.
---

# Test Quality: Find and Fix Weak Tests

Review existing tests for weakness and strengthen them: add missing edge cases, sharpen
assertions, remove over-mocking of internals, and eliminate flaky patterns. Adapted to this
repo's plain-pytest-function conventions (not BDD `describe`/`it`).

## Project Context

ViREx-Bench is a Vietnamese multi-step-reasoning benchmark built on **DSPy**, organized around
four axes — task / model / strategy / decoding — plus shared modules. See `AGENTS.md` for the
full layout and conventions.

Boundaries (from `AGENTS.md`):

- **Do not edit** files under `data/` (datasets managed externally) or `uv.lock` by hand.
- **Do not modify** generated/cached dirs: `.ruff_cache/`, `.cache/`, `_testing_results/`.
- **Never commit** secrets or `.env` contents.
- Ignore files/dirs starting with `_` (e.g. `_testing_results/`, `_test_format.py`) — local
  files, not part of the main codebase.

## Phase 1: Scope the Review

Pick one of:

- **Diff-scoped**: review only test files changed on the current branch.
  `git diff --name-only <base> | rg '(test_|_test\.py$|tests/)'` — then read each changed test
  file in full plus the source file it covers.
- **Suite-scoped**: review a subtree (e.g. `tests/strategies/`, `tests/evaluation/`) or the
  whole `tests/` tree. Use when the user asks for a general test-quality audit.

If the user gives a specific file/module, scope to that. Otherwise ask which scope they want.

## Phase 2: Read Context

For each test file under review:

1. Read the **full test file** (not just diffs).
2. Read the **source file it covers** (e.g. `tests/strategies/tot/test_beam_search.py` →
   `virex_bench/strategies/tot/search/beam.py`) — you cannot judge test quality without
   knowing what behavior exists to be covered.
3. Read **1–2 sibling test files** in the same directory to learn the local conventions
   (parametrize style, fixture usage, naming).
4. Note the module's public surface (exported functions/classes, registry entries) and its
   branching/error paths.

## Phase 3: Check for Weak Tests

Evaluate each test against the checklist below. Flag only issues **actually present** — do not
invent problems to inflate the findings count.

### 3.1 Missing coverage

- **Untested public behavior**: a public function/method in the source file with no test
  exercising it.
- **Missing happy path**: the main user journey of a function is not asserted.
- **Missing failure / error path**: error handling (`if x is None`, `KeyError`, registry
  fallbacks, parse failures) is not tested.
- **Missing branch coverage**: conditionals where only one side is exercised (e.g.
  `parse_bool` tested for truthy but not for unparseable input that defaults to False).
- **Missing boundary / edge cases**: empty input, `None`, zero, duplicate, out-of-range,
  single-element, Vietnamese-sentinel strings (e.g. `"Không có mệnh đề mới."`), empty
  premise lists — whatever the domain implies.
- **Missing regression test**: a bug fix on the branch with no test capturing the
  previously-broken scenario.

### 3.2 Weak assertions

- **Truthiness-only asserts**: `assert result` or `assert result is not None` where a
  concrete value should be checked (`assert result.answer == "..."`).
- **Call-count trivia**: asserting a mock was called N times instead of asserting the
  observable outcome.
- **Superficial asserts**: asserting a subset of fields when the full contract matters, or
  asserting `isinstance(x, BaseModel)` instead of the field values.
- **Tautological asserts**: `assert foo(x) == foo(x)` — tests the test framework.
- **Implementation-coupled asserts**: asserting private attribute state (`_internal_cache`)
  that would break on a harmless refactor. Assert observable outcomes instead (returned
  values, registry membership, serialized report contents).

### 3.3 Over-mocking

This repo's legitimate mock boundary is the **LM layer** (`dspy.LM` / `BaseLM` backed by an
OpenAI-compatible endpoint). Mock there (stub `dspy.LM` to return canned `Prediction`s, or
use `dspy` testing utilities). Beyond that:

- **Do not mock internal strategy/module internals** — exercise `dspy.Module` subclasses
  through their `forward` with a stubbed LM, not by mocking their private helpers.
- **Do not mock the registries** (`get_task` / `get_strategy` / `get_decoding`) — use the real
  registry with real registered components in tests.
- **Do not mock `pydantic` model validation** — construct real models with real field values.
- Flag tests that stub internal functions where a real call would be cheaper and more honest.

### 3.4 Flaky / non-deterministic patterns

- Tests depending on wall-clock time without a frozen clock / injected timestamp.
- Tests depending on `set` / `dict` ordering for equality (compare sorted lists instead).
- Tests depending on random without a seeded source or `random.seed`.
- Tests relying on shared mutable state across tests without fixture isolation.
- Tests making real network/model-serving calls (should use a stubbed LM).

### 3.5 Structure and naming (repo-specific)

This repo uses **plain pytest functions**, not BDD `describe`/`it`. Do not try to convert tests
to BDD style. Instead check:

- **Test function names** are intention-revealing (`test_parse_bool_unparseable_defaults_false`
  not `test_parse_bool_3`).
- **Parametrize** is used for multi-case functions with explicit `(input, expected)` rows
  rather than a chain of `assert`s in one function.
- **Shared setup** lives in the nearest `conftest.py` or a local fixture, not duplicated across
  functions.
- **Type hints** on test helpers and parametrize row types (the repo uses `object` for
  parametrize input types when the inputs are heterogeneous — see
  `tests/strategies/cr/test_common.py`).
- **Docstrings** only for non-obvious tests; do not add docstrings that restate the function
  name.

## Phase 4: Build the Findings List

Present findings as a numbered table sorted by severity (Critical > High > Medium > Low):

```
## Test Quality Findings

| # | Severity | File:line           | Issue                                      |
|---|----------|---------------------|--------------------------------------------|
| 1 | High     | tests/.../test_x.py | Missing failure path for None input        |
| 2 | Medium   | tests/.../test_y.py | Truthiness-only assert on .answer field    |
| 3 | Medium   | tests/.../test_z.py | Over-mocks internal _resolve() helper      |
```

If no issues are found, say so clearly and stop.

## Phase 5: Fix Weak Tests

For each finding, fix the test directly (prefer fixing the test over changing production code;
only propose production-code changes if the code is genuinely hard to test because of poor
seams, and even then keep it minimal). Apply fixes directly — do not leave TODOs or create
stub files.

When fixing:

- **Add missing edge cases** as new `@pytest.mark.parametrize` rows or new `test_*` functions,
  matching the existing style in the file.
- **Sharpen assertions** to concrete expected values.
- **Replace over-mocks** with real calls through the stubbed-LM boundary.
- **Add regression tests** for branch bug fixes (a `test_<bug>_regression` function that would
  have failed before the fix).
- **Remove flakiness** by controlling time/order/randomness or switching to sorted-list
  compares.

## Phase 6: Verify

Run the affected tests and the project's checks on every modified test file:

```bash
uv run --no-sync pytest tests/path/to/changed_test.py
uv run --no-sync ruff check tests/path/to/changed_test.py
uv run --no-sync ruff format tests/path/to/changed_test.py
```

Pyright runs in strict mode — do not introduce type errors; suppress only genuine false
positives with `# pyright: ignore`.

After all fixes, report a concise summary: what was strengthened, what was added, and confirm
the affected tests pass.
