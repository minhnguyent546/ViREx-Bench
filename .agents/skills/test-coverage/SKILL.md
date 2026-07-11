---
name: test-coverage
description: Measure test coverage for the virex_bench package, rank low-coverage files, identify untested branches and edge cases, then propose and write meaningful tests to fill gaps. Use when asked to check or improve test coverage, find missing tests, or fill coverage gaps.
---

# Test Coverage: Measure, Find Gaps, Fill Them

Run coverage on the `virex_bench` package, rank files by coverage, identify the highest-risk
gaps, propose tests that map to real behavior, and write them after user approval.

## Project Context

ViREx-Bench is a Vietnamese multi-step-reasoning benchmark built on **DSPy**, organized around
four axes — task / model / strategy / decoding — plus shared modules. See `AGENTS.md` for the
full layout and the testing conventions to follow.

Boundaries (from `AGENTS.md`):

- **Do not edit** files under `data/` (datasets managed externally) or `uv.lock` by hand.
- **Do not modify** generated/cached dirs: `.ruff_cache/`, `.cache/`, `_testing_results/`.
- **Never commit** secrets or `.env` contents.
- Ignore files/dirs starting with `_` (e.g. `_testing_results/`, `_test_format.py`) — local
  files, not part of the main codebase.

## Coverage Tooling

`coverage` and `pytest-cov` are in the `dev` dependency group, and the coverage config lives
in `pyproject.toml` under `[tool.coverage.run]` / `[tool.coverage.report]`:

- `source = ["virex_bench"]` — scopes measurement to the package (not tests, scripts, notebooks).
- `omit = ["tests/*"]` — excludes test files from measurement.
- `show_missing = true` + `sort = "-Cover"` — `term-missing` reports uncovered line numbers,
  sorted by coverage descending.
- `exclude_also` — excludes `if TYPE_CHECKING:` blocks, `@abstractmethod`, `raise
  NotImplementedError`, and `logger.debug` calls from coverage (these are not real runtime
  paths worth testing).

So `--cov=virex_bench --cov-report=term-missing` is enough; the rest is read from config.
`pytest-asyncio` is configured with `asyncio_mode = "auto"` (async test functions are
auto-marked, no `@pytest.mark.asyncio` needed) and `asyncio_default_fixture_loop_scope =
"session"` (one event loop for the whole session).

## Phase 1: Run Coverage

Run the full suite with coverage:

```bash
uv run --no-sync pytest --cov=virex_bench --cov-report=term-missing
```

`testpaths = ["tests"]` is set in `pyproject.toml`, so pytest discovers tests under `tests/`
automatically — no need to pass `tests/` explicitly. For a machine-readable breakdown (useful
when ranking many files), add `--cov-report=json:coverage.json`.

If the suite is slow or partially failing, scope to a subtree for a first pass
(`uv run --no-sync pytest --cov=virex_bench --cov-report=term-missing tests/strategies/`)
but always report which scope you ran.

## Phase 2: Rank and Identify Gaps

Build a ranked list of files by coverage percentage (lowest first). For each file with gaps,
classify the gap type:

1. **Untested module** — a source file under `virex_bench/` with no corresponding test file
   under `tests/` (the test tree mirrors the package layout).
2. **Uncovered function** — a public function/method with 0% coverage.
3. **Uncovered branch** — a function with partial coverage (conditionals, error paths,
   `if x is None` guards, registry fallbacks).
4. **Missing edge case** — a function whose tests only hit the happy path (e.g. `parse_bool`
   is tested for truthy/falsy but not for unparseable input).

Cross-reference with risk: registry lookups (`get_task` / `get_strategy` / `get_decoding` /
`get_metric` / `build_judge`), the evaluation orchestrator (`evaluation/evaluate.py`), and the
search algorithms (`strategies/tot/search/`) are higher-priority than pure data containers
under `types/`.

Present the ranked gap list to the user as a table before writing anything:

```
File                                          | Cov  | Gap type              | Risk
----------------------------------------------|------|-----------------------|------
virex_bench/strategies/tot/search/mcts.py     | 23%  | Uncovered branch      | HIGH
virex_bench/evaluation/evaluate.py            | 45%  | Missing edge case     | HIGH
virex_bench/decoding/self_consistency.py      | 61%  | Uncovered function    | MED
...
```

## Phase 3: Propose Tests

For each gap (highest risk first), draft a test idea with:

- **Scenario**: what behavior is being exercised (in user/observable terms, not "hit line 42").
- **File under test**: the source module and the specific function/branch.
- **Test file**: the target path under `tests/` mirroring the package layout.
- **Expected coverage gain**: rough percentage delta.

Present the proposal as a numbered list and **ask the user for approval before writing any
test**. Pause until they agree. Do not edit test files before approval.

## Phase 4: Write Tests

After approval, write the tests following the existing conventions in this repo:

- **Plain pytest functions** — `def test_<behavior>() -> None:`. Do not introduce BDD-style
  `describe`/`it` blocks or class-based grouping; this repo uses plain functions.
- **Parametrize for multi-case functions** — use `@pytest.mark.parametrize` with explicit
  `(input, expected)` rows (see `tests/strategies/cr/test_common.py` for the style).
- **Type hints** on test helpers and parametrize row types.
- **Docstrings** only for non-obvious tests (e.g. "An unparseable verifier verdict defaults
  to the conservative False.") — do not narrate the obvious.
- **Mirror the package layout** under `tests/` — `virex_bench/strategies/tot/search/mcts.py`
  → `tests/strategies/tot/test_mcts_search.py`.
- **DSPy seams** — LM calls go through `dspy.LM` / `BaseLM` backed by an OpenAI-compatible
  endpoint. Mock the LM at that boundary (e.g. with `dspy.utils.dummies` or a stub `dspy.LM`
  returning canned `Prediction`s), never by mocking internal strategy internals. The
  strategies are `dspy.Module` subclasses; exercise them through their `forward` with a
  stubbed LM rather than reaching into their private helpers.
- **Fixtures** — shared fixtures go in the nearest `conftest.py`; keep tests isolated and
  deterministic (no real network, no real model serving, no timing-dependent asserts).

### Banned test patterns (do not write these)

- **Coverage-padding**: asserts solely `is not None` / `!= ""` to inflate metrics — assert a
  specific expected value.
- **Zero-assertion smoke tests**: `def test_foo(): foo()` — proves nothing except "doesn't
  panic".
- **Tautological asserts**: `assert foo(x) == foo(x)` — tests the framework, not the code.
- **Implementation-coupled asserts**: asserting private attribute state that would break on a
  harmless refactor. Assert observable outcomes (returned values, registry membership, report
  contents).
- **Flaky tests**: depending on timing, network, dict/set ordering, or random without a
  seeded source.

## Phase 5: Verify

After writing, re-run coverage on the affected scope and confirm the gain:

```bash
uv run --no-sync pytest --cov=virex_bench --cov-report=term-missing tests/path/to/changed_test.py
```

Then run the project's checks on every new/modified test file:

```bash
uv run --no-sync ruff check path/to/test_file.py
uv run --no-sync ruff format path/to/test_file.py
```

Pyright runs in strict mode — do not introduce type errors; suppress only genuine false
positives with `# pyright: ignore`.

Report the before/after coverage delta and note any remaining gaps (e.g. dead code, truly
unreachable branches) that cannot be reasonably tested.
