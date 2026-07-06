---
name: simplify
description: Review changed code for reuse, quality, and efficiency, then fix any issues found.
---

# Simplify: Code Review and Cleanup

Review all changed files for reuse, quality, and efficiency. Fix any issues found.

## Project Context

ViREx-Bench is a Vietnamese multi-step-reasoning benchmark built on **DSPy**, organized
around four axes — task / model / strategy / decoding — plus shared modules. See `AGENTS.md`
for the full layout. When reviewing and fixing, respect these boundaries:

- **Do not edit** files under `data/` (datasets managed externally) or `uv.lock` by hand.
- **Do not modify** generated/cached dirs: `.ruff_cache/`, `.cache/`, `_testing_results/`.
- **Never introduce** committed secrets or `.env` contents.

## Phase 1: Identify Changes

Run `git diff` (or `git diff HEAD` if there are staged changes) to see what changed. If there are no git changes, review the most recently modified files that the user mentioned or that you edited earlier in this conversation.

Ignore directories and files starting with `_` (e.g. `_testing_results/`, `_test_format.py`) — these are local/testing files, not part of the main codebase.

## Phase 2: Launch Three Review Agents in Parallel

Use the Agent tool to launch all three agents concurrently in a single message. Pass each agent the full diff so it has the complete context.

### Agent 1: Code Reuse Review

For each change:

1. **Search for existing utilities and helpers** (`rg`) that could replace newly written code. Common locations: `virex_bench/utils.py`, `logger.py`, per-package `common.py` (e.g. `strategies/cr/common.py`, `strategies/tot/common.py`), and files adjacent to the changed ones.
2. **Flag any new function that duplicates existing functionality.** Suggest the existing function to use instead — e.g. reimplementing `premises_to_text`, ad-hoc logger setup instead of `init_logger`, or a new registry lookup that duplicates `get_task` / `get_strategy` / `get_decoding`.
3. **Flag any inline logic that could use an existing utility** — hand-rolled string manipulation, manual `os.path` handling, direct `os.environ` reads that should go through `virex_bench/envs.py`, and similar patterns are common candidates.

### Agent 2: Code Quality Review

Review the same changes for hacky patterns:

1. **Redundant state**: state that duplicates existing state, cached values that could be derived
2. **Parameter sprawl**: adding new parameters to a function instead of generalizing or restructuring existing ones; small helper methods that are referenced only once (inline them)
3. **Copy-paste with slight variation**: near-duplicate code blocks that should be unified with a shared abstraction
4. **Leaky abstractions**: exposing internal details that should be encapsulated, or breaking existing abstraction boundaries (e.g. bypassing the task/model/strategy/decoding registries, or reaching into a `dspy.Module`'s internals)
5. **Stringly-typed code**: raw strings/dicts where a `pydantic.BaseModel`, enum, or string-union already exists; `dataclasses` where the codebase uses `pydantic.BaseModel`
6. **Style mismatches**: truthy/falsy checks (`if x`) where `is None` / `is not None` is required for valid falsy values; `pathlib.Path` where the codebase uses `str` + `os.path`; short/abbreviated names (`ri`, `ip`, `el`) where verbose self-documenting names are the convention; `%`-style logging instead of f-strings; `print()` for diagnostics instead of the shared logger; `type[T]` misuse (passing an instance where a class is expected, e.g. `signature: type[dspy.Signature]`)
7. **DSPy convention drift**: new strategies that aren't `dspy.Module` subclasses or don't follow the existing signature/module patterns in `strategies/`; signature docstrings or field descriptions written in a language other than **English** (task data stays native, but the instruction contract is English)
8. **Unnecessary or over-verbose comments**: comments explaining WHAT the code does (well-named identifiers already do that), narrating the change, or referencing the task/caller — delete these; keep only non-obvious WHY (hidden constraints, subtle invariants, workarounds). For comments worth keeping, trim multi-line/rambling explanations down to a single concise line.

### Agent 3: Efficiency Review

Review the same changes for efficiency:

1. **Unnecessary work**: redundant computations, repeated file reads, duplicate LM/API calls, N+1 patterns
2. **Missed concurrency**: independent operations run sequentially when they could run in parallel (e.g. per-example LM calls, independent propose/evaluate steps in ToT/CR)
3. **Hot-path bloat**: new blocking work added to the per-example evaluation loop or per-LM-call path
4. **Unnecessary existence checks**: pre-checking file/resource existence before operating (TOCTOU anti-pattern) — operate directly and handle the error
5. **Memory**: unbounded data structures, missing cleanup (e.g. accumulating full LM transcripts across a large eval run)
6. **Overly broad operations**: reading entire files/datasets when only a portion is needed, loading all items when filtering for one

## Phase 3: Fix Issues

Wait for all three agents to complete. Aggregate their findings and fix each issue directly. If a finding is a false positive or not worth addressing, note it and move on — do not argue with the finding, just skip it.

After fixing, confirm the changed files still pass the project's checks:

```bash
uv run --no-sync ruff check path/to/file.py
uv run --no-sync ruff format path/to/file.py
```

Pyright runs in strict mode — do not introduce type errors, and only suppress with `# pyright: ignore` for genuine false positives, never to silence a legitimate error. If the change touches behavior, also run the relevant tests (`uv run --no-sync pytest tests/...`, which mirror the package layout).

When done, briefly summarize what was fixed (or confirm the code was already clean).
