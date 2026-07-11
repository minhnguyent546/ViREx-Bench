---
name: generate-release-notes
description: Generate grouped, user-facing release notes from a range of git commits.
---

# Generate Release Notes

Turn a range of git commits into concise, grouped release notes. Commits follow
Conventional Commits (`type(scope): subject`); the notes reorganize them into
user-facing sections, drop noise, and cite each commit hash.

## Phase 1: Resolve the Commit Range

Figure out which commits to summarize:

- If the user names a range (`v0.1.1..HEAD`, `abc123..def456`, "since the last release"),
  use it directly. For "since the last release", resolve the latest tag with
  `git describe --tags --abbrev=0` and use `<tag>..HEAD`.
- If the user names a starting point only, use `<ref>..HEAD`.
- If nothing is specified, ask which range to cover (or default to commits since the
  latest tag) before proceeding — don't guess.

Then read the commits **oldest to newest** so the notes track the order changes landed:

```bash
git log --reverse --pretty=format:'%h %s' <range>
```

Use `--pretty=format:'%h %s%n%b'` if you need commit bodies to disambiguate what a
change actually does. Prefer reading the diff (`git show <hash>`) for any commit whose
subject is vague before writing its line.

## Phase 2: Categorize Commits

Map each commit to a section by its Conventional Commit type and scope. Section order
and rules:

1. **Features** — `feat:` commits (excluding `feat(scripts)`). New user-facing capability.
2. **Scripts** — any commit scoped to `scripts` regardless of type (`feat(scripts)`,
   `fix(scripts)`, etc.). Serving launchers and eval-script changes live here, not under
   Features/Fixes.
3. **Fixes** — `fix:` commits (excluding `fix(scripts)`).
4. **Docs** — `docs:` commits.

Omit a section entirely if it has no commits. Additional sections (e.g. **Build**,
**CI**, **Refactor**) may be added only if the range contains commits the user cares
about there; by default fold routine `chore:`, `build:`, `ci:`, `test:`, and version-bump
commits **out** of the notes unless the user asks to include them. Merge commits are
always dropped.

## Phase 3: Write the Notes

For each commit, write one bullet describing the change from a **user's perspective**,
not a restatement of the commit subject. Guidelines:

- Lead with the concrete change and, where it adds clarity, the mechanism — e.g. a new
  env var, prefix, or flag — using `inline code` for identifiers, model names, flags,
  and env vars.
- End every bullet with the short commit hash in parentheses: `(d940208)`.
- **Group related commits into a single bullet** when they form one logical change (e.g.
  three commits each adding one serving script → one bullet listing all three), and cite
  every hash: `(b5a8931, b09a304, d9c2662)`.
- Keep bullets tight — one line each where possible. Don't pad with "This commit…".
- Preserve oldest-to-newest ordering within each section.

### Format

Sections are bold labels followed by a bulleted list, in the order given in Phase 2:

```
**Features:**
- <user-facing description with `identifiers`> (<hash>)

**Scripts:**
- <description> (<hash>)
- <grouped description> (<hash>, <hash>, <hash>)

**Fixes:**
- <description> (<hash>)

**Docs:**
- <description> (<hash>)
```

## Phase 4: Deliver

Output the release notes directly in the response as a single markdown block the user can
copy. Do **not** write a file unless the user explicitly asks for one. If the range was
ambiguous or you dropped commits you were unsure about, note that briefly after the notes.
