# @magnitudedev/scratchpad

Scratchpad path expansion — resolves the `$M` token in path strings to the scratchpad root directory.

## Purpose

The scratchpad is a private workspace directory available to the agent at runtime. Paths referencing it use the `$M` or `${M}` token as a prefix. This package provides the function that expands those tokens into real filesystem paths, while enforcing that the resolved path stays within the scratchpad boundary.

## Expansion behavior

### Tokens

Two token forms are recognized:

- `$M` — short form
- `${M}` — brace form (for contexts where `$M` may be ambiguous)

### Patterns that expand

| Input | Resolves to |
|---|---|
| `$M` or `${M}` | Scratchpad root |
| `$M/foo` or `${M}/foo` | Scratchpad root + `/foo` |
| `./$M/foo` or `./${M}/foo` | Scratchpad root + `/foo` (leading `./` stripped) |
| `../$M/foo` or `../${M}/foo` | Scratchpad root + `/foo` (leading `../` stripped) |
| Any mix of leading `./` and `../` before the token | Scratchpad root + `/foo` |

Leading dot-segments before the `$M` token are stripped so that paths like `./$M/reports/file.md` work as the agent naturally writes them. The token is what matters, not what precedes it.

### Patterns that do **not** expand

Any path that does not contain an `$M` or `${M}` token is returned unchanged. This includes:

- Relative project paths (`src/index.ts`, `./src/foo.ts`, `../lib/bar.ts`)
- Absolute paths (`/etc/passwd`)

These are left for the caller to resolve relative to the working directory.

## Security: boundary enforcement

After expansion, the resolved absolute path must stay within the scratchpad root. If it escapes, the original input is returned unchanged — the expansion simply does not happen, and the caller's policy layer handles it as an out-of-bounds path.

This blocks:

- **Traversal above root**: `$M/../etc/passwd` — the `..` resolves above the scratchpad, so expansion is rejected
- **Double-slash escape**: `$M//etc/passwd` — the double slash normalizes to an absolute path (`/etc/passwd`), which escapes the scratchpad root
- **Any other path trick** that causes the resolved result to fall outside the scratchpad directory

### Safe normalization within bounds

Path segments that stay within the scratchpad are normalized safely:

- `$M/a/../b` → scratchpad root + `/b` (the `a/..` cancels out)
- `$M/a/./b` → scratchpad root + `/a/b` (the `./` is dropped)
- `$M/foo/` → scratchpad root + `/foo` (trailing slash removed)

## Return value

The function returns a structured result with three fields:

- **path** — the resolved absolute path (or the original input if not expanded)
- **expanded** — whether `$M` expansion was performed
- **displayPath** — a human-readable path relative to the scratchpad root; for non-expanded paths, this is the original input

The `expanded` flag lets callers distinguish "this was a scratchpad path" from "this was a project path" without re-checking the input string.
