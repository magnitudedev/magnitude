# Magnitude Evals

LLM evaluation framework for testing model behavior in the Magnitude agent environment.

## Quick Start

```bash
cd evals

# Interactive mode
bun eval

# Run against a specific model
bun eval run prose -m anthropic:claude-sonnet-4-6

# All default models
bun eval run prose --all-models

# Run each scenario 3 times for consistency checking
bun eval run prose -m openai:gpt-5.3-codex -r 3
```

## Commands

### `bun eval run [eval-id]`

```bash
# Multiple models
bun eval run prose -m anthropic:claude-sonnet-4-6 -m openai:gpt-5.3-codex

# Specific scenarios only
bun eval run prose -m anthropic:claude-sonnet-4-6 -s write-code-file -s edit-code-file

# Repeat each scenario N times (consistency/flakiness testing)
bun eval run prose -m openai:gpt-5.3-codex -r 5

# Quiet mode (one line per scenario)
bun eval run prose -m anthropic:claude-sonnet-4-6 -q

# JSON output
bun eval run prose -m anthropic:claude-sonnet-4-6 --json

# Custom concurrency (default: 4)
bun eval run prose -c 8

# Skip saving results
bun eval run prose --no-save
```

**Options:**
| Flag | Description |
|------|-------------|
| `-m, --model <spec>` | Model as `provider:model` (repeatable) |
| `-s, --scenario <id>` | Run specific scenarios only (repeatable) |
| `-r, --repeat <n>` | Run each scenario N times per model (default: 1) |
| `-c, --concurrency <n>` | Max parallel scenarios (default: 4) |
| `-q, --quiet` | One line per scenario |
| `--json` | JSON output |
| `--all-models` | Run all default models |
| `--no-save` | Don't save results to disk |

### `bun eval list`

List available evaluations and their scenarios.

## Default Models

| Provider | Model |
|----------|-------|
| Anthropic | `anthropic:claude-sonnet-4-6` |
| OpenAI | `openai:gpt-5.3-codex` |
| Google | `google:gemini-3-pro-preview` |
| Google | `google:gemini-3-flash-preview` |
| MiniMax | `minimax:MiniMax-M2.5` |
| Z.AI | `zai:glm-4.7` |

Uses your configured Magnitude provider auth (OAuth subs, API keys from `~/.magnitude/config.json`) — same credentials as the main agent.

## Results

Saved to `evals/results/<timestamp>/` after each run:

```
results/2026-02-17T13-30-00/
  summary.md           # Overview + per-scenario breakdown
  anthropic-claude-sonnet-4-6.md  # Full report per model
  results.json         # Raw structured data
```

## Available Evals

### `prose` — Prose Delimiter Escaping

Tests whether models correctly use prose delimiters and avoid incorrect escaping. Runs responses through the **real js-act sandbox** (QuickJS WASM) with mock tools — same execution pipeline as production.

**9 scenarios:**

| ID | What it tests |
|----|---------------|
| `write-code-file` | fs.write with template literals |
| `write-markdown` | fs.write with triple-backtick code fences |
| `edit-code-file` | fs.edit replacing string concat with template literals |
| `update-task-markdown` | updateTask with markdown code blocks |
| `shell-special-chars` | Shell commands with `$`, pipes, globs |
| `combined-write` | fs.write + message in one response |
| `message-with-backticks` | message() explaining code with inline backticks |
| `nested-template-literals` | Tagged templates and nested interpolation |
| `multiline-message` | Multi-line message — detects literal `\n` vs real newlines |

## Tests

```bash
cd evals && bun test
```
