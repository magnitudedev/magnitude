# Terminal Bench 2 Evaluation

Run Magnitude on [Terminal Bench 2](https://www.tbench.ai/) via the [Harbor](https://github.com/harbor-framework/harbor) framework.

## Overview

Terminal Bench 2 (TB2) is a benchmark for evaluating AI agents on realistic, end-to-end CLI tasks in sandboxed Docker containers. Each task provides:

- A Docker environment with pre-loaded files
- A natural language instruction
- Test scripts that verify the final container state

Harbor is the official harness for running TB2. It manages container lifecycle, agent execution, and verification.

## Files

| File | Purpose |
|------|---------|
| `magnitude_agent.py` | Harbor adapter — uploads the prebuilt Magnitude binary into TB2 containers and invokes it |
| `Dockerfile.build` | Docker image definition for building the Linux x64 Magnitude binary |
| `build-linux.sh` | Helper script to build the Docker image and extract `evals/tbench/bin/magnitude` |
| `bin/` | Local output directory for the built Linux x64 binary (ignored by git) |

## Prerequisites

- **Docker** running locally (e.g. OrbStack, Docker Desktop)
- **Harbor** installed: `pip install harbor` or `uv tool install harbor`
- **API key** for your chosen provider (e.g. `ANTHROPIC_API_KEY`)

## Quick Start

```bash
# 1. Build the Linux binary (from repo root)
./evals/tbench/build-linux.sh

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-...

# 3. Run on a single TB2 task
harbor run -d terminal-bench@2.0 \
  --agent-import-path evals.tbench.magnitude_agent:MagnitudeAgent \
  --model anthropic/claude-sonnet-4-6 \
  -l 1

# 4. Run the full benchmark (concurrent)
harbor run -d terminal-bench@2.0 \
  --agent-import-path evals.tbench.magnitude_agent:MagnitudeAgent \
  --model anthropic/claude-sonnet-4-6 \
  -n 4
```

## Building the Linux Binary

Magnitude's CLI binary is built with Bun's native compiler. Since TB2 containers run Linux x64, build the binary in Docker from the repo root:

```bash
./evals/tbench/build-linux.sh
```

The script will:

1. Build `evals/tbench/Dockerfile.build` with the repo root as the Docker build context
2. Run `bun install` inside the image
3. Run `bun run cli/scripts/build-binary.ts bun-linux-x64`
4. Extract the resulting binary to `evals/tbench/bin/magnitude`
5. Mark the extracted binary executable

If `evals/tbench/bin/magnitude` exists, the adapter uploads it directly into the container (fast). Otherwise, it falls back to installing Magnitude via npm inside the container.

## How It Works

1. **Harbor** spins up a Docker container per task with the task environment pre-loaded
2. **Setup**: The adapter uploads `evals/tbench/bin/magnitude` into the container at `/usr/local/bin/magnitude`
3. **Run**: Harbor executes `magnitude --oneshot --provider <id> --model <id> "<instruction>"`
4. **Magnitude** operates on the container filesystem — reads/edits files, runs shell commands
5. **Verification**: Harbor runs the task's test scripts against the final container state

## Configuration

### Provider and model

The `--model` / `-m` flag uses `provider/model-id` format. The adapter extracts the provider and model automatically:

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-...
harbor run -d terminal-bench@2.0 \
  --agent-import-path evals.tbench.magnitude_agent:MagnitudeAgent \
  -m anthropic/claude-sonnet-4-6

# OpenRouter
export OPENROUTER_API_KEY=sk-...
harbor run -d terminal-bench@2.0 \
  --agent-import-path evals.tbench.magnitude_agent:MagnitudeAgent \
  -m openrouter/anthropic/claude-sonnet-4

# OpenAI
export OPENAI_API_KEY=sk-...
harbor run -d terminal-bench@2.0 \
  --agent-import-path evals.tbench.magnitude_agent:MagnitudeAgent \
  -m openai/gpt-5.3-codex
```

### Useful flags

| Flag | Description |
|------|-------------|
| `-l <n>` | Limit to n tasks |
| `-n <n>` | Run n tasks concurrently (default: 4) |
| `-t <pattern>` | Run specific task(s) by name/glob |
| `-x <pattern>` | Exclude task(s) by name/glob |
| `--timeout-multiplier <f>` | Scale task timeouts |
| `--debug` | Enable debug logging |

## Verifying the Setup

Run the oracle agent (no API key needed) to confirm Harbor + TB2 works:

```bash
harbor run -d terminal-bench@2.0 -a oracle -l 1
```

Expected output: 1 trial, reward = 1.0, 0 errors.