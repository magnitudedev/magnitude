# xml-act grammar experiment

Quick experiment for grammar-constrained xml-act generation with Outlines and MLX.

## Setup

Run from the repo root:

```bash
mkdir -p experiments/xml-act-grammar
cd experiments/xml-act-grammar
uv sync
```

## Generate once

```bash
uv run python generate.py
uv run python generate.py --prompt "Write a hello world in Python"
```

## Server mode

```bash
uv add fastapi uvicorn
uv run python serve.py
```

Then send an OpenAI-style request to `POST /v1/chat/completions` with a `messages` array and optional `model` field.

## Files

- `grammar.lark` — Lark grammar for a single xml-act turn
- `generate.py` — CLI generator using a CFG constraint
- `serve.py` — minimal FastAPI wrapper
- `pyproject.toml` — project metadata for `uv`