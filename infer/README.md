# Local MLX Inference for Magnitude

Run a local OpenAI-compatible inference server on Apple Silicon using [vllm-mlx](https://github.com/waybarrios/vllm-mlx).

Default model: `mlx-community/Qwen3.5-27B-6bit`

## Prerequisites

- Apple Silicon Mac with 32GB+ unified memory
- [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
cd infer
uv run serve.py       # install deps, download model on first run, start server
```

Server will be available at `http://127.0.0.1:8012/v1`.

## Configuration

All settings are via environment variables:

| Variable | Default | Description |
|---|---|---|
| `INFER_MODEL` | `mlx-community/Qwen3.5-27B-6bit` | HuggingFace model ID |
| `INFER_HOST` | `127.0.0.1` | Bind address |
| `INFER_PORT` | `8012` | Port |
| `INFER_EXTRA_ARGS` | | Extra CLI flags for `vllm-mlx serve` |

Example:

```bash
INFER_PORT=9000 uv run serve.py
```

## Verify

```bash
curl http://127.0.0.1:8012/v1/models
```