"""Benchmark-owned policy. These limits are deliberately not user configuration."""

from pathlib import Path

BENCH = Path(__file__).resolve().parents[2]
INFERENCE = BENCH.parents[1]
RUNTIMES = BENCH / "runtimes"

MODES = ("prose-repeat", "prose-continue", "tools", "retrieval")
ENGINES = ("magnitude", "llama.cpp", "ollama", "ollama-mlx", "ollama-registry", "mlx-vlm", "omlx")

#: Independent sessions a run can offer at once (the concurrency section's widest point).
LANES = 8
#: Sampling seed wherever an engine supports one.
SEED = 42
REQUEST_TIMEOUT_SECONDS = 1_800
STARTUP_TIMEOUT_SECONDS = 900
#: MLX-VLM prefill step and the most output any fixture family asks for (tools).
PREFILL_TOKENS = 2048
SERVER_MAX_OUTPUT_TOKENS = 32_768
#: Shared capacity alignment accommodates upstream KV allocation granularity.
CONTEXT_ALIGNMENT = 256
