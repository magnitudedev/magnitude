"""Benchmark-owned policy. These limits are deliberately not user configuration."""

from pathlib import Path

MAX_OUTPUT_TOKENS = 32_768
PROSE_OUTPUT_TOKENS = 256
RETRIEVAL_OUTPUT_TOKENS = 1_024
REQUEST_TIMEOUT_SECONDS = 1_800
STARTUP_TIMEOUT_SECONDS = 900
PREFILL_TOKENS = 512
OUTPUT_CAPACITY = 64
DEFAULT_CONTEXTS = (1_024, 4_096, 16_384)
ENGINES = ("magnitude", "mlx-vlm", "omlx", "llama.cpp")


def project_root() -> Path:
    """The checkout owns aliases and results, regardless of the caller's working directory."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("session-bench must run from its inference-v2 source checkout")
    return root


# Shared capacity alignment accommodates upstream KV allocation granularity.
CONTEXT_ALIGNMENT = 256
