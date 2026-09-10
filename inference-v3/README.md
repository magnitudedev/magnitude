# Magnitude inference v3

TileLang-native inference with separate model, weight-operation, continuation,
state-storage and execution-owner contracts. The current performance path is
Qwen3.5-4B using the POC's MLX affine Q4/group-64 Safetensors artifact, with BF16
activations and KV. Loading MLX-format weights does not load the MLX runtime.

From this directory, on the configured development installation:

```sh
uv run --frozen python -m magnitude_engine.serving \
  --target /path/to/Qwen3.5-4B-4bit \
  --backend metal --memory-bytes 8589934592 \
  --context-tokens 131072 --prefill-tokens 512
```

The server uses the shared blueprint, model and continuous-service path. The target
may also be a GGUF file, selecting its separate weight-operation implementation.
The matched POC artifact is `mlx-community/Qwen3.5-4B-4bit`, revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`.

[Implementation progress](specs/26-09-09/implementation-progress.md) records current
measurements and limitations. [Architecture](specs/26-09-09/architecture-proposal.md)
describes the full intended engine; it is broader than the implemented text path.
Metal is the current validation target. The existing compiler patches and Metal
runtime remain part of this development installation; portable compiler/runtime
integration and reproducible packaging remain open.

The [session benchmark](session-bench.md) and context fixtures were copied verbatim
from v2. Run those commands from this directory despite the copied document's v2
wording. The copy's checksums are recorded in
[the source lock](specs/26-09-09/v2-benchmark-source-lock.json). Component observations
use the common typed performance runner; model comparisons require an explicitly
identified artifact, token history, numerical family and reference.
