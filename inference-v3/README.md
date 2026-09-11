# Magnitude inference v3

TileLang-native inference with separate model, weight-operation, continuation,
state-storage and execution-owner contracts. The current performance path is
Qwen3.5-4B using the POC's MLX affine Q4/group-64 Safetensors artifact, with BF16
activations and KV. Loading MLX-format weights does not load the MLX runtime.

TileLang is the git submodule `tilelang`, on the `magnitude` branch of
`magnitudedev/tilelang`: upstream TileLang plus the Metal changes Magnitude
needs (BF16 code generation, reductions, logical GEMM fragments, host-derived
argument binding and kernel caching for the torch adapter, default loop
unrolling). `pyproject.toml` installs it as an editable path dependency. On
macOS, install the Xcode command-line tools and CMake, then from this directory:

```sh
git submodule update --init --recursive
USE_METAL=ON USE_CUDA=OFF USE_ROCM=OFF CMAKE_BUILD_PARALLEL_LEVEL=6 uv sync
```

The first sync builds TileLang from source; [AGENTS.md](AGENTS.md) describes
how to change TileLang, pull upstream, and send changes upstream. Every backend, Metal included,
compiles and executes kernels through `tilelang.compile` and TileLang's
supported execution adapter for the target; on Metal that is the torch adapter
over `torch.mps.compile_shader`. Magnitude owns tensor descriptions, storage
budgets and lifetimes, prepared invocations, ordering, completion and model
state in `DeviceContext`; the shared `TorchDriver` implements those contracts
with Torch storage, MPS events and TileLang kernels. There is no application
compiler, shader launcher or compiler artifact cache: persistent kernel reuse
is TileLang's kernel cache under `~/.tilelang/cache`. Target scheduling
defaults, such as loop unrolling on Metal, live in TileLang's target pipeline;
Magnitude passes no per-target compiler configuration. The only remaining
compiler workaround is the LLVM BF16 legalization in `TorchDriver.compile`,
which is unrelated to Metal.

Run the server from this directory:

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
Metal is the current validation target.

The [session benchmark](session-bench.md) and context fixtures were copied verbatim
from v2. Run those commands from this directory despite the copied document's v2
wording. The copy's checksums are recorded in
[the source lock](specs/26-09-09/v2-benchmark-source-lock.json). Component observations
use the common typed performance runner; model comparisons require an explicitly
identified artifact, token history, numerical family and reference.
