# Magnitude inference v3

TileLang-native inference. Every kernel is portable TileLang and every backend,
Metal included, compiles and executes through TileLang; Magnitude owns storage,
lifetimes, ordering, completion and model state, and passes no per-target compiler
configuration. The current path is Qwen 3.5 dense from a GGUF file or an MLX
affine Safetensors directory, with BF16 activations and KV. Loading MLX-format
weights does not load the MLX runtime.

[design/architecture.md](design/architecture.md) is the map of the engine; the
documents beside it own each component.

## Setup

TileLang is the git submodule `tilelang`, branch `magnitude` of
`magnitudedev/tilelang`; its `3rdparty/tvm` is `magnitudedev/tvm`. Both are
forks that carry our changes ahead of upstream. `pyproject.toml` installs
TileLang as an editable path dependency. On macOS, install the Xcode command line
tools and CMake, then from this directory:

```sh
git submodule update --init --recursive
USE_METAL=ON USE_CUDA=OFF USE_ROCM=OFF CMAKE_BUILD_PARALLEL_LEVEL=12 uv sync
```

The first sync builds TileLang from source. [AGENTS.md](AGENTS.md) describes how
the forks are changed, tested and sent upstream. Persistent kernel reuse is
TileLang's cache under `~/.tilelang/cache`; set `TILELANG_DISABLE_CACHE=1` when a
compiler change must be observed.

The host backend needs an LLVM-enabled build: add `USE_LLVM=ON` with an
`llvm-config` from a release TileLang's TVM supports on the path. It is a
correctness target, not a performance one.

## Run

```sh
uv run --frozen python -m magnitude_engine.serving \
  --target /path/to/model.gguf-or-mlx-directory \
  --backend metal --memory-bytes 8589934592 \
  --context-tokens 131072 --prefill-tokens 512
```

The server exposes an OpenAI-style chat endpoint. `--max-active`, `--max-queued`
and `--output-capacity` size the continuous service. The composition it built,
its digest and the artifact identity are reported in the server properties.

## Test

```sh
uv run pytest tests
MAGNITUDE_TEST_BACKEND=llvm uv run pytest tests/kernels
```

Engine tests take the exclusive-measurement lock, so one run owns the machine at
a time. TileLang's own suites are under `tilelang/testing/python/<backend>`.

## Measure

`python -m performance <composition.json> --workload <workload.json> --output <record.json>`
benchmarks one component built from a composition and writes a self-describing
record. [session-bench.md](session-bench.md) measures serving with simulated agent
sessions. Raw runs go under `runs/`, written-up baselines under `results/`.
