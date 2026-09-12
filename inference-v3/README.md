# Magnitude inference v3

TileLang-native inference through Magnitensor. Magnitude owns model semantics,
batching, serving, logical state and weight-container interpretation. Magnitensor
owns semantic tensor graphs, lowering selection, memory planning, physical
resources, completion and maximal program submission. TileLang compiles and
executes every numerical kernel through its existing target adapters. The current
path supports dense and routed Qwen 3.5 models from GGUF files or MLX affine
Safetensors directories. Loading MLX-format weights does not load the MLX runtime.

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
uv run pytest tests/magnitensor tests/models
```

Tests marked `device` compile and execute small programs on the selected machine.
TileLang's own suites are under `tilelang/testing/python/<backend>`.

## Fast kernel development

Do not load a model or start the server to debug graph construction, lowering
selection, memory planning or submission composition. `magnitensor.analyze`
performs those compiler passes without allocating storage, generating native
code or executing a kernel. It returns the immutable graph, legal candidates,
selected cover, memory plan, submission units and diagnostics; production
`magnitensor.compile` materializes that same plan.

Use the model qualifier to apply this compile-free path to the actual Qwen model
metadata and weight representations:

```sh
uv run magnitude-qualify \
  --target /path/to/model.gguf-or-mlx-directory \
  --backend metal --contexts 16384,65536 --max-batch 8

# Restrict an iteration to one or more standard shapes.
uv run magnitude-qualify --target /path/to/model \
  --case decode-max-batch --case prefill-2048-logits
```

The standard matrix covers state-only prefill, logits prefill, long prefill,
single-sequence decode and maximum-batch decode. The command exits unsuccessfully
if it finds a scalar contraction fallback, a non-maximal submission, a missing
online attention or recurrent preparation schedule, or an inappropriate dense or
MoE schedule. Its JSON reports graph fingerprints, selected schedule counts,
kernel counts, submission counts and temporary bytes.

Once selection is structurally correct, compile and time only the affected
region:

```sh
uv run magnitude-kernel-bench encoded-linear --mode decode --rows 1
uv run magnitude-kernel-bench encoded-linear --mode decode --rows 1 \
  --width 2560 --output 248320
uv run magnitude-kernel-bench parallel-linear --mode decode --rows 1 \
  --width 2560 --output 8192
uv run magnitude-kernel-bench dense-swiglu --mode prefill --rows 512 \
  --width 2560 --intermediate 9216
uv run magnitude-kernel-bench attention --mode decode --context 65536
uv run magnitude-kernel-bench recurrent-prepare --mode prefill --rows 2048
uv run magnitude-kernel-bench gated-recurrence --mode prefill --rows 2048
uv run magnitude-kernel-bench grouped-experts --mode prefill --rows 512
```

The intended iteration order is:

1. run compile-free qualification for the affected shape;
2. run the focused graph, construction and reference tests;
3. benchmark only the affected schedule;
4. run a tiny numerical whole-model test;
5. load a real model only after those checks pass;
6. run session benchmarks only for final acceptance evidence.

Routine kernel iteration must stop at the earliest layer that disproves the
change. Full model startup and long session workloads are acceptance checks, not
debugging loops.

For shape-sensitive schedules, compare the real model geometry rather than the
small defaults. Independent single-row projections must select
`linear.direct-encoded` (or `linear.direct-dense` for unpacked weights),
adjacent attention and recurrent projections sharing an activation must select
`linear.parallel-direct`, and dense prefill blocks must select
`dense_swiglu.matrix`. Prefill attention must select
`causal_attention.matrix-streaming`, and long decode attention must select
`causal_attention.partitioned`. Partitioned decode processes every query-head
group sharing a KV head in one workgroup so K/V traffic and reduction barriers
are not repeated per query head.

## Measure

`session-bench` measures the actual Magnitensor-backed service with simulated
agent sessions for either GGUF or MLX artifacts:

```sh
uv run session-bench run \
  --target magnitude=/path/to/model.gguf-or-mlx-directory \
  --suite context --context 16384,65536 --prose --repeat 1
```

`python -m performance import RUN_DIRECTORY` imports a completed run into the
content-addressed assessment store. Raw runs go under `runs/`, written-up
baselines under `results/`.
