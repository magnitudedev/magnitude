# Magnitude inference v3

TileLang-native inference through Ops. Magnitude owns model semantics,
batching, serving, logical state and weight-container interpretation. Ops
owns formulas, operation composition, memory planning, physical
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
uv run --frozen python -m engine.serving \
  --target /path/to/model.gguf-or-mlx-directory \
  --backend metal --memory-bytes 8589934592 \
  --context-tokens 131072 --prefill-tokens 512
```

The server exposes an OpenAI-style chat endpoint. `--max-active`, `--max-queued`
and `--output-capacity` size the continuous service. The composition it built,
its digest and the artifact identity are reported in the server properties.

Vision requests use OpenAI-style `image_url` content parts containing inline
base64 data URLs, alongside text parts. The Qwen 3.5 MLX artifact must include
its vision weights and processor configuration. Image order, repeated images,
and images in follow-up turns are preserved. The current input contract supports
single-frame images with `detail: "auto"`; remote URLs and video are rejected.
Image preprocessing and encoder loading are lazy for text-only service.

## Test

```sh
uv run pytest tests
uv run pytest tests/ops tests/models
```

Tests marked `device` compile and execute small programs on the selected machine.
TileLang's own suites are under `tilelang/testing/python/<backend>`.

## Formula development

The connected measurement system has qualified a production hybrid formula tree,
an actual-GGUF FFN subtree, source-path observations, and a real edited-operation
TUI cycle. Every measured region links formula work and ideal boundary traffic to
device resource references and its own observation. Full-model throughput and the
remaining engine work are separate, unfinished milestones.

The development interface is the persistent `ops.Lab`. It measures typed formula
occurrences through the production operation, checks the independent reference,
and publishes comparable history automatically. The Textual client uses that same
worker and store. A selected measurement includes recurring I/O, transfers,
allocations, kernels, completion and transient cleanup.

```python
from pathlib import Path
import ops
from ops.lab import Configuration, show
from engine.models.qwen35.formulas import DecoderFormulas

# definition is the existing Qwen ProgramDefinition. invocation_values and
# weight_bindings come from the prepared production input/artifact context.
fixture = definition.fixture(invocation_values, bindings=weight_bindings,
                             capture=reference_input)
configuration = Configuration(
    label="Qwen · decode · retained-prefix fixture",
    fixture=fixture,
    device=lambda: ops.DeviceRuntime.open(device_plan),
    options=definition.options,
    store=Path("runs/formula-observations.sqlite"),
)
with configuration.open() as lab:
    formulas = DecoderFormulas.from_trace(lab.formulas)
    ticket = lab.measure(formulas.blocks[0].feed_forward)
    result = ticket.result.result()
    lab.acknowledge(result, client="development-api")
    lab.show()

# Or select between prepared production contexts in the same TUI.
show((configuration,))
```

The capture callback supplies typed root input values, including independently
decoded artifact values when needed. Intermediate fixtures derive from the
existing formula references. It does not define another model or benchmark.
Initial fixture/reference preparation is separate from warm operation timing.
`lab.subtree(handle)` previews a typed parent-first scope; `lab.measure_subtree(handle)`
measures each boundary independently through its production implementation. Fused
parents retain their fusion; isolated child times are not portions of parent time
and are never summed to fabricate parent performance.

Protocol 2 requests bounded native compute-pass timestamps where supported. The
same observation/history/TUI shows `kernel-device-time`, `kernel-count` and
`kernel-rate:*` alongside complete-operation wall time and its rates. Native
kernel time excludes I/O and submission gaps; it is not whole-operation latency.
Unsupported timing is explicit. `MeasurementProtocol(kernel_limit=None)` disables
instrumentation; changing the protocol creates a distinct comparison series.

TUI controls: **m** measures the selection, **s** previews its subtree, **a** previews the affected scope,
**Enter** confirms that scope, **c** cancels/drains, **r** reloads comparable
history, and **o** returns to the configuration chooser when available. **p**
loads matching device characterization (measuring only if absent); **P** explicitly
refreshes it. First measurement obtains missing resource characterization through
that same path. It includes sustained conditioning and is cached across operation
edits. These are empirical measurements, not physical-peak specifications.
The tree shows isolated time, modeled reference time, ratio, gap and limiting
resource. Details retain the useful units, rate provenance and assumptions; results
above the reference flag inadequate calibration/model applicability rather than
claiming super-optimal performance.
Displaying or inspecting history never launches a measurement. The <5-second
changed-operation-to-visible-result target includes refresh, compilation,
checking and publication. The latest qualified pointwise helper edit took **1.6575s**,
including **1.3548s of changed native compilation**.
This is not a claim about every operation or cold model startup.

For isolated production-shaped attention, open the actual attention formula using
an existing artifact's geometry:

```sh
uv run --no-sync python examples/qwen_attention_lab.py /absolute/path/to/model
```

This offers decode and 2048-row prefill at 16K/64K history. Inputs are explicitly
synthetic, not captured full-model state. Each selected configuration retains at
most one preparation and 1 GiB of references; cold reference construction is
reported separately. Read-only KV inputs remain resident between samples, while
written state is reset from the immutable fixture. No alternate benchmark runner
or implementation is involved.

For actual GGUF weights and the production FFN composition, with explicitly
synthetic hidden inputs, open an isolated layer in the same TUI:

```sh
uv run --no-sync python examples/qwen_artifact_lab.py /absolute/path/to/model.gguf --layer 0 --rows 2048
```

This extracts the typed formula from the model trace without compiling or loading
the rest of the model. The artifact stays open until the measurement worker closes.

Browse recorded model/formula configurations without loading a model or opening a
device with `python -m ops.lab /absolute/path/to/observations.sqlite`, or call
`ops.lab.browse(Path(...))`. The stored hierarchy links each occurrence to its
measured input conditions. Opening history never derives fixture/reference inputs.
A new live configuration establishes those conditions on its first measurement;
it does not silently present another configuration's evidence as current.

`ops.analyze` remains a structural diagnostic: actual defined operations,
readiness/effects, aliases, storage lifetimes and submission composition.
`magnitude-qualify` applies that path to production model metadata at the
declared gate. Kernel names, counts and geometry are provenance, not evidence of
optimality or acceptance thresholds. There is no candidate ranking or latency
predictor. The old `magnitude-kernel-bench` and kernel-family attribution runner
were removed; historical output files remain intact.

## Physical kernel architecture

Weights keep one compact resident representation from import through execution:

- MLX Q4 group-64 uses packed nibbles with BF16 scale and bias planes.
- GGUF Q8_0 uses packed signed-byte groups with one FP16 scale per 32 values.
- GGUF Q4_K, Q5_K, and Q6_K use interleaved 256-value superblocks. Local
  coefficients and FP16 super-coefficients stay beside their payload. Q4/Q5
  use six-bit scale/min pairs; Q6 uses signed eight-bit scales. These are
  never expanded to FP32 planes.

Decode work is packet-native: a 32-lane subgroup consumes a 256- or 512-value
reduction tile, loads each coefficient once, produces paired output rows, and
reduces in FP32. Prefill uses a distinct cooperative packet width: MLX Q4 keeps
its 16-code-per-lane GEMV packet but distributes 8-code words across a matrix
workgroup, matching GGUF's matrix packet and restoring full load parallelism.
Prefill projections unpack exact integer codes into shared tiles for `T.gemm`,
then apply each group's scale and bias in FP32. Activations are never paired with
rounded decoded weight tiles; no full dequantized matrix exists. Shared-memory
accounting includes coefficients, and publication rounding is unchanged.
Full tiles have a structurally unpredicated path, while only the boundary tile
pays validity checks.

One decoder specialization is assembled as one maximal multi-function TileLang
program and submitted through one pre-bound native entrypoint. Repeated layer
schedules are private functions lowered once and called by the public entry
through TileLang's existing module lowering. Immutable weights and compiler-owned
temporary storage are bound once.
Python supplies only invocation data and mutable model state; it does not loop
over layers or launch numerical kernels itself.

The enclosing block can publish its FP32 residual directly from dense down
projection, routed decode down projection or grouped prefill's final combine.
The child FFN result is still rounded to its declared activation dtype first.
Isolating or explicitly capturing the child retains that child's output boundary;
formula composition and comparable measurement identities remain unchanged.

## Measure

`session-bench` measures the actual Ops-backed service with simulated
agent sessions for either GGUF or MLX artifacts:

```sh
uv run session-bench run \
  --target magnitude=/path/to/model.gguf-or-mlx-directory \
  --suite context --context 16384,65536 --prose --repeat 1
```

`python -m performance import RUN_DIRECTORY` imports a completed run into the
content-addressed assessment store. Raw runs go under `runs/`, written-up
baselines under `results/`.
