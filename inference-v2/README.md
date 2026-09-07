# Magnitude inference

An Apple Silicon inference engine with composable model execution, continuous batching,
prefix reuse, speculative decoding, and a Chat Completions API.

## Run the server

Requires Apple Silicon, Python 3.12+, and `uv`. Commands below run from `inference-v2/`.

```sh
uv sync --frozen
uv run --frozen python -m magnitude_engine.serving \
  --target /absolute/path/to/model/snapshot \
  --model magnitude-local --port 8080
```

This starts resident, plain generation using the upstream MLX-VLM model implementation.
The HTTP host owns a private worker process that loads and executes the model.
Add `--head /absolute/path/to/mtp/snapshot` to compose a Qwen target with its MTP drafter.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"magnitude-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The server binds to loopback. `/health` reports worker readiness and `/v1/models` lists the
served model. Chat Completions supports streamed and collected text, reasoning, tool calls,
JSON constraints, sampling, and stop strings.

## Interactive chat

Run a text conversation directly against a private engine worker:

```sh
uv run --frozen python -m magnitude_engine.chat \
  --target /absolute/path/to/model/snapshot --max-tokens 1024
```

Responses stream live alongside prefill progress. Each turn reports cached/new prompt tokens,
TTFT, queue time, prefill and decode rates, and draft acceptance when enabled. `/reset` clears
conversation history, `/exit` quits, and Ctrl-C cancels a response. Use `--prompt "Hello"` for
one turn, or `--engine-blueprint engine.json` for an authored composition. Model, memory and
scheduler arguments are shared with the server.
For models with a thinking switch, `--no-thinking` requests direct answers; otherwise the
checkpoint's default applies.

## Compose an engine

Blueprints are typed, immutable dependency declarations. The host serializes the composition;
the worker builds and owns its live components. Shared dependencies retain their identity.

```python
from pathlib import Path
from magnitude_engine import blueprints as bp

artifact = bp.model.artifacts.Local(path="/absolute/path/to/model/snapshot")
engine = bp.engine.Engine(
    generation=bp.generation.Generation(target=bp.model.auto(artifact)),
    scheduler=bp.engine.scheduling.TimeShared(max_active=4, prefill_tokens=512),
    memory=bp.engine.memory.Budgeted(limit_bytes=28 << 30),
    prefixes=bp.engine.prefixes.Radix(
        retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=32),
    ),
)
Path("engine.json").write_text(bp.dumps(engine))
```

```sh
uv run --frozen python -m magnitude_engine.serving \
  --engine-blueprint engine.json --model magnitude-local --port 8080
```

`bp.model.auto` composes upstream execution with compatible native state. Architecture-specific
programs expose attention, recurrence, embedding and expert dependencies for substitution.
Generation methods compose the target and drafter. Engine policies select scheduling, memory
and prefix retention. When supplying a blueprint, declare these choices in it rather than CLI overrides.

## Organization

```text
src/magnitude_engine/
  composition/    Typed blueprints, serialization and scoped construction
  engine/         Scheduling, admission, delivery, memory policy and prefix retention
  generation/     Sampling, constraints, plain/speculative rounds and drafter coordination
  models/         Architecture programs, operators, loading and transactional KV/recurrent state
  artifacts/      Model metadata, tensor formats, quantization and tokenizers
  resources/      Memory accounting, resource lifetime and bounded I/O
  worker/         Process supervision and host/worker transport
  serving/        Chat rendering, parsing and HTTP
  chat/           Interactive text chat and per-turn diagnostics
  blueprints/     Lightweight public composition API
src/session_bench/  Serving benchmark runner and engine adapters
performance/       Captured graphs, executable theory, component benchmarks and TUI
tests/             Tests grouped by the same responsibilities
```

Blueprints live beside their implementations. Architecture-specific loading and computation
live under `models/architectures/`; shared operators and state storage have their own domains.
The engine coordinates requests, generation coordinates model progress, and executors own
computation and state. Models receive resource dependencies, not the scheduler or prefix index.

Design: [components](design/components.md), [model composition](design/models/composability.md),
[engine](design/engine/components.md) and [performance](design/performance.md).

## Benchmarks

Benchmarks are ordinary functions accepting actual components. Results are written automatically
under `runs/performance/`; variants and sweeps use ordinary Python in the corresponding domain
module. There is no benchmark blueprint or case selector.

```sh
uv run --frozen python - <<'PYTHON'
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.attention.metal import MetalPagedAttention
from performance.benchmarks.attention import benchmark

geometry = dict(query_heads=8, kv_heads=2, key_width=128, value_width=128, element_bytes=2)
for implementation in (GatheredAttention(), MetalPagedAttention()):
    result = benchmark(implementation, geometry=geometry, context_tokens=4096)
    print(result.path)
PYTHON

uv run --frozen python -m performance tui
uv run --frozen python -m performance pull m4-pro-01 /Users/ec2-user/magnitude-mlx/inference-v2/runs/performance
uv run --frozen python -m performance rebuild
uv run --frozen python -m performance check
```

`check` reports missing evidence/bindings and inconsistent assessments; zero theoretical floors
remain explicit. To audit a planned graph before collecting samples, call
`performance.assessment.preflight(assembly.graph, workload, profile)`.
Platform capacities must be justified upper bounds; absent bindings do not produce percentages.

For a loaded engine, `inspect_engine(engine).at("target.layers.3.mixer.attention")` binds the
real selected operator and geometry. Pass it to the same `benchmark` function. Model, generation,
state, control and complete-engine measurements live beside their reusable Python cases in
`performance/benchmarks/`. A recording may bind descendant operating points explicitly for
cross-composition evidence reuse; it never attributes a parent timer to its children.

`python -m performance render VIEW_ID --document design/models/architectures/qwen35.md`
replaces that document's Assembly block from the published state at that operating point.
The TUI has one composition selector and shows the latest applicable evidence per component.
Use arrows to navigate and expand the tree; selected-component details show the actual
hardware, workload and evidence. Press `c` to change composition, `f` to focus a subtree,
Escape to return and `q` to quit.
`import DIRECTORY` ingests finalized bundles; `incomplete` and `recover RUN_ID` inspect and
finalize journals left by dead local processes. Completed records cannot be overwritten.

[Session-bench](session-bench.md) shares prose/tool fixtures and automatically contributes its
HTTP observations to this store. Its original request records and reports remain available.

## Checks

```sh
uv run --frozen pytest
uv run --frozen ruff check src tests performance
uv sync --frozen --project session-bench-runtimes/omlx
uv run --frozen pyright
```

Full type checking includes the oMLX benchmark adapter against its separate reference environment.
Local-model tests are opt-in; each skipped test names the artifact environment variables it needs.

## Shared benchmark fixtures

See [fixture methodology](design/benchmark-fixtures.md) for content, execution modes
and provenance. Prepare without loading model weights:

```sh
uv run --frozen python -m benchmark_fixtures prose.moby-dick \
  --artifact /path/to/model --context 65536 --continuation 256
uv run --frozen python -m benchmark_fixtures tools.bfcl \
  --artifact /path/to/model --context 65536 --continuation 16
```

For an already loaded engine, run a persisted context sweep directly:

```python
from performance.benchmarks.model import prose
results = prose(engine, contexts=(4096, 16384, 65536), mode="replay", measured_tokens=32)
```

The output budget and mode are workload parameters. Replay consumes fixed fixture tokens;
generation feeds predictions back. Session-bench's HTTP budgets retain their own serving boundary.
