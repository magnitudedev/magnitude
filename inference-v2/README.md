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
JSON constraints, sampling, and stop strings. See the [serving contract](../design/inference/serving.md).

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
scheduler arguments are shared with the server. See [metric definitions](../design/inference/chat.md).
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
benchmarks/        Typed component, model, generation and engine experiments
tests/             Tests grouped by the same responsibilities
```

Blueprints live beside their implementations. Architecture-specific loading and computation
live under `models/architectures/`; shared operators and state storage have their own domains.
The engine coordinates requests, generation coordinates model progress, and executors own
computation and state. Models receive resource dependencies, not the scheduler or prefix index.

Design: [composition](../design/inference/composition.md),
[scheduler](../design/inference/engine/scheduler.md),
[state and speculative generation](../design/inference/engine/speculative-generation.md).

## Benchmarks

Select a typed Python experiment as `module:variable`. Preview its composition or run it in
an isolated child process:

```sh
uv run --frozen python -m benchmarks benchmarks.cases.attention:metal --describe
uv run --frozen python -m benchmarks benchmarks.cases.state:branch \
  --output runs/state-branch.json
```

Use a new output path for each run. Results preserve the composition, source identity, raw
samples, validation outcomes and counters. Setup and validation are outside timing; device
completion is inside it.

| Cases under `benchmarks.cases` | Measurement |
|---|---|
| `attention`, `recurrence`, `state` | Operators and KV storage |
| `single_session`, `upstream`, `parity` | Prefill/decode and matched execution controls |
| `prefill_batch`, `continuous_batching` | Shared model work and concurrent engine service |
| `generation` | Plain/speculative generation and cold/warm engine workloads |
| `baseline_models` | Model and quantization substitutions |

For BFCL-derived workloads through actual servers, use [session-bench](session-bench.md).
It documents model aliases, reference runtimes, session shapes and result inspection.
The [benchmarking design](../design/inference/benchmarking.md) defines measurement boundaries
and comparison requirements.

## Checks

```sh
uv run --frozen pytest
uv run --frozen ruff check src tests benchmarks
uv sync --frozen --project session-bench-runtimes/omlx
uv run --frozen pyright
```

Full type checking includes the oMLX benchmark adapter against its separate reference environment.
Local-model tests are opt-in; each skipped test names the artifact environment variables it needs.
