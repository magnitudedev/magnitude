# Session bench

`session-bench` measures inference serving with simulated agent sessions: tool decisions,
long histories, sequential turns, parallel sessions and branches. It uses pinned BFCL V4
cases or Moby Dick passages to build deterministic inputs. It does not run an agent or produce
an official BFCL score.

## Run it

Run commands from `inference-v2/`. `uv` installs the Python runner automatically.

Create `models.local.json` in this directory with your own model locations:

```json
{
  "qwen-q4": {
    "mlx": "/path/to/mlx-model-directory",
    "gguf": "/path/to/model.gguf"
  }
}
```

The file is gitignored. Relative paths resolve from this directory. An entry needs only the
representations you use: MLX for `magnitude`, `mlx-vlm`, and `omlx`; GGUF for upstream `llama.cpp`.
Aliases are personal conveniences, not checked-in model definitions or proof that two artifacts
are equivalent.

Pinned Hub references also work: `hf:owner/repository@<40-character-commit>` for an MLX snapshot,
or `hf:owner/repository@<40-character-commit>#filename.gguf` for a GGUF file. Preparation downloads
missing artifacts through the standard Hub cache and records their content hashes.

```sh
# Magnitude is the default engine.
uv run session-bench run --model qwen-q4 --context 4k,16k,64k

# Compare serving engines on the same model artifact.
uv run session-bench run --model qwen-q4 \
  --engine magnitude --engine mlx-vlm \
  --suite context --context 4k,16k

# Select simulated session shapes.
uv run session-bench run --model qwen-q4 \
  --suite session,parallel,fork --context 4k,16k

# Use prose with the same serving schedules.
uv run session-bench run --model qwen-q4 --prose \
  --suite context,session --context 4k,16k,64k

# Compare against upstream llama.cpp using the alias's GGUF representation.
uv run session-bench run --model qwen-q4 \
  --engine magnitude --engine llama.cpp --context 4k

# Inspect the selection without installing engines or loading/downloading weights.
uv run session-bench run --model qwen-q4 --suite all --context 4k --dry-run

uv run session-bench models
uv run session-bench engines
uv run session-bench suites
uv run session-bench runs
uv run session-bench show <run-id>
```

Repeat `--model` and `--engine` to select their combinations. Use repeated
`--target ENGINE=ARTIFACT` instead for explicit pairs. `--category` accepts comma-separated
`simple-python`, `parallel`, `parallel-multiple`, or `all`; `--case` selects a specific current decision.
Canonical background history still comes from the selected categories. `--dry-run` inspects
the selection; token-bound histories are prepared when running with the first target. `--repeat` repeats the
balanced schedule. Add `--json` for machine-readable discovery and results; progress goes to stderr.

## Prose mode

`--prose` selects the shared, downloaded `prose.moby-dick` fixture. Each request asks
the model to continue a passage, returning only prose. Requests omit tools and tool
choice. `--category` and `--case` are tool-only filters and cannot accompany `--prose`.

All sections remain available. `single` uses a short passage; `context`, `parallel`
and `concurrency` use independent reading sessions sized at the requested checkpoints.
`session` and `memory` preserve previous messages and advance through the book.
`fork` prepares a parent, then branches from its shared canonical history. Every
independent session starts at the beginning of the normalized book, with a distinct
session label. Input construction is defined in [benchmark fixtures](design/benchmark-fixtures.md#session-bench-prose).

Prose generation is autoregressive and allows **256 output tokens**, ending at EOS
or that budget. Either is a valid performance endpoint; reports record actual output
lengths. Empty text, tool calls, malformed streams, and context exhaustion before the
budget are failures. There is no comparison against the book's wording or answer-quality
score. Later inputs use canonical book text, regardless of what the model generated.

## Maintained sections and policy

| Section | Traffic |
| --- | --- |
| `single` | One short content request without added history; ignores context checkpoints |
| `context` | Independent full-prefill requests near each checkpoint |
| `session` | One sequential session growing through the checkpoints |
| `parallel` | Four independent sessions released together at each checkpoint |
| `fork` | A parent request followed by four branches sharing its canonical history |
| `concurrency` | Independent requests at concurrency 1, 2, 4 and 8 |
| `memory` | Four growing sessions with process-tree memory sampling |

Defaults are `context` at `1k,4k,16k`. `k` means 1,024. Checkpoints are approximate **input**
sizes: shared fixture preparation reaches each target at a complete interaction boundary,
using the first selected engine’s tokenizer. Reports show each engine’s actual native count.
All targets receive the same logical messages and tools. Generated answers never change later
inputs. The decision corpus contains 797 interleaved cases from the pinned three-category subset.
These sections are serving traffic recipes, not a full evaluation of all 797 decisions.

Tool requests allow **32,768 output tokens**. There is no setting to lower this ceiling. Normal
termination ends generation early. Truncation always fails, even when the partial tool call parses.
Preparation counts rendered inputs with each engine's tokenizer and reserves the full output
allowance for the selected mode, with shared capacity rounded up to 256-token allocation boundaries. Insufficient model
context fails explicitly.

Sampling is greedy, with seed 42 where supported and thinking disabled. The initial comparison
baseline disables retained prefixes for every section. Shared history describes the input traffic;
it does not imply cache reuse. Memory reports show process-tree RSS, including loading, rather than
isolated GPU allocation or retained-cache memory.

Each target runs in a fresh process on each pass, with a separate qualification request. There are
at least two passes; multiple targets rotate their starting order. Target engines run sequentially,
while each section controls concurrency within a target. One machine lock prevents overlapping
session-bench runs across checkouts. Preparation and warmup are excluded from measurements.

## Engines

- `magnitude` launches this package's Python engine in the same checkout and environment.
- `mlx-vlm` launches the stock server in its locked environment under `session-bench-runtimes/mlx-vlm`.
- `omlx` uses the pinned upstream source in `session-bench-runtimes/omlx`, with the benchmark's
  native timing instrumentation and readiness checks.
- `llama.cpp` launches upstream `llama-server` from `PATH` and records its version and binary hash.
  It must support the current serving, template, tokenization and timing interfaces.

The reference runtime directories contain dependency manifests and locks; all benchmark Python
source, including adapters, is in `src/session_bench`. MLX engines require a compatible Apple Silicon
host. The old Magnitude llama.cpp/ICN engine is not supported.

Engine environments are prepared before measurement and run offline. Artifact, source and installed
package identity are rechecked before launches. Readiness must confirm the required policy; unknown
or inconsistent capabilities fail instead of being silently substituted.

## Results and reproduction

Every run creates `runs/session-bench/<UTC-id>/` automatically, including failed preparation:

| File | Contents |
| --- | --- |
| `command.txt` | Copyable command and working directory, expanded to explicit artifact references |
| `run.json` | Original argv/cwd, selection, host and owner identity |
| `plan.json`, `requests.jsonl` | Shared schedule, corpus digest, fixture provenance, input bodies and expectations |
| `results.jsonl` | Incrementally saved observations, outcomes, timings and correctness |
| `summary.json`, `report.md` | Workload mode, actual prompt/output lengths, aggregates, denominators and failures |
| `events.jsonl`, `memory.jsonl`, `footprints.jsonl` | Lifecycle and process-tree memory evidence |
| `logs/` | Engine output and raw request streams |
| `source/`, artifact/runtime records | Relevant source snapshots, locks, versions and artifact hashes |

The whole results directory is gitignored. Completed observations survive cancellation and crashes.
SIGINT and SIGTERM retire owned engines; an engine supervisor also watches for loss of the runner.
Inspection labels unfinished runs as running or interrupted. Current readers do not migrate old
formats; the saved Markdown and raw evidence remain readable.

Rerun the command in `command.txt`; it does not depend on local aliases or read a saved configuration.
Local paths must still contain the recorded model bytes. Exact historical reproduction also needs
the recorded code, dependency versions and hardware. A command on newer code runs its current
behavior; results do not promise a permanently stable schema or CLI.

Reports separate client TTFT/completion latency from native phase timings. Stock MLX-VLM's timing
basis is server token emission and must not be treated as an equivalent native service measurement.
The JSON summary includes median and nearest-rank p95 client latency, with eligible counts.
Premature truncation and execution/protocol failures are excluded from performance statistics.
For tools, only `context` may include semantically invalid but protocol-complete responses;
other sections require correctness. Prose uses the text/protocol and output-budget rules above.
Exit codes are 0 for complete success, 1 for a benchmark failure, 2 for invalid input, and 130 for
cancellation.

Completed request observations also enter `runs/performance/` automatically, with their
originating hardware and HTTP boundary. `python -m performance import RUN_DIRECTORY`
imports an existing session run idempotently. These observations do not score internal
model components or replace session-bench's detailed reports.

## Development

From `inference-v2/`:

```sh
uv run pytest
uv run ruff check src/session_bench tests/session_bench
uv sync --locked --project session-bench-runtimes/omlx
uv run pyright
```

The focused tests run without loading models. They exercise deterministic histories, artifact and
command identity, streaming validation, subprocess lifetime, cancellation and incremental results.
Type checking resolves the oMLX instrumentation against that adapter's separately locked
reference environment. It does not add oMLX to the engine's own dependencies.
Shared content construction and execution-mode definitions are in
[benchmark fixtures](design/benchmark-fixtures.md).
