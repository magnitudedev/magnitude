---
applies_to:
  - packages/inference-benchmark/**
  - packages/inference-benchmark-dashboard/**
---

# Inference benchmark

The benchmark compares instrumented Chat Completions engines using reproducible agent workloads.
It owns experiment definitions, model-artifact preparation, the shared trial plan, target execution,
measurement, semantic validation, immutable run evidence, and reporting. The CLI and local dashboard
are two interfaces over those same responsibilities.

## Comparison basis

A comparison uses one content-addressed trial plan for every target. The plan fixes messages, tools,
expected actions, request limits, release schedule, dependencies, checkpoints, repetitions, context
tokens per sequence, and parallel sequence capacity. Target adapters may translate the serving
policy into native configuration but may not change the workload.

Every evaluation must reference the same plan digest. A target cannot run the plan if its declared or
allocated parallel capacity differs. Invalid output and failures remain in the results but do not
contribute to performance summaries.

A **strict comparison** uses identical weights, quantization, tokenizer, chat template, speculation
policy, and device placement. Otherwise the result is a labeled **product comparison**. Differing
prompt-token counts for corresponding requests necessarily make a comparison product-level.
This observed classification is separate from the experiment's comparison protocol. The normal
`defineExperiment` constructor holds speculative policy constant. An experiment that intentionally
compares baseline, embedded MTP or DSpark, and DFlash uses the explicit
`defineSpeculativeDecodingComparison` constructor. That protocol fixes one oMLX target artifact and
every non-speculative setting. Its result is labeled a **controlled speculative-decoding**
comparison, never a strict comparison or a generic product comparison. Reports carry the protocol
and each configured method.

## Required endpoint evidence

Measured targets use streaming `POST /v1/chat/completions` with one choice, standard SSE deltas,
`stream_options.include_usage: true`, and a terminating `[DONE]`. This is an instrumented superset
of Chat Completions, not a separate benchmark endpoint.

The terminal event supplies:

- `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens`,
  `usage.completion_tokens`, and `usage.total_tokens`;
- `timings.cache_n` and `timings.prompt_n` for cached and evaluated input tokens;
- `timings.prompt_ms` for native prompt evaluation time; and
- `timings.predicted_n` and `timings.predicted_ms` for generated tokens and native generation time.

Counts are non-negative integers and durations are finite, non-negative milliseconds. Evidence is
valid only when:

```text
usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
usage.prompt_tokens = timings.cache_n + timings.prompt_n
usage.prompt_tokens_details.cached_tokens = timings.cache_n
usage.completion_tokens = timings.predicted_n
```

Missing or inconsistent terminal evidence is a protocol failure. The benchmark never estimates,
repairs, or independently renders token counts. Context rejection is recorded as a target outcome.
Prompt-affecting controls shared by the targets are fixed, including disabling model-selected
thinking through `chat_template_kwargs.enable_thinking: false`.

## Criteria and workloads

Criteria:

- **Responsiveness:** monotonic client time from submission to headers and first semantic output.
- **Prefill:** uncached prompt evaluation rate,
  `1000 * timings.prompt_n / timings.prompt_ms`.
- **Decode:** completion generation rate,
  `1000 * timings.predicted_n / timings.predicted_ms`.
- **Memory:** baseline, peak, and retained host/device footprint attributable to the managed target.
- **Distribution:** completion latency, achieved throughput, fairness, tails, outcomes, and failures.

Native timings exclude queueing, chat rendering, transport, and client parsing. The benchmark does
not derive those durations by subtracting native time from client latency. A zero-token phase has no
rate.

Workloads:

- **Single request:** one cache-disjoint tool decision.
- **Sequential session:** an ordered agent history that grows each turn.
- **Independent concurrency:** unrelated requests released together.
- **Forked concurrency:** concurrent branches sharing an established prefix.
- **Concurrency pressure:** increasing offered concurrency.
- **Memory pressure:** multiple resident histories at increasing depths.
- **Context scaling:** cache-disjoint completed BFCL histories selected near explicit approximate
  token checkpoints, measuring full-prompt behavior as context grows.

### Criteria × workload matrix

| Workload | Responsiveness | Prefill | Decode | Memory | Distribution |
| --- | --- | --- | --- | --- | --- |
| **Single request** | Baseline TTFT | Isolated input throughput | Isolated output throughput | Loaded baseline and request peak | Run-to-run variance and failures |
| **Sequential session** | TTFT by history depth | Throughput as context grows | Decode rate by depth | Resident-prefix growth | Tail latency and failures across turns |
| **Independent concurrency** | TTFT under parallel load | Native prefill-rate distribution under load | Native decode-rate distribution under load | Batch and workspace peak | Achieved throughput, fairness, tails, and failures |
| **Forked concurrency** | Branch TTFT after setup | Verified cache reuse and uncached prefill rate | Native branch decode rates | Shared-prefix and branch footprint | Achieved throughput, fairness, tails, and failures |
| **Concurrency pressure** | TTFT growth by offered load | Native prefill degradation | Native decode degradation | Footprint by concurrency | Saturation, rejections, errors, and tail growth |
| **Memory pressure** | TTFT with resident histories | Input throughput under memory load | Decode under memory load | Peak and retained scaling | Degradation, OOM, and failure onset |
| **Context scaling** | TTFT by actual prompt size | Full-prompt throughput by actual size | Decode rate by actual size | Peak footprint by actual size | Long-context degradation and rejection |

Profiles change depths, concurrency points, and repetitions, not workload meaning. Every trial
records its workload, checkpoint, repetition, and cache state.

## Inputs and identity

- **Model:** Logical weights and architecture shared by comparable artifacts, identified by a stable
  logical model ID and declared context limit. Trial planning needs no engine artifact.
- **Model artifact:** An immutable engine-loadable representation. Each artifact pins repository,
  commit, quantization, files, sizes, and digests. Multi-file snapshots use a complete committed
  lock. Artifact differences make a result product-level rather than invalid.
- **Engine:** A serving implementation with engine-specific settings and an owned launch adapter.
- **Acceleration policy:** A discriminated decoding mode. oMLX supports ordinary decoding, embedded
  MTP, embedded DSpark when the checkpoint is natively identified as such, and DFlash with a
  separately locked MLX draft snapshot. Embedded modes do not declare a separate drafter artifact.
- **Variant:** One model artifact, one engine, and the complete serving settings evaluated together.
- **Experiment:** A TypeScript-authored, serializable definition of variants, workload profile,
  request policy, and execution order. TypeScript is the only authoring format.
- **Request timeout:** An experiment-level request-policy limit. It defaults to five minutes and can
  be raised explicitly for long-context trials without changing engine or workload semantics.
- **Context target:** A reproducible planning target expressed as approximate tokens and a declared
  characters-per-token conversion. The planner selects one shared semantic BFCL prefix nearest the
  resulting canonical character count; it never creates engine-specific content.
- **Corpus:** BFCL V4 data is commit-pinned, externally cached, and SHA-256 verified. The repository
  stores only lock and selection metadata. Deterministic tool-decision units retain provenance and
  compose into histories with synthetic tool results.

Preparation is distinct from measurement. It resolves immutable revisions, downloads and verifies
artifacts, freezes engine environments, qualifies tokenizer metadata, and records executable
versions. A measured run never installs dependencies, downloads files, or updates a revision.
The managed oMLX environment pins oMLX to one immutable Git revision and freezes MLX, MLX-LM,
MLX-VLM, and DFlash in `uv.lock`. Prepared evidence records those identities, the lock digest, and
the owned adapter-source digest; a run re-verifies both digests and the installed identity before
launch. oMLX qualification loads the locked snapshot and verifies its tokenizer, chat template,
tool rendering and parsing, context capacity, selected decoding backend, terminal native timings,
and request-local speculative counters. Installation and qualification belong to preparation only.
`bun benchmark models lock-mlx` is the canonical way to create a complete MLX artifact lock; it
uses the standard Hugging Face cache and creates no model database.
Reusable TypeScript model definitions own logical model identity and artifact pins; experiments
import them rather than duplicating artifact metadata. Hub artifacts remain in the standard Hugging
Face cache and are never copied into run storage. The corpus remains independently fetchable and
inspectable through the CLI, while experiment preparation invokes the same corpus service.

## Execution and outcomes

The model remains loaded for all trials against a target. Managed targets run sequentially so two
loaded models do not contend for one device. Cache-disjoint identities and resident histories create
the required state without cache-clearing endpoints or per-trial restarts.

Target adapters may bind an exact model instance and translate serving capacity into native launch
configuration, but cannot alter shared request content. Actual managed capacity is checked before
measurement.

Each request records semantic validity, terminal evidence, stream observations, and one outcome:
valid, invalid semantics, rejected, timed out, cancelled, protocol error, transport error, or target
failure. Semantic validity requires the expected tool calls and allowed arguments; protocol validity
requires consistent terminal evidence. Standard workload summaries require both gates. Context
scaling keeps semantic validity as a separate result and summarizes any valid-or-invalid semantic
outcome with complete protocol evidence, because the workload measures engine behavior at the
selected prompt size rather than task accuracy. Transport, protocol, rejection, timeout, and
cancellation outcomes never contribute performance metrics.

Managed memory sampling covers the complete target process tree. Shutdown sends SIGINT first, waits
for every captured process to exit, and kills only the owned tree if graceful termination fails.
Observations record their source, scope, and limitations; measurements from unlike source classes
are not compared as byte ratios.

oMLX is the owned unified MLX serving path. Every managed process receives one run-scoped private
base directory, deterministic global and per-model settings, and one symlink to the prepared target
snapshot. Hugging Face discovery, fallback models, persistent caches, memory guarding, burst decode,
SpecPrefill, TurboQuant, ANE prefill, and unrequested speculative paths are explicitly disabled.
The adapter imports pinned oMLX as a library; it never starts stock MLX-LM or MLX-VLM servers and
never reads or writes `~/.omlx`. Readiness is stronger than health: the expected alias must be the
only discovered model, its engine must be loaded, configured context and concurrency must match,
the exact requested backend must be active, and an internal tool-calling qualification request must
complete with canonical evidence.

The owned adapter instruments the narrow native execution seams. It measures scheduler prompt
evaluation of the uncached suffix, including the first batch step where oMLX evaluates the final
prompt token, then measures scheduler generation work from that boundary. It converts native
seconds to milliseconds and combines those durations with oMLX's terminal token counts. It
deliberately does not use HTTP latency as prompt evaluation time. The canonical terminal SSE event
has empty choices and is followed by `[DONE]`.

Speculative evidence is method-neutral at the benchmark boundary and method-specific at the oMLX
adapter. Embedded MTP and DSpark use request-local native MTP statistics; DSpark readiness also
validates the loaded checkpoint's native DSpark heads. DFlash uses its structured cycle and summary
events and a separately locked MLX draft snapshot. Every speculative request reports the actual
backend plus drafted and accepted counts, accepted cannot exceed drafted, and every measured
speculative run must demonstrate nonzero drafting. Logs and server-global counters are not evidence.

Context sweeps are distinct from server context capacity. The experiment configures one capacity
large enough for its maximum checkpoint, then constructs separate cache-disjoint requests by
stacking canonical completed BFCL interactions. Planning counts the stable serialization of shared
messages and tools and requires no tokenizer. Approximate targets choose content only; terminal
engine evidence remains authoritative for actual prompt tokens and all rates.

Legacy MLX-LM and MLX-VLM adapters remain engine types for historical product comparisons, but they
must not be mixed into the standardized oMLX result set. A speculative-decoding comparison uses
fresh oMLX processes for the same locked target artifact in baseline and selected speculative modes.

## Runs and interfaces

A run is one immutable resolved execution of an experiment. Default balanced blocks reverse variant
order on alternating blocks. Speculative-decoding comparisons rotate every variant through each
block position and therefore require a block count divisible by the number of variants. Only one
model is loaded at a time. Every oMLX variant and block gets a fresh process and private base
directory so process-global patches and caches cannot cross variants. Before launch, the run writes a
manifest containing the resolved experiment, plan, artifacts, engine environments, host identity,
and execution order. It appends lifecycle events while active and writes raw results and derived
Markdown only after evaluation. A workspace lock permits one managed run at a time.

The agent-facing CLI owns explicit corpus fetch/status, experiment preparation, planning, execution,
inspection, watching, and cancellation. Existing endpoints, ICN, llama.cpp, legacy MLX adapters,
and managed oMLX are variant engine types over the same evaluator rather than separate run modes.
JSON is not an experiment
authoring format and there is no second model-profile registry.
The local dashboard discovers the same TypeScript experiments and reads the same filesystem run
evidence. Dashboard actions spawn the non-interactive CLI; browser lifetime does not own a run. The
dashboard has no database and cannot edit experiment source. Starting it prints its UI and API URLs
and streams child-process diagnostics to the invoking terminal.
