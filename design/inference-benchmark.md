---
applies_to:
  - packages/inference-benchmark/**
---

# Inference benchmark

The benchmark compares instrumented, llama.cpp-compatible Chat Completions services using
reproducible agent workloads. It owns the shared trial plan, reproducible inputs, target execution,
measurement, semantic validation, and reporting.

## Comparison basis

A comparison uses one content-addressed trial plan for every target. The plan fixes messages, tools,
expected actions, request limits, release schedule, dependencies, checkpoints, repetitions, context
tokens per sequence, and parallel sequence capacity. Target adapters may translate the serving
policy into native configuration but may not change the workload.

Every result must reference the same plan digest. A target cannot run the plan if its declared or
allocated parallel capacity differs. Invalid output and failures remain in the results but do not
contribute to performance summaries.

A **strict comparison** uses identical weights, quantization, tokenizer, chat template, speculation
policy, and device placement. Otherwise the result is a labeled **product comparison**. Differing
prompt-token counts for corresponding requests necessarily make a comparison product-level.

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

### Criteria × workload matrix

| Workload | Responsiveness | Prefill | Decode | Memory | Distribution |
| --- | --- | --- | --- | --- | --- |
| **Single request** | Baseline TTFT | Isolated input throughput | Isolated output throughput | Loaded baseline and request peak | Run-to-run variance and failures |
| **Sequential session** | TTFT by history depth | Throughput as context grows | Decode rate by depth | Resident-prefix growth | Tail latency and failures across turns |
| **Independent concurrency** | TTFT under parallel load | Native prefill-rate distribution under load | Native decode-rate distribution under load | Batch and workspace peak | Achieved throughput, fairness, tails, and failures |
| **Forked concurrency** | Branch TTFT after setup | Verified cache reuse and uncached prefill rate | Native branch decode rates | Shared-prefix and branch footprint | Achieved throughput, fairness, tails, and failures |
| **Concurrency pressure** | TTFT growth by offered load | Native prefill degradation | Native decode degradation | Footprint by concurrency | Saturation, rejections, errors, and tail growth |
| **Memory pressure** | TTFT with resident histories | Input throughput under memory load | Decode under memory load | Peak and retained scaling | Degradation, OOM, and failure onset |

Profiles change depths, concurrency points, and repetitions, not workload meaning. Every trial
records its workload, checkpoint, repetition, and cache state.

## Inputs and identity

- **Model:** A built-in profile pins repository, commit, file, size, and digest. A Hugging Face URL
  resolves to a commit; a local path is an offline override. Every managed target receives the same
  verified artifact. GGUF metadata supplies the declared context limit, not benchmark-side prompt
  rendering.
- **Corpus:** BFCL V4 data is commit-pinned, externally cached, and SHA-256 verified. The repository
  stores only lock and selection metadata. Deterministic tool-decision units retain provenance and
  compose into histories with synthetic tool results.

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
requires consistent terminal evidence. Only requests satisfying both gates contribute to performance
summaries.

Managed memory sampling covers the target process tree. Observations record their source, scope, and
limitations; measurements from unlike source classes are not compared as byte ratios.
