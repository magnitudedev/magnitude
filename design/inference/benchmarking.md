---
applies_to:
  - inference-v2/benchmarks/**
  - inference-v2/tests/benchmarks/**
  - inference-v2/src/session_bench/**
  - inference-v2/tests/session_bench/**
---

# Inference benchmarking

Measurements follow component ownership. A faster end-to-end answer does not by
itself establish faster model execution, and a faster kernel does not establish
better serving performance.

| Boundary | Measures | Appropriate control |
|---|---|---|
| Operator / storage | Completed tensor work, state movement and allocation | Same inputs and state with an independent implementation |
| Model executor | Prefill or decode from a fixed state | Upstream program at matching geometry and output demands |
| Generation | Plain or speculative advancement, acceptance and repair | Independent requests and the same target algorithm |
| Engine | Scheduling, admission, batching, retention and delivery | Matched offered traffic through our engine and an upstream generator |
| Serving | Rendering, HTTP, semantic output and sessions | Actual stock servers through the shared Python session client |

Component and engine experiments are typed Python values. Their subject blueprint
constructs the actual component under test; workload inputs and dependencies are
explicit. A shared runner owns isolated child lifetime, warmup, repetitions,
completion, validation and immutable result records. There is no separate TOML
experiment language. [Session bench](session-bench.md) supplies canonical
BFCL-derived serving traffic, adapters and reports within the same Python package.

## Valid comparisons

Record artifact and runtime identity, composition, input geometry, output work,
memory policy and timing boundary. Complete asynchronous work inside timing;
keep loading, reset and correctness checks outside unless the claim includes them.
Measure shared device service once. Per-request participation and public latency
are different quantities and must be labelled as such.

Execution parity requires matched rendered input and generated work. Fixed-output
controls record actual tokens and compare outputs, not just configured allowances.
Tool workloads may produce different prose, calls or reasoning despite identical
logical requests. Preserve those differences as serving evidence; do not interpret
their completion ratio as a kernel or scheduler ratio. Emission-based upstream
timings are not interchangeable with native model-service counters.

Batch size, query width and quantization can change numerical results. Retain
rejected equality checks and diagnose them against independent computation at
matching geometry. Do not relax a correctness gate to make a timing admissible.
Unsupported batching must be visible rather than silently measured as batching.

## Evidence and progress

Start with resident plain execution, separating single-session and concurrent
traffic. Qualify speculation and custom components as explicit substitutions.
Cover prompt and decode service, mixed arrivals, context lengths, membership
changes and prefix reuse. A throughput improvement that weakens an interruption
target is a policy trade-off and must be compared and reported as such.

Keep failed controls and raw observations alongside successful results. Source
snapshots and run records are historical evidence, not maintained source. Promote
useful diagnostic workloads into typed subjects and retain meaningful regression
tests; archive superseded one-off scripts outside the working source tree.
Performance ceilings and cross-engine parity remain measured claims, never
guarantees inferred from unit tests or a small number of favorable cases.
