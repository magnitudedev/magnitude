# Performance

**Performance evidence qualifies the exact Magnitensor graph, selected
lowerings and compiled execution used by production. Prefill, time to first
token and decode are independent acceptance dimensions.**

## Evidence chain

```text
model function + workload
        │ trace and specialize
        ▼
semantic graph + static facts
        │ select regions, representations, schedules and storage
        ▼
compiled callable ──invoke repeatedly──► completion
        │                                  │
        └── provenance                     └── independent validation + metrics
```

A performance record identifies the model function, artifact, workload,
semantic graph fingerprint, selected region and operation lowerings, weight
representations, materialization plan, compilation units, machine, TileLang
build and compiler provenance. This makes a graph-selection change
distinguishable from an improvement to the same portable kernel or to TileLang
lowering.

Compilation, weight import and tuning are measured separately from steady-state
invocation. They remain product costs and receive their own evidence; they are
never silently included in or excluded from request latency.

## Acceptance

| Dimension | What the measurement includes |
|---|---|
| Prefill | The complete model computation for an admitted prompt chunk, including state publication required before decode |
| Time to first token | Client-observed request admission through availability of the first generated token |
| Decode | Repeated accepted model advances from an identical retained history |
| Long context | The same phase measurements with state traversal and capacity behavior exercised at representative long histories |
| Tail behavior | Non-ideal token, hidden, expert and vocabulary shapes required by supported models |

The primary supported workloads require parity with or improvement over the V2
engine in every applicable dimension. A large win in one phase does not excuse
a regression in another. An implementation is not accepted because its
abstractions are clean, because an isolated kernel is fast, or because it
improves over a slower V3 path; the whole selected production path must meet the
phase target while preserving the numerical and architectural contracts.

## Diagnosis follows ownership

```text
excess materialization or dispatches ──► Magnitensor graph and region selection
wrong representation or schedule       ──► Magnitensor lowering and tuning
portable program cannot express work   ──► TileLang language or capability gap
portable program lowers poorly         ──► TileLang compiler, runtime or target adapter
request admission or batching delay    ──► Magnitude service and generation policy
```

Evidence must locate the limiting layer before a change is proposed. Performance
never justifies a backend-specific side channel, direct TileLang use from the
engine, model identity in generic compiler policy, or benchmark-only execution.

## Measurement rules

| Rule | Reason |
|---|---|
| Production and measurement build the same model function and compiled callable | A benchmark-owned composition measures another system |
| One machine-wide lock protects device timing | A run never shares the device with another timing or test |
| Validation precedes assessment | A fast wrong result is not evidence |
| Workload and source remain fixed during comparison | Selection and latency are attributable |
| Observation, model and comparison remain separate fields | A measured latency, an analytical bound and a target are not interchangeable |
| Warm and cold behavior are named explicitly | Compilation, caching and execution cannot be conflated |
| Warm host work is reported against dynamic bindings and submission units | Per-layer, per-weight or per-kernel Python work is an architectural failure even before latency is aggregated |

## Whole-model measurement

A decode measurement performs no prefix construction inside the measured
interval and shares no mutable tail with another sample. The case retains one
accepted history and aborts each measured advance after completion, so every
sample begins from the same state. The reference is independent logits from the
same artifact.

Prefill measurements cover the complete selected graph rather than a collection
of favorable kernels. Time to first token is observed across the serving
boundary so admission, packing, state publication and sampling costs remain
visible.

## External evidence

Session bench measures serving with simulated agent sessions over disposable
server processes. It is opaque to the engine and imported as external evidence;
native service diagnostics are never substituted for client-observed latency.
Its inputs are the [benchmark fixtures](benchmark-fixtures.md).

## Where numbers live

Raw runs live under `runs/`; written-up baselines live under `results/`. Design
documents carry no benchmark numbers. A claim points at a record, and a record
that cannot identify what was selected, on what machine and from which source is
not evidence.
