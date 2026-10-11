---
applies_to:
  - inference/engine/src/composition.rs
  - inference/engine/src/host.rs
  - inference/engine/src/families.rs
  - inference/engine/src/options.rs
  - inference/engine/src/preview.rs
  - inference/engine/src/worker/**
  - inference/engine/families/contracts/**
  - inference/service/server/src/worker_process.rs
  - inference/service/server/src/residency/worker.rs
---

# Inference composition

The engine is a library. Its root crate binds the common engine crates (artifacts, families,
state, batching, kernels and executor, generation, scheduler, templates and chat) into one facade;
Seismic sits below it and is the only device and byte authority. Serving (`engine/serving`) and
the engine CLI depend on the engine; the engine depends on neither, and has no HTTP dependency.

```text
Host process: package interpretation, chat semantics, protocol framing, worker supervision
                       │ immutable, device-free values over the worker protocol
Worker: one loaded model
  Execution owner          device, weights, numerical state and release policy
  Scheduler                admission, batching, time-shared prefill and decode, retention
  Generation               plain or proposal / verify / accept, sampling, constraint masks
    Executor               family program over state storage on one Seismic device
```

## Construction

Construction is one staged progression:

```text
EngineConfiguration -> resolve -> ResolvedEngineConfiguration { host, manifest }
                    -> preview (read-only planning)   or   prepare (tuning only)   or   load -> ReadyEngine
```

- **Resolve** is device-free. It opens the package, recognizes its family, interprets the model
  definition and input adapter, and resolves the model policy and served context. It bounds the
  served image resize to the cells the service limits' load encodes per image (the image cell limit,
  [execution plan](engine/execution-plan.md)), so the host's media processor admits no image the
  worker has no vision graph for. Nothing opens a device or imports a tensor.
- **Host artifacts** own chat semantics: tokenizer, templates and their inspection, input adapter
  and media placeholder policy. They stay on the host.
- **The execution manifest** is owned and serializable; it is the only value a worker needs.
- **Preview** plans against current devices without opening or allocating anything.
- **Load** runs the same worker either in-process over a channel transport (engine CLI, tests) or
  in a worker process over framed standard streams (the service). The worker protocol and the
  worker code are identical in both cases.
- **Prepare** runs the same worker, transport and build handshake through the preparation a load
  begins with (opening the device and preparing programs, tuning whatever the kernel cache lacks),
  reports preparation and tuning progress as a load does, and exits once prepared. Tuning imports
  only the weights of the few layers its cases rotate over; a prepared worker never allocates
  serving state, imports the whole model or serves.
- **Readiness** binds host artifacts to the connected worker only when the worker loaded exactly
  the package the host resolved, with the same template fingerprint and input modalities; any
  mismatch is a typed failure, never a partial engine.

## Families

Model families are registered once. Recognition runs over every registered family and exactly one
must claim a package; no match or several matches is a typed unsupported-family outcome. A family
supplies recognition, model-definition construction, its input adapter and media placeholder
policy, and its program inputs. Everything downstream (planning, scheduling, generation, serving
and assessment) consumes the model definition and the adapter contract only, so adding or
replacing a family (including a compiled-program family) changes nothing outside the engine.

Separate drafter admission is an optional-component outcome retained by the host,
not a numerical property of the target. The draft's own `general.architecture`
names the drafter family that interprets it against the already-interpreted
target: `dflash` is a block drafter whose tapped layers the target family maps
onto its sublayers; `gemma4-assistant` is a Gemma 4 assistant, bound as the
target's draft head (it reads the target's history and needs no target-family
mapping). Any other architecture names no drafter family. Explicit checkpoint
attention semantics are required of block drafters; a missing field disables
drafting, and malformed or unsupported draft semantics retain their distinct,
family-attributed diagnostic while leaving the target available.
The host surfaces the reason once when resolving a load. A method that would
draft with the unavailable component then resolves to plain target decoding:
automatic or explicit separate-draft selection, including when the target also
has an MTP head, and MTP on a target without its own head. An explicitly
requested MTP method on a target with its own head remains independent. Resolution precedes draft workflows, feature capture, weight imports,
state allocation and tuning. Target interpretation errors never become draft
fallbacks. Header assessment and payload-backed loading use the same admission
and method resolution.

An explicitly configured draft-budget policy is shared by MTP and separate
block drafters. It names generated proposal counts and verified prefix counts
within the admitted method cap. The default retains the fixed method width;
adaptive candidates are opt-in and validated before worker construction. The
resolved policy crosses the worker boundary and contributes to generation-method
identity. Plain fallback discards the optional draft policy. Kernel preparation
still covers the admitted cap; selecting a budget never triggers compilation.
The execution owner creates one bounded component-service evidence table for the
loaded adaptive method and injects it into its method factory. Fixed loads allocate
no evidence table because their immutable policy cannot consume adaptive estimates. It is not part
of the manifest, checkpoint, or cross-load cache. Ordinary physical completions
populate it without extra device work; the request-local adaptive controller can
only use covered execution classes. The load also owns ordinary host-service
residuals and committed-prefix evidence, so another request continues the same
whole-round estimate and incumbent. Physical service learning is keyed by the
component's execution class and context coverage, never an action-round counter.
A component curve reports the conditional-mean variance one completion at that
context can resolve; persistent chronological bias remains uncertainty rather
than promised information. Prepared service forecasts derive from selected,
validated kernel measurements and the sealed graph's actual invocation geometry;
capacity dimensions alone cannot identify a live workload. Untimed work remains
an explicit multiset of operation families and nonnegative multiplicities. The
sealed node declares its row axis, independent fixed output count, readonly row
capacities, visible-history dependence and resolved window. Only these declared
axes are normalized; artifact and remaining checked geometry stay exact.
Fixed call overhead, row work, source-declared quadratic row work and stored
history work have separate unknown prices. Quadratic work is an empirical basis,
not an assertion about exact causal attention pair counts; fresh-free nodes do
not declare that term. Ordinary
isolated completions calibrate its residual under a nonnegative additive empirical
model. Bounds may transfer across different multiplicities only when the completed
multiset covers every requested family; unknown geometry remains uncovered. The
allocation of a measured residual among families remains uncertainty, not promised
information. A contradictory signed residual invalidates the forecast rather than
becoming zero service. Residual calibration retains a bounded chronological
window across all work vectors, including changing history lengths. This forecasting model is not an exact latency guarantee.
Fixed requests keep their configured widths and add
no reconciliation timer.

## Host/worker boundary

Only immutable, device-free values cross the worker boundary: the manifest, prepared inputs,
generation options, a serializable constraint description and bounded output. Live tokenizers,
parsers and device objects never cross. Every request ends in exactly one terminal outcome; end of
stream before it is a worker failure, and partially streamed output is never replayed on another
worker. Typed engine errors keep their cause and retry meaning across the boundary.

The service owns process spawn, framing, bounded queues, parent-death containment and crash
handling. The engine owns the protocol values and their encoding, versioned with the engine build.
[Serving](serving.md) defines the protocol library above this boundary;
[scheduling](engine/scheduler.md) and [speculative generation](engine/speculative-generation.md)
define the worker's service contracts.

## Qualification

Tests cover device-free resolution, unique family recognition, preview without allocation,
identical in-process and worker-process behavior, readiness mismatch rejection, one terminal
outcome per request, and worker loss without stranded work. Performance qualification uses the
[measurement boundaries](evaluation.md#measurement-boundaries), not construction success or an architecture name.
