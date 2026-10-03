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
    Executor               family program over device-local state storage
```

## Construction

Construction is one staged progression:

```text
EngineConfiguration -> resolve -> ResolvedEngineConfiguration { host, manifest }
                    -> preview (read-only planning)   or   prepare (tuning only)   or   load -> ReadyEngine
```

- **Resolve** is device-free. It opens the package, recognizes its family, interprets the model
  definition and input adapter, and resolves the model policy and served context. Nothing opens a
  device or imports a tensor.
- **Host artifacts** own chat semantics: tokenizer, templates and their inspection, input adapter
  and media placeholder policy. They stay on the host.
- **The execution manifest** is owned and serializable; it is the only value a worker needs.
  It may carry explicitly supplied experimental two-CUDA placement. Without it the ordinary
  single-device path is unchanged. A build without that feature refuses the selected strategy,
  never falls back. Single-device preview and prepare-only refuse paired selection rather than
  misrepresenting it as one-device capacity or preparation.
- **Preview** plans against current devices without opening or allocating anything.
- **Load** runs the same worker either in-process over a channel transport (engine CLI, tests) or
  in a worker process over framed standard streams (the service). The worker protocol and the
  worker code are identical in both cases.
- **Prepare** runs the same worker, transport and build handshake through the preparation a load
  begins with (opening the device and preparing programs, tuning whatever the kernel cache lacks),
  reports preparation and tuning progress as a load does, and exits once prepared. Tuning imports
  only the weights of the few layers its cases rotate over; a prepared worker never allocates
  serving state, imports the whole model or serves.
- **Paired load** uses the same generic execution owner, worker and session, retaining both
  prepared local allocation domains and paired request state. Readiness carries separate stage
  plans and both actual physical allocation censuses; the legacy primary plan describes only
  the readout domain. Observations map each actual memory pool to its own physical device.
  VRAM capacities are not pooled. See [explicit pipeline execution](engine/pipeline-execution.md).
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
[benchmark hierarchy](benchmarking.md), not construction success or an architecture name.
