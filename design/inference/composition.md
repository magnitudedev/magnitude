---
applies_to:
  - inference-v2/src/magnitude_engine/**
  - inference-v2/tests/composition/**
  - inference-v2/tests/worker/**
---

# Inference composition

An engine coordinates requests; a model executor advances model state. Even with
one loaded target, these are separate responsibilities. Components receive their
dependencies explicitly rather than looking up the containing engine.

```text
Host: artifact acquisition, typed engine blueprint, worker client, HTTP rendering
                       │ graph + requests / bounded results
Worker: construction scope
  Engine
    Scheduler                  admission and bounded service
    Memory budget / pressure   capacity and reclamation
    Prefix index / retention   matching complete generation checkpoints
    Generation
      Target executor
        Program                architecture and neural computation
        State store            transactional KV and recurrent state
        Execution owner        device completion and resource lifetime
      Generation method        plain or proposal / verify / accept
        Drafter executor       program and state, when required by the method
```

## Construction and binding

Blueprints are immutable, statically typed dependency declarations. Each concrete
declaration uses `@component` and one lazy `implementation()` binding to its live
contract. Declarations and implementations live together in their owning domain;
the public blueprint namespace is a backend-free facade. There is no codegen,
separate registration, import-order initialization, or per-component version.

The serialized graph preserves shared node identity. Its bounded codec accepts
declared scalar values, tuples and blueprint references, rejects invalid or cyclic
graphs, and resolves only catalogued declarations. Saved graphs require matching
source/runtime code; arbitrary implementation imports and compatibility shims are
not part of the wire contract.

The worker validates the complete constructor wiring before acquiring resources.
It constructs dependencies once per graph identity and closes owned components
once, in reverse dependency order. Constructors unwind their own partial failures.
Both primary and cleanup failures remain visible.

An executor blueprint pairs a program source with a compatible state factory.
Binding that pair to model resources loads weights, derives artifact geometry and
produces the live executor. Host declarations contain neither device objects nor
mutable request state. The automatic artifact composition uses a resident upstream
program and native state; custom computation, streaming and drafting are explicit
component substitutions, never execution-level flags. Upstream architecture reuse
does not imply that every cache, media or batching capability is qualified.

## Runtime boundaries

The engine owns scheduling and prefix policy. Executors receive a budget and an
execution owner, not access to the scheduler or prefix index. Prefix entries hold
opaque complete generation checkpoints, including any drafter alignment obligation.
Physical KV placement, page mappings and recurrent images belong to state storage.
Slab allocation is a storage choice; attention computation consumes a compatible
view without becoming an admission or retention policy.

Architecture programs compose their embedding, attention/recurrent mixing and
feed-forward dependencies. Resident or streamed weight implementations retain
their I/O, scratch and consumer-lifetime complexity behind those contracts.
Replacing a component must preserve its outputs, state effects and device-lifetime
guarantees; unsupported combinations fail during binding rather than adding engine
branches.

The generation method owns the target–drafter relationship: features, proposals,
acceptance, repair and linked checkpoints. The scheduler sees bounded service and
resource feedback. [Scheduling](engine/scheduler.md) and
[speculative generation](engine/speculative-generation.md) define those contracts.
The host owns [serving](serving.md), process supervision and bounded transport;
the worker alone owns the live engine and neural resources.

## Qualification

Tests must cover backend-free declaration imports, shared identity after graph
round-trip, whole-graph preflight, partial-construction cleanup and observed worker
disposal. Component substitution must preserve state and lifetime contracts.
Performance qualification uses the [benchmark hierarchy](benchmarking.md), not
construction success or a model's architecture name.
