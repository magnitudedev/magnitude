---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-contracts/src/**
  - inference/crates/icn-server/src/main.rs
  - inference/crates/icn-server/src/memory_supervisor.rs
  - packages/acn/src/local-inference-hardware.ts
  - packages/acn/src/local-model-*.ts
  - packages/acn/src/model-slot-coordinator.ts
  - packages/acn/src/model-request-preparation.ts
  - packages/agent/src/errors/model-start.ts
  - packages/protocol/src/schemas/model-state.ts
  - packages/sdk/src/index.ts
  - packages/client-common/src/utils/model-memory.ts
  - packages/client-common/src/utils/model-slots.ts
  - cli/src/app.tsx
  - cli/src/features/composer/**
  - cli/src/features/local-inference/footer-status.tsx
  - cli/src/features/model-menus/**
  - cli/src/features/agent-status/**
---

# System memory management

ICN uses one memory-safety policy for model fitting, load admission, and runtime eviction. Fitting
decides what a machine can normally serve, admission decides whether it is safe to load now, and
eviction protects the machine when conditions change. These decisions must use the same physical
memory domains and peak-memory evidence.

## Policy

For total physical system memory `T`:

```text
warning reserve W = max(20% of T, 4 GiB)
assess reserve  S = max(10% of T, 2 GiB)
abort reserve   B = max( 5% of T, 1 GiB)
```

For a serving profile, `M` is the native planner's complete predicted system-domain allocation:
weights, context and KV, compute buffers, projector allocations, and target/draft allocations.
File size plus a fixed allowance is not valid evidence.

## Decisions

Stable compatibility ignores current activity:

```text
compatible iff M + max(product reserve, S) <= T
tight fit iff compatible and M + W > T
```

Assessment publishes the factual byte accounting for every physical memory domain: capacity,
required allocation, compatibility reserve, warning reserve, and remaining compatible headroom.
It does not persist presentation categories such as comfortable, tight, or too large. ACN and
clients derive compatibility and warning presentation directly from those quantities; catalog
availability is not a second fit signal.

Internal fitting assessments name the post-reserve quantity `usable_capacity_bytes`. The term
`available` is reserved for live observations such as `current_available_bytes`; it never denotes
cached compatibility capacity.

Load admission uses a fresh whole-system availability sample `A`, taken again after planning and
immediately before the worker is created:

```text
admit iff A > M + B
```

Unknown peak evidence and a native `DoesNotFit` result fail closed. Every successful hardware
snapshot includes a current whole-system availability measurement; inability to obtain one fails
the snapshot rather than publishing an unknown value. Windows commit availability is an additional
independent gate using the same admission, eviction, and recovery boundaries.

Memory-domain identity is one typed physical-topology concept shared by discovery, planning,
assessment, cache validation, and resident allocation accounting. The planner and hardware topology
use the single canonical identity `system` for memory shared with the operating system. Dedicated
domain identities are derived from normalized physical-device identity, never display names.
Every complete native assessment includes the system domain, including an explicit zero-byte
requirement when all planned allocations are charged to dedicated devices. Missing, duplicate, or
unknown domain identities are incomplete evidence and fail closed.

The disposable inference worker is supervised from spawn through loading and serving every 100 ms:

```text
terminate worker iff A <= B
```

Monitor loss for one second also terminates the worker. After a pressure termination, admission
stays closed until availability exceeds `B + 512 MiB` for five seconds. There is no automatic
reload.

## Memory domains

A physical allocation is charged once. Unified CPU/GPU allocations use the system reserve.
Dedicated device domains retain their independent product reserve; the larger system reserve is
never subtracted from dedicated VRAM. UI loadability compares current system availability only
with the assessment's system-domain requirement. Because loading replaces the singleton worker
before admission, the UI adds the current resident worker's system-domain allocation to predicted
post-unload availability.

## Product behavior

ICN owns and publishes the thresholds. ACN and clients consume the values rather than reconstructing
the formulas. A resident worker failure is published with its configuration identity and becomes a
typed slot runtime-loss state.

An assigned model remains configured while it is loading, unloaded, or blocked. Loadability is
separate, transient state; a failed load must not erase the selection or make the client report that
no provider is configured. Admission failures are generic model-not-ready results from the
pre-provider preparation phase, not provider or network failures, so they do not enter connection
retry/backoff handling. Their typed code, message, and whether a later manual retry may succeed are
preserved without exposing local-inference concepts in the provider contract.

The Models menu keeps `REQUIREMENTS` and uses `STATUS` for:

- `Tight fit` — detail: `High memory use`
- `Too large` — detail: `Requires more memory than this system has`
- `Free memory` — detail: `Not enough memory available - close memory-intensive apps`

`Too large` is stable. `Free memory` is reactive and clears when availability changes. A
low-memory rejection during load or termination during serving appears in the activity rail as:

`Model stopped · Low memory - close memory-intensive apps and try again`

The chat shell keeps a compact local-inference badge overlaid at the upper-right of the scrolling
timeline. While a local model is ready, the badge shows its resident runtime allocation: the sum of
the server-published model, context, compute, and auxiliary allocations across participating memory
domains. It is not whole-system used memory and has no capacity denominator. Memory disappears
outside ready residency; transitional, idle, and failed badge states remain visible without a
placeholder. The Hardware menu owns whole-system, application, free-memory, and per-allocation
detail.

While current hardware or system-domain evidence is unavailable, clients do not infer either
compatibility or available headroom and present memory status as unavailable. Model selection
remains a durable configuration action; current load admission remains server-authoritative and
fails closed without complete evidence. This transient unknown state is not mislabeled as
`Free memory`.
