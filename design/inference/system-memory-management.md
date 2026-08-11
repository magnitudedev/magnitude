---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-contracts/src/**
  - inference/crates/icn-server/**
  - packages/acn/src/local-inference-**
  - packages/acn/src/local-model-**
  - packages/acn/src/model-slot-**
  - packages/acn-protocol/src/**
  - packages/client-common/src/hooks/**
  - cli/src/features/local-inference/**
  - cli/src/features/model-*/**
---

# System memory management

ICN uses one memory-safety policy for model assessment, load admission, and serving-time eviction.
Assessment decides what a machine can normally serve, admission decides whether it is safe to load
now, and eviction protects the machine when conditions change. These decisions must use the same
physical memory domains and peak-memory evidence.

The ownership and normalization rules in
[Client, ACN, and ICN ownership](../cross-boundary/client-acn-icn-ownership.md) apply. In
particular, platform-specific memory mechanisms terminate inside ICN, while Magnitude warning
policy belongs to ACN.

## Policy

For total physical system memory `T`, ICN owns:

```text
assess reserve  S = max(10% of T, 2 GiB)
abort reserve   B = max( 5% of T, 1 GiB)
```

ACN independently owns recommended application headroom `H = max(20% of T, 4 GiB)`. `H` is
advisory product policy; it does not participate in assessment, admission, or eviction.

For a serving profile, `M` is the native planner's complete predicted system-domain allocation:
weights, context and KV, compute buffers, projector allocations, and target/draft allocations.
File size plus a fixed allowance is not valid evidence.

Stable product compatibility uses the exact one-sequence, one-complete-context baseline. During
load, ICN evaluates native sequence capacities from one through four. Candidate `P` uses complete
physical context `configured context × P`; ICN selects the greatest candidate that satisfies
stable compatibility and fresh live admission. Current pressure may reduce selected `P`, but never
configured per-request context.

## Decisions

Stable compatibility ignores current activity:

```text
compatible iff M + S <= T
```

Assessment publishes the factual byte accounting for every physical memory domain: capacity,
required allocation, compatibility reserve, and remaining compatible headroom. It does not publish
or persist application warning thresholds or presentation categories. ACN trusts the terminal
`Fits` / `DoesNotFit` result and applies its own recommended-headroom policy only to fitting
configurations; catalog availability is not a second capacity signal.

Internal capacity assessments name the post-reserve quantity `usable_capacity_bytes`. The terms
`physical available` and `allocation headroom` are reserved for live observations; neither denotes
cached compatibility capacity.

Reserve selection produces one fingerprinted hardware snapshot whose topology contains the
resulting stable capacity for every domain and device-local limit. Planning workers, load workers,
accounting, and cache validation consume that same snapshot. They cannot receive or apply a second
reserve policy after topology capture.

Load admission uses a fresh normalized allocation-headroom sample `A`, taken again after planning
and immediately before the worker is created:

```text
admit iff A > M + B
```

Unknown peak evidence and a native `DoesNotFit` result fail closed. Every successful hardware
snapshot includes both physical memory and normalized system-allocation capacity/headroom;
inability to obtain any platform input required for the normalized observation fails the snapshot
rather than publishing an unknown value. On Windows, commit accounting is an internal platform
input to the common allocation values and never appears as a separate public contract branch.

Memory-domain identity is one typed physical-topology concept shared by discovery, planning,
assessment, cache validation, and resident allocation accounting. The planner and hardware topology
use the single canonical identity `system` for memory shared with the operating system. Dedicated
domain identities are derived from normalized physical-device identity, never display names.
All native host/device locations are interpreted by that topology's single resolver. Native fit,
speculative, projector, performance, and runtime evidence can report locations and bytes but cannot
create domains or decide whether memory is shared.
Each source normalizes its evidence into one charge per owner and allocation location. A charge
carries one complete breakdown of model, context, compute, and auxiliary bytes. The shared
accountant resolves that location once and aggregates the complete breakdown into both its physical
domain and any applicable device-local limit. Category fragments and independently maintained
required-byte totals are not valid accounting inputs.
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

Separately, a resident generation is released normally after ten continuous minutes with no
accepted inference lease. The monotonic interval begins when the generation becomes ready without
a lease or when its final lease ends. Metadata and observation do not count as activity.

Inference admission and idle release share the backend mutation boundary. A request that acquires
a lease first runs to completion and starts a fresh full interval after the final lease. Idle
release that closes admission first rechecks the exact generation, zero active leases, and the
complete elapsed interval before publishing `IdleTimeout` releasing state and gracefully reaping
the worker. A stale deadline cannot affect a replacement generation, and idle release does not
enter pressure recovery.

## Memory domains

A physical allocation is charged once. Unified CPU/GPU allocations use the system reserve.
Dedicated device domains retain their independent ICN assessment reserve; the larger system
reserve is never subtracted from dedicated VRAM. ACN current-headroom guidance compares ICN's
normalized allocation headroom with the assessment's system-domain requirement. Because loading
replaces the singleton worker before admission, ACN credits the current resident worker's
system-domain allocation once, capped by ICN's normalized allocation capacity.

## Product behavior

ICN owns and publishes the assessment and abort thresholds. ACN owns the recommended-headroom
formula and publishes derived application guidance; clients consume that guidance rather than
reconstructing it. A resident worker failure is published with its configuration identity and
becomes a typed slot residency-loss state.

An assigned model remains configured while it is loading, unloaded, or blocked. Loadability is
separate, transient state; a failed load must not erase the selection or make the client report that
no provider is configured. Admission failures are model-not-ready results from the pre-provider
preparation phase, not provider or network failures, so they do not enter connection retry/backoff
handling. A low-memory failure preserves the required allocation, normalized allocation headroom,
safety reserve, strict load boundary, minimum additional availability, and
parallel sequence count. Other operation failures preserve their typed code, message, and whether
a later manual retry may succeed. These facts stay on the model-instance boundary without exposing
local-inference concepts in the provider contract, and clients own their presentation rather than
parsing server prose.

The Models menu keeps `REQUIREMENTS` and uses `STATUS` for:

- `Tight fit` — detail: `High memory use`
- `Too large` — detail: `Requires more memory than this system has`
- `Free memory` — detail: `Not enough memory available - close memory-intensive apps`

`Too large` is stable. `Free memory` is reactive and clears when availability changes. A
low-memory rejection during load or termination during serving appears in the activity rail as:

`Model stopped · Low memory - close memory-intensive apps and try again`

The CLI footer notification area also projects insufficient current headroom for the selected
local model as `! Low memory: close memory-intensive apps (need X.Y GB) to load model`, using the
published minimum additional availability. In the composer's two-row footer layout, it is presented
compactly as `! Low memory: Free X.Y GB to load`. This warning persists while the authoritative
current-headroom state remains insufficient and takes precedence over download activity; it is not
an ephemeral response to the selection event.

Onboarding keeps its recommendation and action labels unchanged in the model list. The selected
model's right-hand detail pane preserves the recommendation explanation and independently presents
stable fit and current loadability. The metadata, explanation, and memory guidance are separated by
one blank row each. When current headroom is insufficient, the compact memory block tells the user
the published minimum additional availability to free before loading, explains that the model fits
stable total capacity while current system use prevents loading, and suggests closing
memory-intensive applications. This live guidance is advisory: it neither changes recommendation
rank nor authorizes loading. If authoritative admission still rejects the attempt, the same pane
presents the structured low-memory facts and a retry action without exposing the raw diagnostic as
primary copy.

While the selected local model is ready, the composer footer shows its correlated model-instance allocation
after context: the sum of the server-published model, context, compute, and auxiliary allocations
across participating memory domains. It is compactly labeled, for example `24 GB mem`; it is not
whole-system used memory and has no capacity denominator. Memory disappears outside ready
residency and has no placeholder or transitional state. The Hardware menu owns whole-system,
application, free-memory, and per-allocation detail.

The Hardware menu places a flat `CURRENT MODEL` section below hardware identity and above memory.
It always shows the selected local model, including while unloaded, and derives loading, running,
stopping, unloaded, and failed state from `ModelSlots`. Running and resident stopping use
allocation evidence carried by the same model-instance slot state, while aborting a pre-residency
load retains the tagged planned-allocation form. The hardware mirror supplies only
topology, capacity, and live availability. While unloaded, an authoritative ICN preview runs the
same exact configuration planner,
one-through-four candidate assessment, and fresh admission policy as a real load, and displays the
parallelism it would select now. Clients never estimate this value. Preview evidence is advisory
and never gates assignment or load; ICN repeats authoritative admission when the actual load is
submitted. The mounted Hardware view requests this plan through one observational ACN query.
Neither ACN nor the client stores it in canonical model state or evaluates it in the background.

Frozen-topology candidate assessments are cached as disposable derived evidence and shared by
preview and load. Live hardware polling therefore reruns only the fresh admission selection when
stable assessment evidence is unchanged. Catalog assessment and recommendation work does not rerun for
availability-only hardware changes.

The menu's state-appropriate Actions load or Stop the selected model without destructive
confirmation. Stop carries only the exact model-instance identity. The activity rail exposes the
same Stop while loading; both actions converge through the single slot lifecycle and operation
owner.

While current hardware or system-domain evidence is unavailable, clients do not infer either
compatibility or available headroom and present memory status as unavailable. Model selection
remains a durable configuration action; current load admission remains server-authoritative and
fails closed without complete evidence. This transient unknown state is not mislabeled as
`Free memory`.
