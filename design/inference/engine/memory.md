---
applies_to:
  - inference-v4/engine/model-executor/**
  - inference-v4/engine/model-state/**
  - inference-v4/engine/service/**
  - inference-v4/engine/src/execution.rs
  - inference-v4/engine/src/options.rs
  - inference-v4/engine/src/service/**
  - inference-v4/seismic/runtime/src/**
  - inference-v4/seismic/api/src/lib.rs
  - inference-v4/seismic/backends/cuda/src/driver.rs
  - inference-v4/seismic/backends/cuda/src/executor.rs
---

# Engine memory

One loaded engine owns one heap in the physical memory domain of its selected device. The
domain is host RAM for a CPU or unified-memory device and device-local memory for a dedicated
GPU. Seismic identifies the domain, measures its capacity and live availability, charges every
allocation to it, and enforces the latest limit the heap grants. The heap owns the engine's
holdings and memory decisions; the request owner chooses release victims and runs one release
order. The hosting service supplies one threshold policy (the planning and emergency reserves);
no caller supplies a memory budget or retention percentage.

## Who guarantees what

| Layer | Guarantee | Fires when |
|---|---|---|
| Engine prevention | The engine never causes an out-of-memory condition: every claim leaves headroom above the planning reserve, enforced by Seismic's limit | Always |
| Engine graceful response | When other programs push headroom to or below the planning reserve, the engine releases in the least destructive order and unloads if headroom does not recover | Reclaim band |
| Service guard | The hosting service kills the worker process on the first observation at or below the emergency reserve, independent of the engine's state | The engine is stuck, too slow, or cannot free enough |

The engine is the only layer that decides what to release. The service's kill is fault
containment; in correct operation it never fires.

## Thresholds

Every memory domain has two thresholds from its own capacity: the planning reserve
`P = max(capacity / 10, 2 GiB)` and the emergency reserve `E = max(capacity / 20, 1 GiB)`. The
values are defined once, in the policy the host passes to the engine. They apply equally to host
RAM and to a dedicated device's own memory; there is no other reserve and no OS pressure signal.

## Standing and claims

The heap's standing reports its holdings by class, each used domain's newly observed headroom,
and its band. Every allocation, including startup imports, optional
components, workspace growth and numerical state growth, has a claim before it occurs. A claim
names its holding class, its minimum physical peak charge and any preferred charge for useful
headroom. A reallocation claim includes the interval when old and new backing coexist. Seismic's
charge ledger remains the byte authority: the sum of classified holdings equals its charge.

Stable fit capacity bounds sealed address-space reservations and metadata-only model fit. It is
the allocation domain's total capacity under applicable process limits and, on Metal, the device's
recommended working set, less the planning reserve. A live claim uses fresh headroom for that same
domain, bounded by process limits, and must leave headroom above the planning reserve; on Metal it
must also fit the working set's remaining bytes. The observation already excludes the engine's own
charges and other processes' use, so an existing charge is never subtracted again. A load onto a
dedicated device also claims its staged uploads against host RAM under the host's reserve.

Replacing committed state backing does not release its old allocation while a tensor view or
submitted work still holds it. The state owner observes retired physical allocations without
retaining them, and counts each still-charged allocation once until its last holder releases it.
The holder determines whether that charge is live or in flight. Reclamation receives credit only
for a decrease in Seismic's charge, never for the removal of a state or retention index entry.
An exclusive in-place resize transfers the existing allocation charge: growth claims only its
additional backed bytes, and shrink releases its tail charge only after the backend releases that
physical tail. A held view requires separate backing and preserves the full old charge until the
view is released. A recoverable resize failure preserves the old backing and its charge together.
If a multi-plane growth fails after an earlier plane grew in place, discarding that unpublished
growth restores its prior physical extent and charge before accepted state can run again.
For a mixed multi-plane replacement, planes that need separate backing are formed before any
in-place resize. A shrink prepares every plane but one on separate backing, retaining CUDA's
address reservation for later growth. It unmaps at most one old tail as the final fallible step,
since an unmapped tail cannot restore its former contents if another plane later fails.

The heap grants a claim only while every domain the load uses is in the Normal band. It tries the
preferred charge first, then the minimum. It sets Seismic's enforced limit to current charges plus
the allocation domain's ceiling (headroom less the planning reserve).
A limit may fall below retained charges: that forbids further allocation until releases reduce
the charge. An unclaimed allocation therefore fails within Seismic. A refused minimum leaves
accepted request state unchanged and returns the required and available bytes as a demand
deficit for the request owner to resolve.

Every charged byte belongs to exactly one release class:

| Class | Contents |
|---|---|
| Surplus | Committed state headroom and idle scratch beyond active need |
| Retained | Prefix checkpoints held only for reuse |
| Dormant component | Optional head or vision weights with no active consumer |
| Live | State and method data needed by open requests |
| In flight | Storage held by submitted work until completion |
| Model | Target weights, sealed resources and the pristine recurrent seed |

## Bands

The heap observes every domain it uses on every claim and every 100 ms while loaded. Headroom is
the domain's observed available bytes bounded by applicable process limits: host free-and-inactive
or available memory and commit, CUDA free bytes, or Vulkan budget less usage.

| Headroom | Band | Engine behavior |
|---|---|---|
| Above the planning reserve | Normal | Claims are granted if headroom stays above the planning reserve |
| At or below the planning reserve | Reclaim | Only other programs cause this. Pause admission and growth, release, and unload if it persists |
| At or below the emergency reserve | (still Reclaim) | The hosting service kills the worker on its first observation |

An unavailable required observation, or hidden process limits, is Blind: growth stops
immediately, and a continuous second of Blind is treated as Reclaim. If Reclaim persists for one
second after releases are exhausted and in-flight work completes, the engine unloads the model. An
admission attempted during Blind returns the typed `MemoryObservationUnavailable` result; during
Reclaim it is refused as memory pressure. Already accepted work retains its state while the engine
retries its observation. The engine reads no OS pressure signal.

## Release order

The request owner applies the same order to every deficit and stops when that deficit clears:

1. Release surplus backing and idle scratch.
2. Evict retained prefixes, least recently used first.
3. Unload dormant optional components.
4. For a demand deficit, reduce the pending batch by removing its last request and then
   reducing its token allowance.
5. For demand or Reclaim, preempt live requests while preserving accepted tokens for replay.
6. For demand, wait for a completion, publication, cancellation or peer release to advance the
   resource epoch.
7. For unresolved demand, fail only the affected operation with `InsufficientMemory { required,
   available }`; never unload the model for one oversized claim.
8. For persistent Reclaim, unload the model and finish open and new requests with
   `ModelUnloaded { cause: MemoryPressure }`.

Reclaim uses steps 1–3, 5 and 8 and stops as soon as headroom is back above the planning
reserve; each victim is preempted once, and in-flight work retains its storage until physical
completion. Removed index entries do not count as released bytes until Seismic's charge actually
falls. After unloading, the engine does not reload itself.

The engine reports its standing and typed outcomes. Its hosting service decides whether to
queue or report failed requests and when to reload an unloaded model. The service's emergency
kill is independent fault containment; it chooses nothing to release.

## Acceptance criteria

- Classified holdings sum to Seismic's device charge after every claim, release and unload.
- No new device allocation bypasses a fitting claim, and neither a rejected minimum nor a
  failed physical allocation changes accepted numerical state.
- Reclamation follows the single order and stops when the measured deficit clears.
- No engine claim leaves any used domain's headroom at or below its planning reserve; a claim
  never unloads the model; persistent Reclaim ends in the typed unloaded state within the
  one-second bound.
- Threshold values exist in one policy definition; no code path reads an OS pressure signal.
- Stable fit capacity is capacity under process limits (and the Metal working set) less the
  planning reserve.
- Sealed history and bank reservations are bounded by stable domain capacity, while only
  committed backing is charged against live availability.

## Physical state placement

Logical state ownership and physical backing are separate authorities. The
model-state subsystem owns logical histories, recurrent banks, codec planes and
references. A placement manager owned by that subsystem maps those logical
resources to Seismic-backed physical slots. Callers retain logical identities
and placement snapshots; they never retain mutable physical bank indices.

A relocation is a transaction over a source placement and a destination
placement. It claims the complete simultaneous source/destination peak, copies
all affected planes or banks, waits for completion, validates the destination,
and publishes the new placement atomically. Until publication, the source is
authoritative. On failure, the destination and claim are discarded and the
source rows, values, placement and charge remain unchanged. Retired source
backing is classified as in-flight or pinned until submitted work and external
views release it.

The memory heap is the sole authority for claims, bands, holding classes and
release decisions. Seismic is the sole byte and allocation-charge authority.
The state store does not infer global availability from row counts; retention
does not own a second byte budget; native resource preclaims enter the same
heap; and process supervision chooses no release, reacting only to the heap's
typed unload outcome or to headroom at or below the emergency reserve.
