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
order. No caller supplies a memory budget, reserve, or retention percentage.

## Standing and claims

The heap's standing reports its holdings by class, the domain's newly observed available bytes,
and the platform's pressure level. Every allocation, including startup imports, optional
components, workspace growth and numerical state growth, has a claim before it occurs. A claim
names its holding class, its minimum physical peak charge and any preferred charge for useful
headroom. A reallocation claim includes the interval when old and new backing coexist. Seismic's
charge ledger remains the byte authority: the sum of classified holdings equals its charge.

Stable capacity bounds sealed address-space reservations and metadata-only model fit. It is the
allocation domain's total capacity under applicable process limits and, on Metal, the device's
recommended working set. A live claim uses fresh available memory for that same domain. This
observation already excludes the engine's own charges and other processes' use. Neither stable
fit nor a live claim subtracts a fixed reserve or subtracts an existing charge again.

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

The heap grants a claim only while pressure is Normal. It tries the preferred charge first,
then the minimum. It sets Seismic's enforced limit to current charges plus the fresh grant.
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

## Pressure

The heap reads availability on every claim and checks pressure periodically while loaded.
Platform signals, process limits and driver budgets determine pressure; the engine introduces
no byte threshold. Normal permits fitting claims. Pressure pauses growth and releases surplus,
retained state and dormant components until the platform returns to Normal. Emergency pauses
admission and growth, then may preempt live requests after releasable holdings are exhausted.
On Linux, the visible cgroup's current usage reaching `memory.high` supplies Pressure and
reaching `memory.max` supplies Emergency. PSI averages describe past stalls and do not supply
a current pressure level; when no categorical signal is available, fit still uses live headroom
and applicable process limits.
An unavailable required observation is Blind: growth stops immediately, and a continuous one
second of Blind is treated as Emergency. If Emergency persists for one second despite release
and completion of in-flight work, the engine unloads the model. Dedicated CUDA memory has a
fresh free-byte observation but no platform pressure level; its own claim must fit that value.
An admission attempted during Blind returns the typed `MemoryObservationUnavailable` result;
already accepted work retains its state while the engine retries its observation.

## Release order

The request owner applies the same order to every deficit and stops when that deficit clears:

1. Release surplus backing and idle scratch.
2. Evict retained prefixes, least recently used first.
3. Unload dormant optional components.
4. For a demand deficit, reduce the pending batch by removing its last request and then
   reducing its token allowance.
5. For demand or Emergency, preempt live requests while preserving accepted tokens for replay.
6. For demand, wait for a completion, publication, cancellation or peer release to advance the
   resource epoch.
7. For unresolved demand, fail only the affected operation with `InsufficientMemory { required,
   available }`; never unload the model for one oversized claim.
8. For persistent Emergency, unload the model and finish open and new requests with
   `ModelUnloaded { cause: MemoryPressure }`.

Pressure uses only steps 1–3 and waits for recovery rather than preempting live requests.
Emergency uses steps 1–3, 5 and 8; each victim is preempted once, and in-flight work retains
its storage until physical completion. Removed index entries do not count as released bytes
until Seismic's charge actually falls. After unloading, the engine does not reload itself.

The engine reports its standing and typed outcomes. Its hosting service decides whether to
queue or report failed requests and when to reload an unloaded model. Process supervision
contains faults; it does not substitute a second memory-pressure policy.

## Acceptance criteria

- Classified holdings sum to Seismic's device charge after every claim, release and unload.
- No new device allocation bypasses a fitting claim, and neither a rejected minimum nor a
  failed physical allocation changes accepted numerical state.
- Reclamation follows the single order and stops when the measured deficit clears.
- Pressure never preempts live requests; a claim never unloads the model; persistent Emergency
  ends in the typed unloaded state within the one-second bound.
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

The memory heap is the sole authority for claims, pressure, holding classes and
release decisions. Seismic is the sole byte and allocation-charge authority.
The state store does not infer global availability from row counts; retention
does not own a second byte budget; native resource preclaims enter the same
heap; and process supervision reacts only to the heap's typed emergency or
unload outcome.
