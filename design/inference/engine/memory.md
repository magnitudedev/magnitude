---
applies_to:
  - inference/engine/executor/**
  - inference/engine/state/**
  - inference/engine/scheduler/**
  - inference/engine/src/census.rs
  - inference/engine/src/execution.rs
  - inference/engine/src/execution/**
  - inference/engine/src/options.rs
  - inference/engine/src/worker/**
  - inference/seismic/runtime/src/**
  - inference/seismic/api/src/**
  - inference/seismic/backends/cuda/src/driver.rs
  - inference/seismic/backends/cuda/src/executor.rs
---

# Engine memory

Ordinary single-device execution owns one heap in the physical memory domain of its selected
device. An explicitly selected pipeline retains one independent heap per participating
device; its budgets are never pooled (see [explicit pipeline execution](pipeline-execution.md)).
Local pipeline admission charges assigned resident weights, projected state, prepared programs,
constants, graph pools, the shared workspace maximum and a separately counted activation
transfer buffer. The ordinary single-device transfer term is zero. The
domain is host RAM for a CPU or unified-memory device and device-local memory for a dedicated
GPU. Seismic identifies the domain, measures its capacity and live availability, charges every
allocation to it, and enforces the latest limit the heap grants. The heap owns the engine's
holdings and memory decisions; the request owner chooses release victims and runs one release
order. The hosting service supplies one threshold policy (the planning and emergency reserves);
no caller supplies a memory budget or retention percentage.

CUDA and Vulkan views with the same physical GPU UUID share one device-local memory domain and
charge ledger. They remain separate execution devices, but hardware inventory counts their
capacity once and presents both backends under that GPU, regardless of model name, device count,
or enumeration order. Distinct UUIDs retain distinct domains even when their names match.

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
components, workspace growth and numerical state slab growth, has a claim before it occurs.
The heap exists from the opened device's first startup allocation: startup claims are heap
claims, each held through its allocation, and the loaded domain inherits the same heap, so no
separate preclaim check or capacity table decides beside it. A refused startup claim fails the load
with `InsufficientMemory` naming the domain that refused it (the allocation domain, or host RAM
for a dedicated device's staging), so the host reports that domain's reserve. A claim is held until Seismic's
charge reflects its physical operation (an import and binding, or an added state slab)
and is then released; the charged bytes join a classified holding. A claim
names its holding class, its minimum physical peak charge and any preferred charge for useful
headroom. Seismic's
charge ledger remains the byte authority: the sum of classified holdings equals its charge.

Stable fit capacity bounds metadata-only model fit. It is
the allocation domain's total capacity under applicable process limits and, on Metal, the device's
recommended working set, less the planning reserve. A live claim uses fresh headroom for that same
domain, bounded by process limits, and must leave headroom above the planning reserve; on Metal it
must also fit the working set's remaining bytes. The observation already excludes the engine's own
charges and other processes' use, so an existing charge is never subtracted again. A load onto a
dedicated device also claims its staged uploads against host RAM under the host's reserve.

State is stored in fixed-size history and bank slabs on every backend: one history slab tensor per
history domain of a store, and one bank slab tensor. The 64 MiB slab target is fixed for an engine
build and contributes to its build identity. A growth claim covers one new slab per domain that
needs one, each joining its domain without copying existing rows or banks. The store releases an empty
slab without a new claim. A slab bound by submitted work remains charged until that work and its
binding views finish; its holding class reflects its strongest remaining holder. Reclamation
receives credit only for a decrease in Seismic's measured charge, never for the removal of a state
or prefix cache entry. A failed allocation or copy preserves the published placement and charge.

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
| Retained | Cached prefixes held only for reuse ([prefix cache](prefix-cache.md)) |
| Dormant component | Optional head or vision weights with no active consumer |
| Live | State and method data needed by open requests |
| In flight | Storage held by submitted work until completion |
| Model | Target weights, sealed resources and the pristine recurrent seed |

The engine's allocation census reports the charge from its reconciliation against every holder
(requests, cached prefixes and submitted work), each byte once: context is surplus, retained, live
and in-flight state and request media; compute is the committed graph pools, prepared programs
and preallocated activation transfer storage,
plus any charge not attributed to a holder; auxiliary is optional component weights; model is
target weights, bound constants and the pristine recurrent seed.

## Bands

The heap observes every domain it uses on every claim and every 100 ms while loaded. Headroom is
the domain's observed available bytes bounded by applicable process limits: host free-and-inactive
or available memory and commit, CUDA free bytes, or Vulkan budget less usage.

| Headroom | Band | Engine behavior |
|---|---|---|
| Above the planning reserve | Normal | Claims are granted if headroom stays above the planning reserve |
| At or below the planning reserve | Reclaim | Only other programs cause this. Refuse admission, hold residency and growth in memory waits, release, and unload if it persists |
| At or below the emergency reserve | (still Reclaim) | The hosting service kills the worker on its first observation |

Process limits are the visible ones. Inside a container the container's own cgroup limit is
visible while cgroups above it may be hidden; hidden ancestors bound nothing beyond host headroom,
so a contained process plans and claims against its own limit and host headroom.

An unavailable required observation is Blind: residency and growth wait on memory
immediately, and a continuous second of Blind is treated as Reclaim. If Reclaim persists for one
second after releases are exhausted and in-flight work completes, the engine unloads the model. An
admission attempted during Blind is refused at once with the typed `MemoryObservationUnavailable`
result; during Reclaim it is refused at once as memory pressure. Already accepted work retains its state while the engine
retries its observation. The engine reads no OS pressure signal.

## Release order

The request owner applies the same order to every deficit and stops when that deficit clears:

1. Release surplus backing and idle scratch.
2. Evict cached prefixes, least recently used first.
3. Unload dormant optional components.
4. For a demand deficit, reduce the pending batch by removing its last request and then
   reducing its token allowance.
5. For demand or Reclaim, preempt live requests while preserving accepted tokens for replay.
6. For demand, park the affected work on its request, which waits alone until capacity is freed
   and the availability epoch advances; the pipeline and other requests continue.
7. For demand nothing can free any more ([scheduler capacity rule](scheduler.md)), fail only the
   waiting requests with their typed capacity error `{ resource, required, available }`; never
   unload the model for one oversized claim.
8. For persistent Reclaim, unload the model and finish open and new requests with
   `ModelUnloaded { cause: MemoryPressure }`.

Reclaim uses steps 1–3, 5 and 8 and stops as soon as headroom is back above the planning
reserve; each victim is preempted once, and in-flight work retains its storage until physical
completion. A release that changes slab bindings needs the binding right
([state transactions](state-transactions.md)), so during a flight it runs at that flight's
completion. Removed index entries do not count as released bytes until Seismic's charge actually
falls. After unloading, the engine does not reload itself.

A loaded owner with multiple explicitly assigned allocation domains reports each
local reconciliation separately. Readiness retains each local plan; periodic
observations label headroom by the actual memory pool identity, never by assuming
all allocation readings belong to the primary domain. Missing domain identity or
an incomplete domain count fails observation rather than underreporting storage.
No summed capacity authorizes a claim on either device.

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
- State growth allocates exactly one claimed slab per growing domain without copying existing
  history or banks.
- Reclaim releases empty slabs and compacts referenced rows and banks into free space in held slabs
  without a new claim, including on Metal, Vulkan, CPU and CUDA.
- No history presents more spans to a launch than its domain's span limit, for any sequence of
  growth, interleaving, forks, window release, rollback and Reclaim; graphs are sealed to that limit
  and a launch beyond it is rejected before submission, never discovered by a graph lookup.
- Startup, lazy component and state-growth allocations all claim from the one device heap.

## Physical state placement

Logical state ownership and physical backing are separate authorities. The
model-state subsystem owns logical histories, recurrent banks, codec components and references.
Each store has history slabs for each of its history domains and bank slabs. A history domain is a
set of attention layers sharing one row numbering: Token (a row per token of the context),
Window(n) (a history references its last `n` rows plus tentative rows) or Shared (no storage; its
layers read a source layer's regions). A history slab contains fixed-offset regions for every
component of its domain; a bank slab contains complete recurrent banks, whose bytes follow from
their components (convolution windows, gated delta and F32 state-space states, and their tapes).
Row numbers identify a slab of their domain and an offset within it. Every history span lies within
one slab. Rows are placed in pages: a domain's page is the smallest multiple of 256 rows for which
its span limit below is at most 63; a slab holds a whole number of pages, and a domain reserves a
whole number of pages, so every page is complete and a growth's demand in whole pages is exactly
what admission checks, provisioning backs and the claim takes. A history takes rows
only in place after its end within its last page, or as whole free pages (pages without a
referenced row), preferring the page that begins at its end; a fresh history takes the middle page
of the largest run of free pages. Every page a history references is therefore complete except its
first and last, so a history of `r` rows presents at most `ceil(r / page) + 1` spans, and a Shared
reader also sees at most `ceil(advance / page) + 1` spans of rows appended within one advance. The
domain's span limit is that bound at its row limit. It is exact, guaranteed for every placement
path (growth, interleaved requests, forks, window release, rollback and Reclaim), and it is the
limit graphs are sealed to and every launch is checked against: a launch class beyond it is
never constructed. The rows after a history's end in its last page are never given to another
history. A history whose last page is partial but whose next row a sibling fork took copies that
page's rows (fewer than a page) into a free page before it grows; that relocation needs one free
page, and without memory for it the request fails before any launch. A history's slack is under
one page per domain. Free space is tracked within each slab, and a freed slab index may be reused.
A Window(n) history releases its references on rows before `n` behind its accepted position after
each advance; those rows are free again, a page is reused once none of its rows is referenced, and
a slab left empty is released by the rules below.
Callers retain logical identities and published placement snapshots, never mutable physical bank
indices.

Assessment charges history per domain at slab granularity: a Token domain by the rows of the
assessed context, and a Window(n) domain by its steady footprint of `n` plus one advance per live
history and `n` per checkpoint, independent of the context, each rounded up to whole pages (plus
one page for a window's partially released first page) and then to whole slabs of the domain.

Compaction is the single mechanism for moving referenced rows or claimed banks between slabs. It
plans destinations in free space of slabs already held, submits all copies, waits for completion,
then publishes the placement with a new generation. Until publication, the source is
authoritative; failure leaves accepted values, placement and charge unchanged. In Reclaim it
empties the least occupied slabs so they can be freed, moving whole occupied pages to free pages at
the same offsets, so every history keeps its page structure and its span limit. Empty slabs are
released at once in Reclaim and, at idle, beyond one free slab per store. No compaction requires a
memory claim.

The memory heap is the sole authority for claims, bands, holding classes and
release decisions. Seismic is the sole byte and allocation-charge authority.
The state store does not infer global availability from row counts; the prefix
cache owns no byte budget or cached price (it bounds its entry count and names the
least recently used victim, and the heap observes what dropping it released);
native resource preclaims enter the same
heap; and process supervision chooses no release, reacting only to the heap's
typed unload outcome or to headroom at or below the emergency reserve.

Complete resource-plan admission checks the startup peak, including transient
assigned-weight import demand, against that allocation domain before allocation.
An insufficient domain retains a typed required/available capacity cause through
model loading; arithmetic or footprint inconsistencies remain internal errors.
