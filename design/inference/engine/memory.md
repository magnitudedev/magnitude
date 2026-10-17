---
applies_to:
  - inference/engine/executor/**
  - inference/engine/state/**
  - inference/engine/scheduler/**
  - inference/engine/src/execution.rs
  - inference/engine/src/options.rs
  - inference/engine/src/worker/execution.rs
  - inference/seismic/runtime/src/**
  - inference/seismic/api/src/**
  - inference/seismic/compiler/src/blocks.rs
  - inference/seismic/backends/metal/src/device.rs
  - inference/seismic/backends/cuda/src/blocks.rs
  - inference/seismic/backends/cuda/src/direct.rs
  - inference/seismic/backends/cuda/src/driver.rs
  - inference/seismic/backends/cuda/src/executor.rs
  - inference/seismic/backends/cuda/src/nvml.rs
  - inference/seismic/backends/cuda/src/open.rs
  - inference/seismic/backends/vulkan/src/device.rs
  - inference/seismic/backends/vulkan/src/direct.rs
  - inference/seismic/backends/vulkan/src/lib.rs
  - inference/seismic/backends/vulkan/src/memory.rs
---

# Engine memory

One loaded engine owns one heap in the physical memory domain of its selected device. The
domain is host RAM for a CPU or unified-memory device and device-local memory for a dedicated
GPU. Seismic identifies the domain, measures its capacity and live availability, charges every
allocation to it, and enforces the latest limit the heap grants. The heap owns the engine's
holdings and memory decisions; the request owner chooses release victims and runs one release
order. The hosting service supplies one threshold policy (the planning and emergency reserves);
no caller supplies a memory budget or retention percentage.

A dedicated device's capacity is its usable memory: the device's total less the driver and runtime
overhead the process cannot use for its own allocations, namely the driver's reserve, what the
operating system withholds from a process (WDDM), and the process's own context. One function per
device computes it, once per process and physical device, before the process allocates anything;
the topology's pool carries it, and stable fit, assessment, the catalog's fit verdict,
recommendations and the engine's claim ceiling all read that one figure:

| Device | Usable capacity |
|---|---|
| A GPU with a CUDA view, on Windows (WDDM) | `cuMemGetInfo` free bytes once the process's context exists and before it allocates: the device total less the reserve, the WDDM share and the context, which the driver reports as one figure, per process and independent of other programs |
| A GPU with a CUDA view, elsewhere (Linux) | NVML's total less its reserve (`nvmlDeviceGetMemoryInfo_v2`) less the process's context, measured as the fall in NVML's device-wide free bytes across creating it, and never less than `cuMemGetInfo` free once the context exists. `cuMemGetInfo` free there is the whole device's, lowered by what other programs hold, so it alone is not the capacity. NVML is found by the device's CUDA UUID. Without NVML, its v2 memory query or a matching device, or when the process already has the device's context, the capacity is `cuMemGetInfo` free: exact when no other program holds memory at discovery, lower by what they hold otherwise |
| A GPU with only a Vulkan view, under WDDM | The device-local heap's `VK_EXT_memory_budget` budget, WDDM's per-process budget |
| A GPU with only a Vulkan view, elsewhere | The device-local heap, which excludes the driver's reserve |

The device's total memory is descriptive only. The live free-memory reading (CUDA free bytes, Vulkan
budget less usage) is only the guard against other programs; it adds no capacity.

CUDA and Vulkan views with the same physical GPU UUID share one device-local memory domain and
charge ledger. They remain separate execution devices, but hardware inventory counts their
capacity once, as its CUDA view's usable capacity when it has one, and presents both backends under
that GPU, regardless of model name, device count, or enumeration order. Distinct UUIDs retain
distinct domains even when their names match.

## Who guarantees what

| Layer | Guarantee | Fires when |
|---|---|---|
| Engine prevention | The engine never causes an out-of-memory condition: every claim leaves headroom above the planning reserve, enforced by Seismic's limit, and a load stops when its host is in distress | Always |
| Engine graceful response | When a domain enters Reclaim, the engine releases in the least destructive order and unloads if the domain does not recover | Reclaim band |
| Service guard | The hosting service kills the worker process on the first observation at or below the emergency reserve or at critical kernel pressure, independent of the engine's state | The engine is stuck, too slow, or cannot free enough |

The engine is the only layer that decides what to release. The service's kill is fault
containment; in correct operation it never fires.

## Thresholds

Every memory domain has two thresholds from its own capacity, the planning reserve `P` and the
emergency reserve `E`:

| Domain | Planning reserve `P` | Emergency reserve `E` |
|---|---|---|
| Host RAM, unified memory included | `max(capacity / 10, 2 GiB)` | `max(capacity / 20, 1 GiB)` |
| A dedicated device's own memory | `max(capacity / 20, 512 MiB)` | `max(capacity / 40, 256 MiB)` |

Host RAM keeps room for the operating system and other programs whatever the machine's size. A
dedicated device's capacity already excludes the driver's reserve, the operating system's share and
the process's own context, and everything the engine's work takes of it later is charged or held,
so its reserve is a small margin over the measured residual. The values are defined once, in the
policy the host passes to the engine; there is no other reserve. Host distress (Bands) is the only
other condition that moves a domain's band.

The device reserve is sized by measurement on the smallest supported devices. With every byte of
the engine's device usage charged or held (Standing and claims), what remains is the residual: the
capacity less observed headroom less `held`. On an 8 GiB A10-8Q vGPU (CUDA, WDDM) it stayed within
−50 to +20 MiB across first-load tuning, 8-way bursts after long prompts, 78K-token contexts, Gemma 4
images and perf runs; on an 8 GiB Radeon PRO V710 slice (Vulkan, WDDM) within −65 to +5 MiB through
the 104K-token Qwen3.5 4B sequence (`specs/26-10-11/gpu-memory/notes.md`). The 512 MiB floor is
therefore not a margin for untracked growth: it covers one recorded graph's instantiation between
two observations, a block's granule rounding, and on a device without WDDM other programs'
allocations between observations, with the residual 25 times over. A device whose residual
approaches the floor raises it here, never a per-backend constant elsewhere.

## Standing and claims

The heap's standing reports its holdings by class, each used domain's newly observed headroom,
and its band. Every allocation, including startup imports, optional
components, workspace growth and numerical state slab growth, has a claim before it occurs.
The heap exists from the opened device's first startup allocation: startup claims are heap
claims, each held through its allocation, and the loaded domain inherits the same heap, so no
separate preclaim check or capacity table decides beside it. A refused startup claim fails the
load, naming the domain that refused it (the allocation domain, or host RAM for a dedicated
device's staging): with `InsufficientMemory` and the bytes required and available when the claim
exceeds the domain's ceiling, or with `MemoryPressure` and no byte count when the domain is in
Reclaim. A weight import runs under one claim for as long as its weights take to become resident,
so it observes its domains every 100 ms as weights are published and fails the load with
`MemoryPressure` if one enters Reclaim. A claim is held until Seismic's
charge reflects its physical operation (an import and binding, or an added state slab)
and is then released; the charged bytes join a classified holding. A claim
names its holding class, its minimum physical peak charge and any preferred charge for useful
headroom. Seismic's
charge ledger remains the byte authority: the sum of classified holdings equals its charge.

Beside the charge, Seismic reports what each opened device holds of its memory domain (`held`): the
charge plus the device memory its backend holds beyond its allocations, so nothing the engine's
work takes from the device grows outside it. CUDA and Vulkan take device memory from the driver in
blocks (64 MiB, doubling to 1 GiB, a larger request in a dedicated block of whole driver granules)
and sub-allocate every allocation from them, so a model's many tensors cost no per-allocation driver
rounding (under WDDM every `cuMemAlloc` takes whole 2 MiB pages); a block is freed once it holds no
allocation. `held` counts the blocks' free space (`unallocated`, which later allocations take
without the device holding more) and the driver memory the engine's work takes outside every block,
each measured as the change in the driver's own figure across the operation that takes it: a CUDA
graph's instantiation (free memory); a Vulkan device's creation, pipeline creation, the recording of
a graph and of a submission (the process's heap usage, less blocks the operation created), and its
own argument and staging ranges. A device's recorded graphs hold at most a fixed 192 MiB, the least
recently replayed dropped beyond it, and assessment plans that bound as a term of every CUDA and
Vulkan load. A new block is sized within the room the limit grants less the free space blocks
already hold, and holds only its allocation when a doubled block does not fit that room.

Stable fit capacity bounds metadata-only model fit. It is
the allocation domain's capacity under applicable process limits and, on Metal, the device's
recommended working set, less the planning reserve. A live claim uses fresh headroom for that same
domain, bounded by process limits, and must leave headroom above the planning reserve; on Metal it
must also leave the working set's remaining bytes above the same reserve, and on a dedicated device
the capacity less what the device holds (`held`) above it, so what the engine holds never exceeds
stable fit capacity: fit and the live ceiling share one capacity basis, and a live figure that later
grows never lets a load outgrow what assessment promised. A claim may also take the device's
`unallocated` block space, which it fills without the device holding more. The driver's submission memory lies outside the engine's charge,
and a working set filled to its end fails a submission as a lost device instead of refusing a claim.
The observation already excludes the engine's own charges and other processes' use, so an existing
charge is never subtracted again. Released storage returns to its domain: a backend that
sub-allocates device memory frees each underlying allocation once it holds no storage, so a
released charge reappears in observed headroom and the release order can satisfy a deficit. A load onto a
dedicated device also claims its staged uploads against host RAM under the host's reserve.
A stored table's slots and directory ([residency](residency.md)) are host-shared storage (Seismic's
upload allocation), charged to the domain it is physically allocated in, which the device reports
(`DeviceMemoryInfo::shared_pool`): the allocation domain of a unified-memory device (Apple, DGX
Spark, the CPU) and of a Vulkan device whose upload memory is a large host-visible device-local
heap (ReBAR/SAM); host RAM, the staging domain, of a dedicated CUDA device (mapped pinned memory)
and of a Vulkan device without one. Assessment charges them there
(`AssessmentMemoryCharge::assess_fit`), so a load that does not fit host RAM is refused although
the device holds the rest; the target import's claim stages them through host RAM; and the census
reports them in host RAM. Seismic charges every allocation of a device, these included, to its
ledger, so its enforced limit, charges plus the device's ceiling, is unchanged by them.

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
the charge. An unclaimed allocation therefore fails within Seismic. Beside its high-water mark, Seismic's
usage reports the largest demand its limit refused since the mark was restarted (what was charged
with what was asked for), so a consumer such as kernel tuning tells a refusal by the limit from any
other failure and learns what the refused work needed. A refused minimum leaves
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
and in-flight state and request media; compute is the committed graph pools and prepared programs,
plus any charge not attributed to a holder; auxiliary is optional component weights; model is
target weights, bound constants and the pristine recurrent seed. Standalone feature storage,
including an open drafter boundary, is counted once across all physical owners and submitted
consumers. Weak census observations and generation obligations do not extend its lifetime;
reclaim estimates exclude allocations still owned by a fork, retained prefix or submitted work.

## Host memory of a load

A load's worker holds system RAM beside its device holdings, and assessment charges all of it there
(the allocation domain of a unified device, the staging domain of a dedicated one), held from the
preparation of the programs to the end of serving, so it is charged once beside the larger of the
load and serving phases. Two kinds of term make it up:

- Exact from the plan: the host tables' row caches, the most host heap the target and state copy
  graphs' sealed plans hold, the target graphs' static bindings, and the drafter's graphs (a head's,
  shared head's or separate draft's sealed plans and constant values).
- The worker's own memory, which no structure of the plan describes: its runtime (process, Seismic's
  catalog, the opened device), the model's vocabulary (the artifact's token tables, the scheduler's
  tokenizer and the grammar vocabulary), its prepared programs (checked entries, prepared
  implementations and compiled pipelines, the call storage qualification leaves with them, and the
  graph preparation they drive beyond the sealed plans), and for a model that reads images the vision
  encoder's graphs and programs, prepared when a request first sends one. Each is a count the plan
  fixes times a cost measured on the backend: a runtime constant, a cost per vocabulary entry, a cost
  per native kernel of the load's kernel inventory (the device-free listing of the kernels its
  preparation forms) and a vision constant.

An optional component a request loads (vision or a drafter head) claims its import window beside its
weights; assessment charges that window at the fit depth, as the serving phase's peak.

The costs are measured, not margins: from loads on the backend, the process footprint at load end
less everything the plan holds at that moment (weights, prepared resources, startup state and the
exact host terms) is the worker's own memory, and the costs are the least-overestimating bound
under which no measured load exceeds its plan. A load whose footprint exceeds what its plan holds
raises them here, never a margin elsewhere.

| Backend | Runtime | Per vocabulary entry | Per kernel | Vision | Measured on |
|---|---|---|---|---|---|
| Metal | 175 MB | 1,800 B | 5.7 MB | 210 MB | M4 Max, macOS 15.5: MiniCPM5 1B, Qwen3.5 0.8B, 2B and 4B, LFM2.5 1.2B, Gemma 4 E2B QAT (with its drafter and projector) and Q4_K_M (`bugs/26-10-17/unplanned-host-heap-in-memory-verdict.md`) |
| CPU, Vulkan, CUDA | Metal's | Metal's | Metal's | Metal's | Not yet measured: the vocabulary and program structures are the engine's own and the same on every backend, the runtimes are not |

A dedicated device also loses memory to the worker outside every charged allocation, and assessment
charges it in the allocation domain with the prepared resources: the loaded images of the native
kernels the load prepares (CUDA modules, Vulkan pipelines), a cost per kernel of the kernel inventory,
and the driver's per-process context where the usable capacity does not already exclude it. A CUDA
context is measured on the device itself when its capacity is measured and is outside that capacity,
so it is not charged; a Vulkan device's creation takes from the heap or budget its capacity is, so it
is. A unified device's kernels and runtime are in its worker's host bytes.

| Backend | Context | Per kernel | Measured on |
|---|---|---|---|
| CUDA | In the usable capacity | Not yet measured | A10G, L40S (Linux), A10 (Windows): round 3 |
| Vulkan | Not yet measured | Not yet measured | Radeon PRO V710 (Linux, Windows): round 3 |
| Metal, CPU | In the host bytes | In the host bytes | |

The device costs are measured as the host costs are: the device memory the GPU lost at each load
moment, less what the engine holds then (`held`), by
`inference/scripts/device-memory-costs.py` on an otherwise idle GPU.

## Bands

The heap observes every domain it uses on every claim and every 100 ms while loaded. Headroom is
the domain's observed available bytes bounded by applicable process limits: host movable memory
(macOS: free, active and inactive pages, every page neither wired nor compressed), available memory
(Linux) or available memory and commit (Windows), CUDA free bytes, or Vulkan budget less usage.

On macOS every engine holding is wired (a Metal residency set, or locked CPU-backend memory), so no
holding appears in movable memory and the kernel never compresses or swaps one. A page list does
not say whether a page is needed, so macOS headroom counts every movable page: a claim is refused
only when it cannot fit beside wired and compressed memory.

A residency set's residency is a request that can lapse about 1 s after the set's last GPU work (macOS
26 on an M1 Ultra, with or without memory pressure; a 3 GB model on macOS 15.5 stayed wired). macOS
then un-wires all of it (weights, state and scratch alike) into active pages, frees essentially
nothing by doing so (quantized weights do not compress), and the next GPU work pays about 10 ms per
GB to wire it again before its first result. The Metal device therefore renews its set's residency
every 100 ms for as long as the set holds an allocation, so holdings stay wired through every idle
gap; there is no idle timeout. Memory is returned by releasing holdings (the release order, or
unloading), never by letting residency lapse. Renewal starts with the set's first allocation and
stops, its thread joined, when the last one is released.

Residency remains a request: under extreme shortage macOS may still revoke it and compress some of
the holdings, and the next renewal asks for them back. Headroom subtracts nothing for this. Wired
holdings are already outside movable memory, so subtracting the engine's charge would count them
twice and refuse claims that fit. A revoked holding is counted in headroom only until a renewal
re-wires it, and only on a host whose shortage is displacing memory, which the host-distress
windows and the reserves already govern.

| Domain | Band | Engine behavior |
|---|---|---|
| Headroom above the planning reserve and no host distress | Normal | Claims are granted if headroom stays above the planning reserve |
| Headroom at or below the planning reserve, or host distress | Reclaim | Refuse admission, hold residency and growth in memory waits, release, and unload if it persists |
| Headroom at or below the emergency reserve, or critical kernel pressure | (still Reclaim) | The hosting service kills the worker on its first observation |

Taking a movable page another program holds displaces it: the kernel compresses it or writes it to
swap. A host that counts displacement reports it over displacement windows, contiguous spans of at
least 500 ms between that process's host samples; Seismic keeps the last 60 s of them. Compression
of other programs' data is the accepted cost of a claim. Host distress is displacement the system is
not absorbing, whoever caused it. Displacement is distress only when every window of the last 30 s
shows it:

- Swapping: pages were written to swap in each window.
- Thrashing: each window compressed at least 2,000 and decompressed at least 5,000 pages per
  second, the kernel's own rates.
- Critical: the kernel classifies memory pressure as critical (no window needed).

Displacement costs other programs speed, not the system's survival. Headroom at the planning
reserve, headroom at the emergency reserve and critical kernel pressure are the out-of-memory
backstops, and each acts on its first observation. Displacement is judged over a span longer than
the bursts the system absorbs. A load wires its weights for as long as its import runs, and on a host
whose memory is full of compressible data that wiring compresses other programs' pages, and faults
some back, for most of the import: about 10 s for a 22 GB import on an M4 Max, and 14 to 16 s at both
thrashing rates for the same import on an M1 Ultra whose memory a program had filled. Two half-second windows of that refused the load
itself. The 30 s span is about twice the longest burst measured. The same rule holds for a load, a
loaded model's growth and residency, and Reclaim and unload, whether the engine or another program
caused the displacement.

Only macOS reports displacement and kernel pressure. A host that reports neither is never in
distress, and a dedicated device's own memory has none.

Process limits are the visible ones. Inside a container the container's own cgroup limit is
visible while cgroups above it may be hidden; hidden ancestors bound nothing beyond host headroom,
so a contained process plans and claims against its own limit and host headroom.

An unavailable required observation is Blind: residency and growth wait on memory
immediately, and a continuous second of Blind is treated as Reclaim. If Reclaim persists for one
second after releases are exhausted and in-flight work completes, the engine unloads the model. An
admission attempted during Blind is refused at once with the typed `MemoryObservationUnavailable`
result; during Reclaim it is refused at once as memory pressure. Already accepted work retains its state while the engine
retries its observation. Displacement counters and the kernel pressure level are the only OS
signals the engine reads.

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

Reclaim uses steps 1–3, 5 and 8 and stops as soon as every domain is back in Normal; each victim is preempted once, and in-flight work retains its storage until physical
completion. A release that changes slab bindings needs the binding right
([state transactions](state-transactions.md)), so during a flight it runs at that flight's
completion. Removed index entries do not count as released bytes until Seismic's charge actually
falls. After unloading, the engine does not reload itself.

The engine reports its standing and typed outcomes. Its hosting service decides whether to
queue or report failed requests and when to reload an unloaded model. The service's emergency
kill is independent fault containment; it chooses nothing to release.

## Acceptance criteria

- Classified holdings sum to Seismic's device charge after every claim, release and unload.
- On a dedicated device the capacity less observed headroom differs from `held` by no more than the
  measured residual recorded below (driver state neither charged nor held), across load, tuning,
  serving bursts, long context and media.
- No new device allocation bypasses a fitting claim, and neither a rejected minimum nor a
  failed physical allocation changes accepted numerical state.
- Reclamation follows the single order and stops when the measured deficit clears.
- No engine claim leaves any used domain's headroom, or on Metal the working set's remaining bytes,
  at or below its planning reserve; a claim never unloads the model; persistent Reclaim ends in the
  typed unloaded state within the one-second bound. A backend's allocation granularity counts: a
  block allocator never adds device usage, its unallocated block space included, beyond the room
  Seismic's limit grants, so a growth that would cross the reserve is a demand deficit, not Reclaim.
- Threshold values exist in one policy definition. Host distress is derived only from Seismic's
  host observation; no other code path reads an OS pressure signal.
- On macOS every engine holding is wired from allocation to release, its residency renewed while it
  is held so it stays wired through idle gaps, and no holding is counted in host headroom; a mapped
  import window stays file-backed.
- macOS host headroom is never below the free and inactive pages of the same sample.
- A load whose host is in distress fails with `MemoryPressure` within one observation of the
  distress and returns its charge; a loaded model in persistent distress unloads within the
  one-second bound. Displacement shorter than the sustained span, such as a load's own import
  burst, refuses no claim and drives no Reclaim.
- A host that reports no displacement has the bands its headroom alone gives.
- Stable fit capacity is capacity under process limits (and the Metal working set) less the
  planning reserve; a dedicated device's capacity is the memory its driver lets the process use,
  measured on the basis its live headroom is read on and independent of other programs.
- State growth allocates exactly one claimed slab per growing domain without copying existing
  history or banks.
- Reclaim releases empty slabs and compacts referenced rows and banks into free space in held slabs
  without a new claim, including on Metal, Vulkan, CPU and CUDA.
- No history presents more spans to a launch than its domain's span limit, for any sequence of
  growth, interleaving, forks, window release, rollback and Reclaim; graphs are sealed to that limit
  and a launch beyond it is rejected before submission, never discovered by a graph lookup.
- Startup, lazy component and state-growth allocations all claim from the one device heap.
- On a unified device, a load's process footprint never exceeds what assessment charges for the
  same state: at load end, its weights, prepared resources, startup state and every host term.
  Each host term is exact from the plan or a count the plan fixes times a cost measured on the
  backend; none is a margin.

## Physical state placement

Logical state ownership and physical backing are separate authorities. The
model-state subsystem owns logical histories, recurrent banks, codec components and references.
Each store has history slabs for each of its history domains and bank slabs. A history domain is a
set of attention layers sharing one row numbering: Token (a row per token of the context),
Window(n) (a history references its last `n` rows plus tentative rows) or Shared (no storage; its
layers read a source layer's regions). A history slab contains fixed-offset regions for every
component of its domain; a bank slab contains complete recurrent banks, whose bytes follow from
their components (convolution windows, gated delta and F32 state-space states, and their tapes).
A load resolves the selected history preset into a validated `CodecSpec` per attention
binding. That descriptor owns each vector's code bitdepth, grouping, coefficient type,
width and packing; the binding's semantic activation element owns preparation rounding.
State allocation and every append/attention graph obtain plane extents from the same
state descriptor. The executor translates affine plane extents into entry dimensions
once; native consumers derive bitdepth/grouping from those dimensions and declare the
representations their implementation covers. A native capability is not permission to
change the load's representation. Tuning fixtures, resource estimates and semantic/cache
bindings carry the same descriptor rather than separate K8/V4 byte formulas.
Row numbers identify a slab of their domain and an offset within it. Every history span lies within
one slab. Rows are placed in pages: a domain's page is the smallest multiple of 256 rows for which
its span limit below is at most 63; a slab holds a whole number of pages, and a domain reserves a
whole number of pages, so every page is complete and a growth's demand in whole pages is exactly
what admission checks, provisioning backs and the claim takes. A history takes rows
only in place after its end within its last page, or as whole free pages (pages without a
referenced row), preferring the page that begins at its end; a fresh history takes the middle page
of the largest run of free pages. Every page a history references is therefore complete except its
first and last, so a history of `r` rows presents at most `ceil(r / page) + 1` spans, and a Shared
reader or a causal target reader with compressed history also sees at most `ceil(advance / page) + 1` spans of rows appended within one advance. The
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
A launch may read rows its spans do not select (a tile covering several requests reads the
union of their spans and masks the rest), so every slot of a slab table names device memory of a
whole slab: a free slot names a backed slab of the same store, and only a store without backed
slabs has none.
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
typed unload outcome, to headroom at or below the emergency reserve, or to critical
kernel pressure.
