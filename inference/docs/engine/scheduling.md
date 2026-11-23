# Scheduling and capacity

**One service owner coordinates requests with one physical batch in flight.**
It schedules legal work and negotiates resource prices without knowing tensor layouts.

## Prompt units

| Unit | What it is | What sets it |
| --- | --- | --- |
| Chunk (launch) | One prefill launch: at most the launch class's rows of one prompt | Fixed at load by the plan's launch classes |
| Prompt group | A prompt's consecutive chunks run layer-major as one chain; its length is the chunks it holds | At each group open: `min(chunks left, chain capacity)` places, as many as yield at once |
| Places | The chunks a group's chain holds at once, each with the class's graph slots and buffers | Claimed per group as it opens (`ExecutorDomain::open_group`), from free room and what yields at once, never preempting a sequence; freed as it closes, its slots idling as cache until released. A load holds one place from load |
| Pass size (`PassLaunches`) | The launches one merged-expert pass covers; its rows are launches × the launch region's rows | Fixed at load: the plan's pass, capped by the workspace arena the launches' graphs fix (`group_pass_rows`), the one answer the plan charges and the load holds. A chain longer than one pass runs over as many passes as it needs |

The bank's speed guarantee, the free room its target leaves for growth that arrives faster
than eviction delivers, is one prompt group's state growth at the held pass's places
(`in_process_bytes`, the load's one in-process figure) and others' rise since the last
reading. A claim that takes cache in process (a group's class, a
sequence's state) allocates those bytes only as the cache's holder frees them: the bank
as its shrink releases them, idle graph slots as their release ends. Until then the bytes
stay the process's, and the claimant waits.

A streamed bank's pass is chosen by price at load (`ExecutionPlanDraft::settle_bank`).
The pass's buffers (its merged experts and row table, shared by every group) are held
from load beside the bank, so a larger pass leaves fewer slots. Each pass size of the
ladder, one launch then doubling up to the whole chain's places, is priced analytically:
its buffers beside the smallest pass's, and the most slots the room leaves after them.
The plan holds the size whose bank serves the workload prior in the least time a second:
decode steps a second times a step's seconds at its slots, plus requests a second times a
whole-context prompt's routed seconds in passes of that size. Sizes priced alike go to
the larger. Only the chosen size is certified, and it certifies to the bytes it was
priced at. The plan, its certified target family and the graphs bound to its pass buffers
all take that one pass (`ModelLoadPlan::pass`). A bank held whole prices nothing: its
pass is the largest whose arena fits beside it.

Prices come from the machine's generic measurements and the model's headers alone
(`PlanPrices`). While the measurements are pending, every load waits for them, whole or
streamed (`PlanError::Pending`), holding nothing of the device. Where they are
unavailable (no kernel cache), a whole load plans without them and a streamed load fails
as an internal error.

## Ownership and progress

| Owner | Responsibility |
| --- | --- |
| Service | Admission, phase selection, batch membership, capacity policy, and request outcomes |
| Generation | Next legal proposal under an allowance and current output credit |
| Model executor | Batch preparation, capacity requirements, and reclaimable resources |
| Execution owner | Serialized live state access and completion reconciliation |

Request status follows facts: queued, runnable, awaiting completion, output-blocked,
preempted, capacity-blocked, or terminal. Idle service waits for events. Completion
notifications have reserved delivery capacity and cannot be starved by control jobs.
Only reconciliation of the outstanding completion permits the next submission.

The host constructs live execution objects on their owning thread. Cloneable
clients carry only control capabilities and immutable results. Outstanding
control calls remain owned through result consumption, including completed but
unclaimed results. Completion and abandoned-result cleanup have reserved delivery
and run before queued controls. Abandoned preparation results are released on the
owner; disconnect retirement waits for any shared submission to reconcile.
Shutdown rejects queued calls, cancels live requests, waits for completion, and
releases the owner's resource domain before the thread exits.

Prepared-input admission binds artifact/tokenizer identity and grammar before
opening numerical state. Model-specific sources cross the worker boundary as host
data; the model supplies their semantic layout before generation binding. Source
installation validates its correspondence with the bound prompt and layout and is
atomic on failure. Device allocation remains part of negotiated preparation.
Abandoned successful admissions release their sources on the owner through reserved
cleanup, including when numerical work has already been submitted.
Publication receivers own bounded accepted output.
Owner shutdown transfers retained output to host-owned terminal publication so
receivers can drain it after live numerical objects are released. Terminal status
is published only after that output has drained; receiver abandonment explicitly
discards it through reserved request cleanup.

## Fairness

- Under simultaneous prefill/decode demand, decode runs first.
- Prefill accrues decode debt according to the configured service share; completed
  decode repays that debt using elapsed time.
- Cost spans preparation through observed completion. Emitted token count is not a clock.
- Without contention, the eligible phase runs freely; contention debt resets.
- Within a phase, waiting age and locality credit rank work, followed by accumulated
  service. Residency and preemption debt contribute to locality.
- Forced-token runs remain decode work. Original encoder conditioning is prefill
  work and returns to shared scheduling at completion.

There is no configured request or batch count. A phase advances requests in
priority order within its aggregate token allowance; requests left over are
eligible again on the next round. State and retained outputs grow under claims on
the memory ledger as requests enter, and released requests return those resources.

A prompt group (a prompt's consecutive chunks run layer-major as one chain) runs
within the places of its own class, claimed as the group opens
(`ExecutorDomain::open_group`): a place per chunk behind its first, within the load's
chain capacity, with their graph slots, the target's group buffers, the
bank's placed words and the tables' stagings. The claim is optional: it takes only
what yields at once (free room, spare, bank slots above the floor), never preempts a
sequence and is never refused, and a group it cannot cover runs with one place. The
class is freed as the group's chain closes. The load commits a group of one place;
nothing is set aside for places beyond it, none is reserved against the bank, and
nothing settles them.

While a chain runs, cache yields wait for the chain's end, so pressure that arrives
during a group waits at most that group's run. A group's length is
at most the arena's places. Within it, only a scratch-served streamed chain reads
memory between its submissions, and only to cut the bank in the one store beside
it (`ExpertBank::release_running`, below); a discrete host's host tier and state
wait the group out.

A prompt group's chain is encoded at the device's pace, and between its submissions
it reads memory only when an OS pressure event has come since the last reading
(`ExecutorDomain::submit_chain`). Where the expert bank lies in the one store (unified
memory) and the shortage takes some of the bank's cache, the
store is cut beside the running chain (`ExpertBank::release_running`): the watcher
drops the records past the cut, moving nothing, and unmaps their experts from the
chain's first step, except those its requests already routed at a layer whose hits
it decided, which stay readable to the chain's later passes there. The storage
installs once those passes are submitted, so every run submitted after resolves only
records below the cut, and its release waits on the device work before it. So
another program's ask that lands mid-chain is answered within the passes the watcher
decided ahead of the encoding, not at the chain's end, and the next boundary reports
it as any bank shrink. A host tier's shrink still runs between flights alone.

A class's slots are each launch's activation with its upload regions, its output in
the chain's rotation of `n + 1` slots, and its readout and drafter slots
(`ResourceLimits::class_slots`); its group buffers are held with them, both or
neither. The arena's class never grows or shrinks while the load serves; only idle
slots beyond it are cache (`CacheKind::GraphSlots`) and are released. A request's successor banks are not a class's: a chain's launches
alternate two whatever its length (`ResourceLimits::target_successor_banks`), and
they are granted when the request opens. While a chain is under way only its holder
claims work (`ExecutorDomain::claims`).

Input-preparation rows publish conditioning independently of generation proposals.
Their completion cannot consume tokens, select a token, or change generation usage.
Preparation and decoder consumption form separate batches at the same prefill
priority. Pending preparation prevents checkpoint capture and retains its resources
through cancellation until completion; cancelled results are never published.
Capacity negotiation may reduce preparation batch membership, but changing a
decoder token allowance cannot change an encoder preparation unit.

## Capacity negotiation

Apply these steps in order when preparation cannot fit:

1. Within the workload state the plan charges, give a group the rows and banks of
   the least recently used retained prefixes outside the latest conversation's
   checkpoints and any submitted use, before the stores grow past it
   (`DomainError::StateCeiling`, `PrefixCache::release_unspared`). This runs before
   cache yields.
2. Yield cache in rank order, lowest value per byte first and at once: spare granules
   and idle graph slots (worth nothing), then retained prefixes, an idle drafter or
   vision encoder and the expert bank's slots above its floor, each by the seconds
   serving it saves per byte held (no time decay). The expert bank yields to a claim
   like any cache. A holder frees what was taken of it as its release ends, and the
   claim allocates those bytes only then.
3. Drop trailing batch members.
4. Reduce the actual legal token allowance.
5. Evict victims, priced by exclusive releasable bytes and replay cost. Unloading the
   model is no step of this order: it happens only on a shortage that persists
   (below).
6. Wait for a service epoch change when a peer event can improve feasibility.
7. Fail with required and available capacity when no such event remains.

Do not retry an already failed physical shape merely because its soft allowance
changed. Indivisible input spans retain their [boundary semantics](inputs.md).

Numerical storage for fresh requests is allocated during negotiated preparation,
not unconditionally during logical admission.

Each store has one memory ledger (`memory::ledger`): the one record of what the process
holds there and the one authority on room. Every byte the process holds is an entry,
committed (a sequence's state, the state floor, the arena, the expert bank's floor,
landing rows, a qualification's or an upload's room, an optional component, retained
outputs, a measurement's room, a group's scratch and classes) or cache (spare granules,
idle graph slots, the bank's slots above its floor, a retained prefix, an idle drafter,
an idle vision encoder). An allocation is backed by an entry or is not made: Seismic
allocates only through the entries and stores the ledger gives it, block slack and
driver objects included, so every byte the process holds is counted once and none
reads as another program's. The ledger computes no size: claims are sized by the page
rule (`magnitude_state::state_bytes`) and the assessment's formulas, the same
functions the plan predicts with.

The bank's records and the KV slabs live in one granule pool per store (`seismic::memory`):
blocks of the backend's plain allocator, fixed-size granules inside, holes allowed and
nothing moved. A claim takes a free granule in process; only growth allocates. A
released granule is claimable once the last flight that read it completes. A block
returns to the OS once every granule in it is free; to empty one for outside demand its
live granules are copied into free granules elsewhere, the pool's only copy. Each block
is charged once, at its exact bytes, to the pool's entry: bank slots and KV slabs are
holdings over granules, and handing a granule from one to the other moves what is held,
never what is allocated. Bytes freed for outside demand are owed to the OS until
returned blocks pay them or claims take freed granules back into use.

A reading is one sample of a store's OS at one instant. The ledger journals its own
changes and pairs a reading with what the process had allocated as the sample shows it,
leaving out each change made after the sample or within the store's lag before it. Each
change spans one OS effect of the process's and every record of its bytes, so no sample
stands beside one and a record outside a change is an invariant violation. So
others' memory (`others`) is what the store uses beyond the supply and the process's own
bytes in it, and the process's own action never moves it. Bytes the OS holds elsewhere
for the process (compressed or swapped, `nonresident`) are its own but in no figure of
the store's use. Resident bytes the ledger holds that the use does not (`unobserved`)
are counted and logged where the OS's count is approximate or the store is a device's,
and are an accounting violation, counted apart, where the OS's count is exact. A reading
that cannot be made is an invariant violation naming the figure.

Supply is what the OS gives without writing anyone's dirty data: free pages above the
OS's own free target, clean file cache, and the cold anonymous pages that compressing or
swapping frees, bounded by where they can go. The process's own cold pages are never
supply. Others' working set is read from harm, the pages others bring back after the OS
reclaimed them, counted in one snapshot with the process's own share so its own paging
never reads as theirs, and judged against a learned baseline over spans of at least the
counters' lag. Harm above the baseline lowers a boundary B on what the process may hold,
and growth past B needs a real change in others. Each new OS pressure event, delivered
by one watch per process (the macOS memory-pressure source, a Linux PSI trigger,
Windows' low-memory notification), lowers B by one pool block while the level is set.
Where the OS counts no share of the process's own, harm cannot be judged and growth is
bounded by supply alone; where it delivers no event there is no pressure step. On unified Metal the grant is further
bounded by the wired ceiling: the wired limit less others' wired bytes and Metal's execution reserve.

The budget is `min(grant, in_use + supply)`, `in_use` the process's own bytes in RAM.
Free room is the budget less everything held, and a claim admits against it, with cache
yielding to it at once. A load is admitted for what it needs to serve: its weights, its
bank's floor and one request's minimum state (the state floor, held for the domain's
life); a sequence's state is held first by that floor, so a request's first rows draw
from room the load already holds. Seismic's store bounds are read from the ledgers at
each admission, never stored, so Seismic refuses no allocation the ledger admits.

The OS is read at each flight boundary, each completion of a change of the process's
own (a resize, a slot release, a measured load priced) and each OS pressure event.
Nothing on an OS signals memory another program frees, so while a shortage, a wait or a
held-back growth stands and no flight is in progress, the worker reads again on an
interval that starts at what the reading took and doubles while readings change
nothing, never past the shortest interval across which others' memory was seen to move
in the store. With nothing standing no timed reading is owed. No period, window or
margin is set.

A load reads its device's stores once, with no change of the process's in flight (a
sample waits for the steps under way when it began and for the entry changes beside it,
each one native allocation or free, which never wait for it: a held-back step is
bounded, an entry change cannot be held back without a deadlock against the lock its
reconcile takes; the load's own open change does not count while it waits), and
holds that reading from its search to its last claim: its claims draw on that reading's
room with the holdings and claims made since counted (`DeviceReading::room_after`), so
no claim takes a fresh sample and a concurrent resize, load or pressure event cannot
invalidate it. The plan, the host tier's floor included, is sized from that same reading
(`PlanningRoom`), with the same supply, ceiling and in-process figure, so the room the
plan sees is what the ledger grants; the assessment's stable fit sizes no load. The in-process figure is derived once per load, from its placed draft, its state
plan and the pass its plan holds, and its target graphs are prepared for that pass.

The expert bank's size is derived, never stored: at every reading each store's room for
the bank's cache is the budget less committed memory, other cache and the speed
guarantee, and the bank follows that target, a shrink before any growth, growing by
whole blocks (`experts::derived`). A load imports its bank at the target its import's
reading derives, so it imports what it keeps. Steady decode moves no term, so it makes
no resize.

A shortage is real when the budget cannot hold what is committed and the guarantee,
when others' harm rises with no growth of the process's within the counters' lag, or
when a new pressure event arrives with the OS's level set. Each reading's shortage is
answered once, in order, each step only where the one before did not meet it: cache
yields in rank order at once; new work waits while the bank sits at its floor; named
sequences are preempted, the lowest value first and just enough; the model unloads, only
on a shortage that holds across readings spanning the store's lag with no change of the
process's pending, nothing left to yield or preempt and a sequence admitted. An idle
load never unloads: its first new sequence is refused instead.

Allocation denial keeps its typed required/available byte facts through model
preparation; native failures without capacity evidence are not reclassified by
matching diagnostic text. Reclamation and eviction report the observed decrease
in charged bytes, including aliases and retained checkpoints.

Victim ordering favors output-blocked requests, then lower preemption debt, then
more exclusive bytes per replayed input, then lower accumulated service. Protected
recovery prevents immediate repeated eviction. Eviction succeeds only when charged
capacity actually decreases.

## Completion and failure

- Admission, completion, publication, cancellation, and retirement can advance the
  capacity epoch; blocked work retries only after relevant change.
- A batch has one physical cost; request attribution retains preparation and phase
  distinctions without claiming individual device timings.
- Per-request selection failures preserve accepted peer progress.
- Fatal execution-owner failure reaches all affected live requests and queued callers.
- Cancellation and disconnect stop future progress; submitted work remains owned
  until completion, and queued output has an explicit drain or discard owner.
