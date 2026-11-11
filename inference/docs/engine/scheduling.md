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

The bank's speed reserve, the free room its target leaves for growth that arrives faster
than eviction delivers, is sized at the held pass: one prompt group's state growth at its
places (`in_process_growth`). A claim that takes cache in process (a group's class, a
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
streamed (`PlanError::Unpriced`), holding nothing of the device. Where they are
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

While a chain runs the ladder defers its releases to the chain's end, so pressure
that arrives during a group waits at most that group's run. A group's length is
at most the arena's places. Within it, only a scratch-served streamed chain reads
memory between its submissions, and only to cut the bank in the one store beside
it (`ExpertBank::release_running`, below); a discrete host's host tier and the
state rungs wait the group out.

A prompt group's chain is encoded at the device's pace, and between its submissions
it takes the periodic reading published since the last (`ExecutorDomain::submit_chain`). Where the
expert bank lies in the one store (unified memory) and that reading is short, the
store is cut beside the running chain (`ExpertBank::release_running`): the watcher
drops the records past the cut, moving nothing, and unmaps their experts from the
chain's first step, except those its requests already routed at a layer whose hits
it decided, which stay readable to the chain's later passes there. The storage
installs once those passes are submitted, so every run submitted after resolves only
records below the cut, and its release waits on the device work before it. So
another program's ask that lands mid-chain is answered within the passes the watcher
decided ahead of the encoding, not at the chain's end, and the next boundary reports
it as any bank shrink. A host tier's shrink and a device bank's moves still run
between flights alone.

A class's slots are each launch's activation with its upload regions, its output in
the chain's rotation of `n + 1` slots, and its readout and drafter slots
(`ResourceLimits::class_slots`); its group buffers are held with them, both or
neither. The arena's class never grows or shrinks while the load serves; the
ladder's surplus rung releases only idle slots beyond it. A request's successor banks are not a class's: a chain's launches
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
   the ladder.
2. Release down the owner's one ladder (`scheduler/src/release.rs`), cheapest loss
   first and short of preemption: surplus state backing and idle graph slots, idle
   state, retained prefixes (least recently used first), then the optional
   components once no request is resident. The expert bank is not on it: it is cache
   on the ledger, which yields to a claim at once. Each release goes to the cheapest
   rung with anything left, so what a costlier release frees for a cheaper rung is
   released before the next costly one. A release that a later flight boundary
   finishes (idle graph slots, released off the submitting thread) holds the ladder:
   nothing costlier runs until it has finished.
3. Drop trailing batch members.
4. Reduce the actual legal token allowance.
5. Evict victims, the ladder's last rung, priced by exclusive releasable bytes and
   replay cost. Unloading the model is no rung: it happens only at a reading where
   the speed guarantee cannot be kept.
6. Wait for a service epoch change when a peer event can improve feasibility.
7. Fail with required and available capacity when no such event remains.

Do not retry an already failed physical shape merely because its soft allowance
changed. Indivisible input spans retain their [boundary semantics](inputs.md).

Numerical storage for fresh requests is allocated during negotiated preparation,
not unconditionally during logical admission. Each store has one memory ledger
(`memory::ledger`), the one authority on room. What the process holds while it
serves is its entries: committed (the arena, the expert bank's floor, state, landing
rows, a measurement's room) or cache (the expert bank's slots past its floor). An
allocation a load makes as it starts (its weights among them) is admitted against
the ledger's reading and is then memory no entry names, which later readings count
with other programs'. A load's allocations (a claim of kind Load, Arena, the bank's
floor, a landing, an upload, a component, a qualification or the state floor) are the user's own action:
they also count the clean file-backed pages the OS drops once (on macOS
`external_page_count`; Linux's `MemAvailable` and Windows's available memory already
count their file cache), and a claim holds them, so the next claim finds them held.
Every claim made while serving, the bank's growth and its backing among them, counts
only the free memory above the floor, which `seismic::Reclaim` states at every call
that reads host RAM's room. A load is admitted for what it needs to serve: its
weights, its bank's floor and one request's minimum state (the state floor, held for
the domain's life); a sequence's state is held first by that floor (committed state up
to it counts once), so a request's first rows draw from room the load already holds.
Its one OS input is each store's available memory, what the OS gives without
compressing or swapping another program's data (on macOS free − speculative +
purgeable pages above the free-page target, as macOS refills free pages to that
target by reclaiming and reclaims file-backed pages only beside idle programs'
anonymous pages; on Linux `MemAvailable` within cgroup limits, as Linux reclaims file
cache within its watermarks before it swaps; a discrete device's free memory), within
a driver's grant (Metal's working set). Memory the process allocates is resident as
the ledger records it (a bank's slot is recorded as its storage is backed), so a
reading never counts the process's own growth as free room or as others'.
What the process may hold is that reading less other programs' memory; a claim
admits against the free room and the cache, which yields to it at once, and Seismic's
store bounds are read from the ledgers at each admission against the store, never
stored, so Seismic refuses no allocation the ledger admits. The stores are read periodically on a thread of their
own, every reading interval, each reading published for the owner to take without
waiting (`MachineReading`).

The expert bank's size is derived, never stored: at every reading each store's room
for the bank's cache is the budget less committed memory, other cache and the speed
guarantee (one prompt group's mandatory growth, plus other programs' rise between the
last two readings), and the bank follows that target, a shrink before any growth,
growing by whole blocks (`experts::derived`). A load imports its bank at the target
its import's reading derives, so it imports what it keeps. The model unloads only at
a reading where free room stays below the guarantee with no cache left and nothing
to preempt.
Allocation denial keeps its typed required/available byte facts through model
preparation; native failures without capacity evidence are not reclassified by
matching diagnostic text. Reclamation and eviction report the observed decrease
in charged bytes, including aliases and retained checkpoints, and beside it the
device store's claim (its allocations, or what its backend holds where that is
more).

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
