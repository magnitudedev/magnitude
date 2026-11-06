# Scheduling and capacity

**One service owner coordinates requests with one physical batch in flight.**
It schedules legal work and negotiates resource prices without knowing tensor layouts.

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
eligible again on the next round. State and retained outputs grow under device
heap claims as requests enter, and released requests return those resources.

A prompt group (a prompt's consecutive chunks run layer-major as one chain) holds
the places of the class it is granted when it opens (`ExecutorDomain::grant_group`),
up to every chunk of a whole-context prompt. The places its planned chunks are worth
are weighed against the expert bank's slots by what each saves per byte
(`GroupReads::grant`): a place saves the re-reads of the group's cold experts across
its chunks, a slot the re-reads of decode steps and of this prompt's groups that
miss it. Places that win memory from slots leave a want the bank's next shrink
serves (`PlacesWant`); until then the group holds the most places that fit what a
growth may take now (`Places::admitted`) and the heap grants. No group runs longer
than the wait bound `W = (P − E) / r`: nothing is released while a chain runs, so
from the planning line `P` the host may take what lies above the emergency reserve
`E` at the fastest rate `r` the memory probes measured it taking
(`MemoryPace::wait_bound`). Before a load has measured its groups' reads, and on a
load without a bank, no measure prices a group's places or its time: it groups
every planned chunk that fits.

A class's slots are each launch's activation with its upload regions, its output in
the chain's rotation of `n + 1` slots, and its readout and drafter slots
(`ResourceLimits::class_slots`); its group buffers are held with them, both or
neither. A load plans and holds the floor of one place; a larger class is kept
while memory is Normal and released to the floor at the ladder's surplus rung,
after any chain under way ends. Nothing grows in Reclaim, where groups run at what
is held. A request's successor banks are not a class's: a chain's launches
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
   first and short of preemption: surplus state backing, idle state, the expert bank
   toward its floor, retained prefixes (least recently used first), then the
   optional components once no request is resident. Each release goes to the
   cheapest rung with anything left, so what a costlier release frees for a cheaper
   rung is released before the next costly one. A release that a later flight
   boundary finishes (idle graph slots, released off the submitting thread; an
   expert bank resize) holds the ladder: nothing costlier runs until each such
   release has finished. The bank's memory stores resize independently, so a
   shrink waits only for a resize of its own store and a store's growth gives way
   only to a shrink its own memory needs. A shrink needs no memory: one its
   storage refuses is a broken invariant, never a rung to pass. The reclaim band
   and a new request's residency release down the same ladder.
3. Drop trailing batch members.
4. Reduce the actual legal token allowance.
5. Evict victims, the ladder's last rung, priced by exclusive releasable bytes and
   replay cost. Unloading the model follows only once every rung is exhausted.
6. Wait for a service epoch change when a peer event can improve feasibility.
7. Fail with required and available capacity when no such event remains.

Do not retry an already failed physical shape merely because its soft allowance
changed. Indivisible input spans retain their [boundary semantics](inputs.md).

Numerical storage for fresh requests is allocated during negotiated preparation,
not unconditionally during logical admission. Capacity facts come from the shared
runtime resource domain, including retained model weights, state, and scratch.
The memory stores are read fresh only by the owner's memory probes, a deficit's
retry and claims; between them the heap follows the process's own charge in
Seismic's ledgers, so a step reads nothing of the OS. A load reads host RAM in one room
from its opening through serving (`HostRoom::Obtainable`): its band, its deficits and
its claims read what the OS can make available, other programs' idle data included
where the OS counts it (macOS), so a load that compressed other programs' data to fit
serves on what it was admitted on. Growth (the expert bank's and a prompt group's
places) reads only what can be taken without displacing more of other programs' data
than was displaced as the load began to serve. Sustained swapping or thrashing is
Reclaim at any headroom, owed what that undisplacing room is below the planning line.
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
