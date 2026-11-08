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
than the wait bound `W = (P − E) / r`: the ladder releases nothing while a chain
runs, and only the one store's expert bank is cut beside it (below), so from the
planning line `P` the host may take what lies above the emergency reserve `E` at
the fastest rate `r` the memory probes measured it taking
(`MemoryPace::wait_bound`).

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
between flights alone. Before a load has measured its groups' reads, and on a
load without a bank, no measure prices a group's places or its time: it groups
every planned chunk that fits. So a plan sizes the two together: a bank beyond its
floor leaves room for the class of every chunk of a whole-context prompt, priced as a
group opening holds it (its graph slots and graphs, the target's group buffers, the
bank's placed words and the tables' stagings; `AssessmentMemoryCharge::group_class`),
and only the floor is admitted without it. That room stays the places': while no group
holds the class, the one store's bank grows into it only as borrowed slots (below),
and a group whose grant finds it borrowed takes it back first, unweighed: the bank
owes what the class lacks, which a cut beside the running chain or the shrink before
the next group gives back.

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
   first and short of preemption: the expert bank's borrowed slots, surplus state
   backing, idle state, the expert bank toward its floor, retained prefixes (least recently used first), then the
   optional components once no request is resident. A holding whose worth is not
   priced yet (every holding before the load's serving rates are, a drafter before
   its drafting is) is no cheap one: the serving settle never moves it (before the
   load is priced a settle moves nothing), and the ladder releases it only once
   nothing priced is left. Each release goes to the
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
The memory stores are read periodically on a thread of their own, every reading
interval, each reading published for the owner to take without waiting
(`MachineReading`); fresh where they are read only by claims, an admission and the
release ladder. Between readings the heap follows the process's own charge in
Seismic's ledgers, so a step reads nothing of the OS and waits on no reading. A
reading settles the serving heap only where it moved the settle's inputs: its
holdings, a request's arrival or end, the load's measures, and each store's room by
a step the settle could take (`SettleStands`). The serving rates' and the decode
price's drift between them moves no step. A settle prices the bank's steps on the
decode model, which is no step's work, so none runs where nothing moved. What a
settle reads is computed once per change of its own inputs: a decode launch once
per rows and depth, a prompt group's seconds once per size it reads and change of
their source, the bank's releases only as far as offers in their own store reach,
and a growth's room on the last reading while the process's own charge stands as it
read it (its claim still reads the machine afresh). It reads the ranking's live
decode and group-read curve as the residency thread's last feed published them,
never feeding the stack itself. Every host-RAM question reads one
measure, what the OS gives without compressing or swapping another program's data
(free, file-backed and purgeable pages): a load's admission, its band, its deficits and
its claims (`HostRoom::Load`). So another program's ask that takes it to the planning line
is answered by the load's releases, not by the OS displacing that program's data. The
plan sizes the expert bank in the stable fit, the one room its assessment settles into
too; what the moment leaves it is the running memory pace's, which settles the bank to it
as the load serves.
Growth (the expert bank's and a prompt group's places) also leaves out what the
OS displaced of other programs' data since the load began to serve. Sustained swapping
or thrashing is Reclaim at any headroom.

The one store's expert bank (unified memory) splits its slots into committed and
borrowed (`BankRoom`). Committed slots sit within that measure, beside the class room
of the planned places no group holds; borrowed slots above them, up to the places'
room and, where the OS reports displacement (macOS), what it can obtain by
compressing other programs' data, less the planning reserve (`FitMode::Borrow`). A
growth takes committed room first, and borrowed room only once the host has shown
no displacement now for `RECOVERY_STABLE_TIME`. Every release takes borrowed slots
first: the committed count only falls to the slots held. While the bank holds room
borrowed beyond allocatable memory, allocatable memory sits near nothing (the OS
keeps its free pages by compressing others), so the hold on it is not that level,
which would read Reclaim for as long as the bank holds the room and make it give it
back and take it again: it is displacement now (`platform::Serving`), swap-outs a
second above every window since the bank last grew, or the kernel's pressure at
warning. Compression alone is not: a bank holding borrowed room keeps the OS
compressing others' ageing pages. That gives all borrowed room back; the load's own claims take
back what they need of it (`DeviceHeap::take_back`), and the level's Reclaim acts on
committed slots once none is borrowed. What a growth into borrowed room displaced is
accepted as the load's own, so the line does not count it once the room is given
back. A host that reports no displacement (Linux, Windows) borrows nothing beyond the
places' room, which its level counts.
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
