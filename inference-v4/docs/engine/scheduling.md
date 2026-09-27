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
control calls are bounded through result consumption, including completed but
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

Input-preparation rows publish conditioning independently of generation proposals.
Their completion cannot consume tokens, select a token, or change generation usage.
Preparation and decoder consumption form separate batches at the same prefill
priority. Pending preparation prevents checkpoint capture and retains its resources
through cancellation until completion; cancelled results are never published.
Capacity negotiation may reduce preparation batch membership, but changing a
decoder token allowance cannot change an encoder preparation unit.

## Capacity negotiation

Apply these steps in order when preparation cannot fit:

1. Reclaim idle resources and unclaimed temporary capacity.
2. Drop trailing batch members.
3. Reduce the actual legal token allowance.
4. Evict victims priced by exclusive releasable bytes and replay cost.
5. Wait for a service epoch change when a peer event can improve feasibility.
6. Fail with required and available capacity when no such event remains.

Do not retry an already failed physical shape merely because its soft allowance
changed. Indivisible input spans retain their [boundary semantics](inputs.md).

Numerical storage for fresh requests is allocated during negotiated preparation,
not unconditionally during logical admission. Capacity facts come from the shared
runtime resource domain, including retained model weights, state, and scratch.
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
