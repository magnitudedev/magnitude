---
applies_to:
  - inference-v2/src/magnitude_engine/generation/**
  - inference-v2/src/magnitude_engine/models/**
  - inference-v2/src/magnitude_engine/engine/**
  - inference-v2/tests/generation/**
  - inference-v2/tests/models/**
  - inference-v2/benchmarks/**
---

# Batched speculative generation

**Requests own speculative progress; execution batches own shared computation.**
Drafting and verification can both batch, without requiring requests to accept
the same number of tokens or remain together across rounds. This is part of the
normal generation contract, including when only one request is active.

## One round, independent progress

The scheduler grants each ready request one bounded generation round and an
output allowance. The method chooses a proposal width within that allowance,
remaining context, and resource limits. The generation runtime drives the rounds
through ready model operations; batch membership is reconsidered between operations.
Round state survives scheduling yields. Each request advances when its own
dependencies are ready; verification does not wait for peers to finish drafting.
Service may pause at a safe operation boundary before the round finishes. An
output allowance bounds output and proposal width, not total execution duration.

```text
Request A: draft ─ draft ─ verify ─ accept 2 ─ reconcile ─ publish 3
Request B: draft ─ draft ─ verify ─ accept 0 ─ reconcile ─ publish 1
           └ batched ┘     └ batched ┘       independent boundaries

Request C:                plain forward ──────────────── publish 1
                          same model-operation contract
```

The example omits stop conditions. With an anchor plus `d` proposals, accepting
`k` proposals commits `k + 1` target inputs and emits those `k` proposals plus one
target-sampled token. The final emitted token normally becomes the next anchor.
Track consumed target position explicitly: emitted length cannot describe every
pipeline or stop boundary. A drafter's position may differ from the target's;
its method owns that alignment.

## Contracts and ownership

| Component | Owns | Exposes |
|---|---|---|
| Scheduler | Eligibility, output allowance, time sharing | Bounded service grants; no speculative algorithm branches |
| Generation runtime | Request rounds, target sampling, acceptance, publication | Ready model operations and completed service |
| Generation method | Proposal construction, drafter state, target-feature dependencies, catch-up | Resumable proposal and reconciliation work |
| Model executor | Compatible physical execution and resource lifetime | Per-request outputs and transactional state advances |
| State implementation | KV/recurrent layout, accepted-prefix commit, rollback or replay | Independent state resolution and any required model work |

Use **typed Python generators as local resumable computations**: yield a model
operation, receive its state advance, and eventually return a proposal or a
completed result. The runtime drives these generators on the existing execution
owner; this adds no threads, event loop, DI framework, or serialized live tasks.
Blueprints still construct the methods and executors in the worker.

A model operation identifies the live executor and sequence, inputs and
conditioning, requested outputs/features, and the minimum already-causal input
prefix. It does not identify a scheduler-specific "draft mode." Method logic
composes ordinary operations; CPU-only proposals need yield none. Neural catch-up
and state-repair forwards must also be exposed as operations, rather than hidden
inside otherwise synchronous callbacks. The dispatcher is the sole submitter of
model work and returns each operation's result to its owning continuation.

Yielding an operation transfers submission responsibility, not ownership of the
request's unresolved state. The round owns its advances through resolution or
cancellation; the executor independently retains shared execution leases through
device completion. Cancellation retires the round only after required draining
and state disposal. A suspended repair remains part of that same owned round.

The executor groups operations only when they share the live executable and
compatible state, conditioning, causal commitment, and supported input geometry.
Output demands combine: the physical call produces logits or named features
needed by any participant. A forced-token row can batch with a row that needs
sampled logits. Different output demands alone must not split neural execution.
Start with equal query widths; different widths form separate groups. Sampler,
grammar, context length, and eventual acceptance count are request properties,
not reasons to share their state or force one batch-wide decision. Executors may
require additional layout restrictions. No request waits for hypothetical future
arrivals merely to fill a batch.

Vocabulary projections are ready neural work too: compatible draft projections
share a physical call, including projections from a retained head seed. Repair
compatibility includes the storage backing the unresolved transactions; repair
must not migrate another request's active state to make a larger batch.

## State and publication boundaries

Each state advance resolves exactly once. Shared execution completion does not
commit every participant: each row commits its own accepted prefix. Rejected
attention suffixes become inaccessible. Recurrent state uses an accepted-state
snapshot or replay of exactly the accepted inputs; repair is completed before
dependent work or publication. Discarded tokens never enter committed history,
grammar state, or retained prefixes.

Native dense caches must represent independent row positions and validity;
paged caches use each row's own page/run view. Neither may truncate peers to the
minimum accepted count. Preserve useful physical storage across steps; do not
repack the complete KV history each round to hide unequal acceptance lengths.
Unsupported cache semantics require an explicit adapter, not guessed trimming.
Capacity accounting describes physical allocation granularity. A page limit and
its slab allocator must agree on reachable capacity; round reservations and limits
consistently rather than admitting tokens into an unallocatable final partial slab.
Growth and rollback images account for their transient allocation peaks before
execution. Retained logical tokens alone are not a physical-memory estimate.

The method reconciles its drafter with the accepted target features. Necessary
catch-up may be deferred if recorded explicitly and performed before that
drafter's next use. Checkpoints include that obligation, target state, method
state, and compatibility identity. Restoring target KV alone is insufficient
when a drafter requires missing hidden states; the method must reseed/replay or
decline that checkpoint.

Publication follows successful target and method reconciliation. Arrays, cache
storage, streamed weights, and other borrowed resources remain leased until
their device consumers finish. Logical completion never permits early release of
resources borrowed by peers in the same physical execution.
Failure during state reconciliation has the same completion obligation as failure
during a forward. Drain device work before releasing tentative state; if draining
fails, retain its allocations and execution leases until worker disposal. Neither
normal cleanup nor another request may reuse resources whose safety is unknown.

A one-input recurrent advance has only initial and final boundaries. Retain those
states without constructing an interior-prefix replay trace. Fully known causal
inputs likewise need no replay trace. Wider tentative advances retain the inputs
needed to reconcile each permitted accepted prefix. Pure tensor computation may
be compiled; state staging, execution leases, and acceptance remain outside the
compiled graph and execute for every invocation. Resident intermediate tensors
are live through their consumers, not artificially rooted through the entire
model execution; streamed scratch retains its explicit consumer-completion lease.

Recurrent storage owns immutable physical batch images. A request leases one row;
an unresolved advance leases both its input and destination rows. After full
commitment, execution pins retain their physical allocation charges without
retaining obsolete readable tensor graphs. Submitted consumers or still-lazy
output graphs retain their own dependencies. Compatible
complete rows borrow their existing batched tensors instead of separating and
reassembling them each step. Divergent membership may assemble a new image, while
peers retain their original rows. A partially accepted row owns its repaired image;
full acceptance can retain the verification image. Charge each complete physical
image until its last row or execution pin ends, even if only one row remains.
Retained checkpoints detach their row so prefix retention does not pin peer state.

## Cases the same contracts must handle

| Case | Required behavior |
|---|---|
| All, some, or no proposals accepted | Resolve each row independently; peers keep their accepted progress |
| No draft seed, plain decoding, or one output slot left | Use zero proposals and ordinary target advancement |
| Different widths or draft depths | Batch compatible ready operations; regroup as rows reach verification |
| Multiple drafter instances | Group by the actual executable and its contract, never a model-name string |
| Forced grammar tokens | Advance the known prefix without drafting; still update required method features |
| EOS or output limit inside a proposal | Publish only the permitted prefix; discard unreachable work and preserve the exact consumed position |
| History penalties or grammar constraints | Preview per-request state along candidates; commit only published history |
| Join, finish, cancellation, or slow consumer | Change eligibility at service boundaries; skip unsubmitted cancelled work and drain submitted work safely |
| Batch allocation cannot fit | After preparation rolls back, retire outstanding committed work before one unchanged retry; otherwise split without proposing again or resampling |
| One operation cannot fit | Resolve/discard its tentative state safely; defer or fail that request rather than silently exceed capacity |
| Device/completion failure | Fail the execution owner when safe isolation is impossible; never report speculative state as committed |

Acceptance preserves the existing **target-sampled prefix matching** mechanism:
sample the target at each candidate position, accept matching proposals up to the
first mismatch or stop, then emit that position's target sample. This requires
proposal tokens, not draft probabilities. Position-addressed RNG prevents batch
membership or rejected work from changing a seeded request's random draws.
Exact token equivalence also requires equivalent target logits; numerical changes
from physical batching must be measured separately from sampling correctness.

## Scheduling and qualification

Charge draft, verify, and repair execution to decode service. Count shared work
once, independently of per-request latency attribution. A round cannot exceed
its granted output/proposal bound. The scheduler uses method/executor cost
feedback to grant bounded work; it never needs an acceptance algorithm or a
drafter-specific branch. During contention, elapsed service budgets can suspend
unfinished rounds. These are soft deadlines: device lookahead and indivisible
operations can overrun them. Output allowances cap outstanding dependency chains;
returning control completes submitted work and retains unresolved logical state.
Uncontended service does not add deadline-induced synchronization.
Batching and speculation can compete for the same
hardware capacity, so speculative width is a method decision within the grant,
not a requirement to maximize drafts under concurrency.

Built-in speculative compositions must qualify physical drafting and target
verification batches, not merely concurrent requests or concatenated serial
outputs. Validate unequal acceptance, mixed widths, recurrent repair, seed
initialization, checkpoints, constraints, cancellation, and memory pressure
against independent request execution. Record physical batch shapes, accepted
tokens, state-copy bytes, draft/verify/repair time, throughput, and latency tails.
Single-request and batched performance are separate acceptance cases.

## Performance contract

Session count is a workload property that changes during execution. Speculation
is a composition choice. All four combinations are first-class qualification
cases, including transitions between one and many active sessions.

| Workload | Required execution behavior |
|---|---|
| One session, plain | Direct single-row execution; no batch assembly, draft work, acceptance scans, or speculative rollback images; preserve bounded device-token pipelining |
| One session, speculative | Preserve the drafter's efficient dependency chain, device-resident proposals, and one target verification per proposal block; no artificial batching delay or stage-wide synchronization |
| Multiple sessions, plain | Persistent compatible execution batches; no speculative state or draft dispatch; membership changes must not cause whole-history reconstruction every step |
| Multiple sessions, speculative | Batch compatible ready drafting and verification operations independently; per-row acceptance and repair; no global minimum acceptance, fixed cohort, or slowest-drafter barrier |

These are specializations of the same contracts, not separate engine loops.
Unused work disappears by composition or actual batch cardinality. Identifying a
batch of one must not require constructing and then undoing a batched cache.

**A scheduling yield is not a GPU fence.** Forward submission may return lazy
device results so dependent operations can be constructed and pipelined. Wait
only for actual host decisions, externally visible completion, or safe resource
reuse. Measure elapsed service at existing completion boundaries; instrumentation
must not introduce a fence per yielded operation to obtain precise timings.
Bound submitted lookahead so asynchronous efficiency cannot monopolize service.

The principal risks are extra CPU dispatch gaps, full-history KV copying,
padding across incompatible shapes, recurrent replay, and delayed resource
release increasing memory pressure. Stable-batch orchestration must scale with
active operations, not history length; state movement must be attributable to
growth, membership changes, or required repair. Storage owns immutable validated
page mappings, including their device tables. Layer reads borrow that mapping
with independently checked visibility horizons. Rebuild mappings on allocation,
truncation or relocation, rather than enumerate and revalidate the complete
history for each layer and token. Publishing an append updates only its changed
page interval. Qualify membership churn and
mixed context/proposal lengths as well as stable batches.

For each of the four cases, compare the same executor and generation algorithm
with minimal orchestration, then the integrated engine against applicable
external controls. Record launch gaps, synchronization count, copied bytes,
draft/verify/repair service, peak memory, and latency/throughput distributions.
An improvement in one case does not excuse an unexplained regression in another.

Speculation's benefit depends on acceptance, draft cost, verification width,
and concurrent load. Batching can reduce the spare compute that speculation
uses. Efficient execution of the chosen composition is required; universal
speedup from a particular drafter is not assumed. Adaptive proposal width, if
selected, belongs to the generation method and remains explicit and measurable.
