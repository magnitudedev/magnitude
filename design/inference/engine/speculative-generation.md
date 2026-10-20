---
applies_to:
  - inference/engine/generation/**
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

Rounds are **local resumable computations**: a round yields a model operation,
receives its state advance, and eventually returns a proposal or a completed
result. The runtime drives rounds on the existing execution owner; this adds no
threads, event loop, DI framework, or serialized live tasks. The worker
constructs the methods and executors.

A model operation identifies the live executor and sequence, inputs and
conditioning, requested outputs/features, and the minimum already-causal input
prefix. It does not identify a scheduler-specific "draft mode." Method logic
composes ordinary operations; CPU-only proposals need yield none. Neural catch-up
and state-repair forwards must also be exposed as operations, rather than hidden
inside otherwise synchronous callbacks. The dispatcher is the sole submitter of
model work and returns each operation's result to its owning continuation.

A target-conditioned drafter enters prompt history as part of each prompt chunk's
operation. A successor-dependent head pairs each target feature with the token
following it, entering only pairs whose tokens lie inside the consumed prefix.
At target position N its head is at N−1 with the final feature still open. The
executor retains that feature in one independently charged device row; generation
records the obligation without owning its device lifetime. The next chunk enters
that row with its first token, followed by the chunk's complete pairs, as one
ordered device entry. A first one-token chunk captures the open row without a head
advance. A feature-only separate draft can enter all rows immediately.

A head that reads the target's history (an assistant drafter) keeps none and is
never primed. Its position is the target's, which only committed target rows
move. Only the chunk closing the prompt (the one selecting or forcing the first
token) demands its final target feature; each verification supplies its last
accepted row's. A draft is one transaction of the anchor alone at the anchor's
target position, conditioned on the feature that selected it; the executor
chains proposals over the target's committed history and writes nothing. Its
checkpoint is that position alone: a resumed request always computes at least
its closing row. Replay enters nothing, so the replayed anchor's round decodes
plainly and its feature resumes drafting.

Media rows enter the target from the projector under placeholder tokens. A
drafter whose history embeds every prompt token cannot represent them, and
generation refuses that drafter for a request with media spans rather than
embedding placeholders. A reading head sees media rows only through the
target's history; a feature-only separate draft sees only their target features.

The executor queues entry directly behind its target chunk and commits them
together. Queued lookahead uses its predecessor's new boundary row, including
before the host observes that predecessor. No continuing prompt chunk requires a
host feature read or retains a whole graph-output arena in a prefix. Host method
entries remain for replay and rows whose successor is sampled. Exporting a
checkpoint removes the sampled successor from the final pending pair, retaining
its open feature so a restored request supplies its own successor token.

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

A request eligible for prefix reuse keeps each intermediate drafter history
complete for a continuation at that boundary. A later peer may introduce a
branch after prefill has started, so the original request's eventual draft
position cannot justify omitting history from a reusable prefix. Requests that
do not contribute reusable prefixes may omit window rows their own first draft
cannot read. The scheduler declares reuse eligibility before residency;
generation carries that constraint into device and transaction priming alike.

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
| Exact batch requirements cannot be reserved | Before submission, service retires eligible retention or preempts an eligible live request; otherwise it queues, splits, or rejects the unsubmitted intent without proposing again or resampling |
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


## Shared draft budgeting

A round's draft budget is a pair: the proposals the drafter generates and the
prefix of them offered to target verification. The verified count cannot exceed
the generated count. Counts exclude the anchor and the target's bonus token. MTP
and separate drafters share one budget mechanism; the method implements proposal
execution and state reconciliation, and the scheduler reports launch durations
without branching on the drafting algorithm.

A load's draft policy is fixed or adaptive. A drafter given neither a width nor
a budget is adaptive over every width it has: all a separate drafter's block
holds, or all a draft head chains. An explicit width alone is that fixed width.

### Fixed

Every round takes one configured budget. A round whose grant cannot hold the
generated count runs plain; it does not invent a smaller block. The load prepares
exactly that generated count.

### Adaptive

Each round's budget is chosen from measurements of the load's own rounds.

**Widths.** Launch rows round up to a row class, so a verification that does not
fill its class pays for rows it does not use. The candidate widths are those whose
verification (the proposals and their anchor) fills a class, the method's widest
width, and none. Both counts of a budget are taken from them. The load prepares
every candidate generated count before readiness and counts them in assessment and
memory admission; the tuner's workload lists their launches as alternatives of
one method. A round near the end of a request takes the widest candidate that
fits the tokens it may still produce.

**Acceptance.** Acceptance is equality with the target's selection, and the
selection at a position is a function of the request's seed, the position and the
logits. So the proposals a full verification would have accepted are those that
agree with the tokens the request goes on to commit. Every drafted proposal is
scored against them, verified or not, at no device cost, under sampling as under
greedy selection. The controller keeps, per request, how often a block is right
through each depth: the mean of the depth's recent blocks, never above a
shallower depth's. A request starts from the load's latest values.

A block's proposals depend on how many were generated, because bidirectional
drafter layers see the whole block. Acceptance is nevertheless kept by depth
alone, and corrected by the rounds that follow any change of generated count.

**Costs.** The load keeps the measured durations of each drafter launch by
generated count and each target launch by rows, per context class and per number
of requests sharing the launch. Nothing is forecast. A launch costs the median of
its latest few durations, so a launch another process held up once prices
nothing; durations older than a fixed number of the load's rounds are dropped when
the launch is measured again. A launch not yet measured costs at least what a
smaller one measured, and at least what the same launch measured at a shorter
context; a larger launch proves nothing about a smaller one. With nothing
measured every width costs the same. The machine's speed moves every launch
alike, so each plain launch (one target row) scales every other measurement of
its context class and request count with its cost: a launch no longer taken
keeps its cost relative to a plain decode, not the seconds it took on a machine
another process made busier or freer, until it is measured again. The plain
decode is the one launch no look takes, so once measured in a context class it is
taken in place of drafting while its cost rests on fewer than its full share of
durations, or on durations as old as that fixed number of rounds: a plain cost
left high by one held-up decode or a since-freed machine would otherwise make
drafting that loses look faster for good.

**Choice.** Expected tokens are one plus the acceptance of each verified depth;
a round's time is its two launches. Three rules:

1. The round takes the budget committing the most tokens per second. Once its
   block is drafted, it verifies the prefix that does so with the drafter's
   launch already paid. Of equal rates the narrowest wins.
2. A width that stops being the fastest is still generated for a while, with
   verification following rule 1. Its depths stay scored, so the text supporting
   it again is seen in the next round. Each round a width is the fastest earns it
   one such round, up to the rounds a look would take to return, and a kept round
   spends more of them the less of the rate is at stake.
3. A look is a round generating wider than rules 1 and 2 would. It is an ordinary
   round that loses a known time to the fastest budget. It goes to the wider
   width with the most to gain for that time, where the gain is that width's rate
   if every proposal were right. Looks take a fixed share of decode time, scaled
   down where little can be gained. The time set aside for looks belongs to the
   load and carries from request to request, and a round with nothing wider to
   look at keeps it, so drafting that stopped paying (as on a CPU, where it
   usually costs more than it saves) is still tried again however short the
   load's requests are. A look is also taken on a run of rounds that
   accepted everything they verified, which is what text that became predictable
   looks like from a narrow width. The run that earns one halves when such a look
   finds its block right and doubles when it finds it wrong, and the load carries
   it from one request to the next.

The controller runs no device work solely to learn: every round it causes
commits tokens. An unmeasured launch is tried because its bound makes it the
fastest; a deeper depth is scored because a look or a kept width drafted it.

**Shared launches.** Requests whose drafted blocks are verified in one launch
share its rows: the launch takes the row class committing the most tokens per
second, and its rows go to the proposals most often right, whichever requests
drafted them. Each request then verifies no more than its share. A drafter launch
batches equal blocks only, so requests generating different counts draft in
separate launches.

**Guarantees and limits.**

- On text that one width serves throughout, decode is within the looks' share of
  that width's rate once its launches are measured.
- On text that changes, verification follows within the few rounds the depth
  means take to move, and generation within the rounds a width is kept or a look
  takes to come. Text that turns predictable under a narrow width brings looks
  round after round while they keep finding their blocks right; this is what
  returns a chained head, whose deeper proposals each cost a pass and so stop
  being drafted, to its wider widths.
- Measuring a load's launches costs one round per width per context class, and the
  plain decode up to its share of rounds per context class and per stale period.
- The means, the kept rounds and the looks' share are fixed constants, not
  statistical guarantees. Acceptance that alternates faster than the means move
  is not followed.
