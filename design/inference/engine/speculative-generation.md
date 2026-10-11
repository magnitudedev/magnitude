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

MTP and external drafters use one request-owned controller over load-owned
evidence. Its action is a pair:
the number of proposals physically generated and the prefix offered to target
verification. The verified count cannot exceed the generated count. Counts exclude
the anchor and target bonus token. The method implements proposal execution and
state reconciliation; the scheduler reports completion evidence without branching
on the drafting algorithm.

Fixed budgets are the default. Explicit adaptive policies name a bounded set of
admitted actions and an initial action. When no configured action fits a round
grant, the method runs a plain round; it does not invent a smaller physical block. The executor prepares the required physical
shapes before readiness and includes them in assessment and memory admission.
Changing only verification length does not create another draft graph. Unequal
noncausal block widths cannot batch through padding: a wider block may change every
proposal. Metadata defines attention and layout semantics, not the profitable width.

The controller learns from ordinary generation. It never runs device work,
fences or transfers solely to learn; it takes another configured action for one
round only when that round's value of information exceeds its expected cost (see
below). It reuses each completed target matching prefix as conditional
match evidence. A first mismatch observes one conditional failure; later suffix
positions are unobserved. An all-matching offered prefix right-censors its unseen
tail. Stop, terminal, forced, and invalid observations do not train continuation.
Shortened offers can contribute their actually observed prefix without labeling
the original action's full service cost as measured. Under deterministic (greedy,
unshaped) selection, acceptance is equality with the selected token, so proposals a
fully matched offer generated but did not verify are compared after the fact with
the target's committed continuation. That completes the round's outcome for every
wider verification of the same block at no device cost. Under sampling the
comparison is not acceptance and contributes nothing.

For separate block drafters, evidence is keyed by the exact physical generated
count. Every verification prefix of that same block shares its proposal sequence;
changing generated count may change its first proposal. MTP explicitly permits
chained-prefix evidence across its generated counts, because later head passes
consume earlier selections and do not feed backward. Causal GGUF flags alone never
authorize physical-block pooling. Target-sampling and shape-dependent numerical
agreement remain separate qualification obligations.

Acceptance evidence is one distribution of first-mismatch positions and
right-censored matching prefixes, owned by the loaded method and shared by all of
its requests. It pools without forgetting while acceptance is steady. A two-sided
e-CUSUM bets on each observed conditional match against the pooled posterior mean at
its depth; while match probabilities are unchanged its false-alarm run length is at
least the configured number of trials. When it alarms, evidence before the
detector's restart is stale and is discarded, so pooling stays long in a steady
state and reacts within a few rounds to a shift. Conditional
match probabilities for independent block comparisons are derived from that same
distribution. An early mismatch determines the accepted-prefix outcome for every
longer offer without pretending that later conditional matches were observed.
Short all-matching offers leave their unseen suffix unresolved. The paired test
uses only offers that resolve the wider action: offer length is fixed before the
outcome, so this selection is unbiased. Censored shorter offers (typically
grant-clipped) are not in the paired test: the test needs each outcome's whole
utility, and scoring the unseen suffix as its worst case would bias against
narrowing; their observed prefix still informs conditional match evidence.

The controller compares expected committed-token rewards against observed
component-service forecasts. Confidence dominance can promote an action; when
uncertainty remains, the same decision compares the information from an ordinary
candidate completion with that from an ordinary incumbent completion. Missing
coverage is priced only by the existing ordinary-round rule below. For shared-prefix actions, conservative bounds on
the narrower action's cost and paired cost difference give a sufficient linear
utility of the accepted-prefix outcome for the ratio of expectations. The
confidence test uses those paired utility observations directly, preserving
reward covariance rather than multiplying independent conditional-depth bounds.
It never substitutes the mean of outcome-wise reward/cost products for the
ratio-of-expectations objective. The paired test uses only observations whose
offered prefix covers the wider action. Offer length is fixed before the outcome,
so this selection is unbiased; a shorter, typically grant-clipped, offer neither
resolves the comparison nor counts as its worst suffix, while its observed prefix
still informs conditional match evidence. A right-censored observation contributes the
least favorable utility of every still-possible suffix.

A fixed mixture of bounded likelihood bets tests positive expected utility.
Its one-sided nominal threshold is 2.5%; stakes are fixed before observing data,
not selected by their apparent success. Evidence discarded by a detected change
leaves the test at neutral. The
fixed-utility, independent stationary null justifies that single-look threshold;
adaptive cost forecasts, repeated decisions and arbitrary drift have no claimed
confidence guarantee. This remains advisory estimation and requires
runtime qualification, including changes in acceptance regime and censoring.

Physical operation prices are grouped by actual identity before lookup. Their
coefficients are derived from the complete conditional expected-reward and
expected-cost contrast, retaining shared-prefix dependence. Only an exactly zero
aggregate coefficient removes a price obligation. Identical work can cancel in a
cost difference while still carrying a nonzero coefficient in a throughput
contrast: the common host residual, for example, remains multiplied by the
expected reward difference. Head identity requires the same physical entry class,
start position and generated width; target start positions remain separate
observations. Equal numerical estimates alone do not establish shared identity.

A load-owned cost model prices each prepared target row class and head
(entry rows, generated count, context) class. It does not assume verification time
is linear in rows. Unobserved component classes or unsupported context coverage
are explicitly unknown. No hot graph traversal, compilation, device operation, or new
timing measurement is permitted. Ordinary isolated component completions
fill a load-owned table keyed by actual dispatched row class, exact generated
count, and context. These physical observations remain useful after terminal or
grant-clipped work, even when continuation acceptance is censored. Missing classes
remain unknown. A bounded chronological window retains actual completed samples,
including repeated contexts. The controller compares expected service, so cached
affine context forecasts bound the conditional mean, not one future completion:
mean leverage scales fitted timing variation, inflated for lag-one serial
dependence of chronological residuals, plus the magnitude of the average
chronological miss measured before each completion entered the fit, which more
samples cannot average away. Uncertainty grows with extrapolation distance instead of expiring
an unused class after a fixed number of tokens. Repeated positions measure timing
noise but cannot identify a context slope; until a second position is seen, other
positions keep the level with a prior as wide as the level itself. Nonpositive
predictions remain unknown. A broad startup prior applies until residual variation
is identifiable; it does not impose a permanent fraction-of-mean uncertainty floor.
The statistical scaling and observed-error envelope are empirical assumptions,
not calibrated confidence guarantees, a proof of latency monotonicity, or a
substitute for missing widths. Shared flights and
unsupported shaping do not populate the table. Greedy unshaped masked and
unmasked selections use separate views of the same load-owned table: target
verification masks every row, while drafting masks only its first proposal.
History penalties and categorical selection remain uncovered.

At load, the selected configuration's qualified point measurements provide an
immutable physical service basis. Terminal cache records retain those same
measurements; no load repeats work to populate a forecast. Actual graph builders
declare each invocation's service row axis, readonly capacity axes and dependence
on visible history. Query position and visible-history length are distinct, and a
windowed layer retains its authored history limit. The case that creates tuning
arguments owns live-row projection onto its performance representatives. Artifact
identity and every other geometry dimension must match; prepared capacity alone
never establishes workload coverage. History interpolation is an empirical
forecast over measured points, not an exact workload identity or latency bound.

Ordinary isolated completions remain the preferred component observations. They
also calibrate the nonnegative additive residual over declared untimed work
families and multiplicities. Transfer is available only when the alternative's
untimed families are covered by those observations with supported multiplicity
ratios. A union of possible graph classes is not an observed work multiset;
different unmeasured class multisets remain unknown. Allocation uncertainty is
persistent, supplies no acceptance or reducible service information, and cannot
turn uncovered geometry into zero cost. These empirical forecasts permit an
independently generated action to be evaluated without manufacturing paired
prefix evidence, extra device probes or a mandatory exploration schedule.

Pending drafter entry is real work. The method supplies one actual round's
head and target operations, accepted-prefix-dependent post-target maintenance,
and successor boundary. It also describes the canonical causal catch-up retaining
one anchor at the incoming and successor boundaries. The forecaster prices these
operations; the controller charges actual round service plus successor catch-up
value minus incoming catch-up value. This potential telescopes over ordinary
rounds and submits no additional work for a decision. A target-reading head has
zero private-history liability by its actual state semantics. Plain work can
append pairs and require real zero-proposal maintenance; it is never represented
by an invented draft-head call or a missing price treated as zero.

Batch economies can make an individual potential-adjusted cost nonpositive. A
throughput comparison is available only with finite, strictly positive supported
conditional denominators; it never clamps an invalid denominator. Missing
current, maintenance or potential prices remain explicit missing evidence.
Shared flights cannot establish isolated service costs. There is no probe
interval, trial warm-up, exploration debt, or mandatory revisit.

A narrower incumbent cannot observe the tail a wider action would verify, and an
action's service is measured only when it runs. A blanket ban on taking another
action would make narrowing a one-way ratchet: a later rise in acceptance could
never be seen. Instead, another configured action runs for one round only when the
value of what it reveals exceeds its expected cost. The round is ordinary generation
with identical output, so its only cost is the expected loss if the incumbent is in
fact better. Its value is the probability-weighted gain if the candidate is better,
over the remaining regime, whose expected length is its observed age since the last
detected change. The posterior combines independent Beta conditional match
probabilities, each with the uniform prior's weight centered on the regime's
pooled match rate (so an unseen tail is expected to behave like the observed
prefix), and point service forecasts whose envelopes are read as two standard
errors. Same-group actions use their shared-prefix paired posterior. Distinct
generated blocks use separate conditional-match posteriors, each centered only on
its own group's observed depths. Their comparison is the sign of candidate expected
reward times incumbent expected cost minus incumbent expected reward times candidate
expected cost; it does not average outcome-wise throughput or manufacture matched
outcomes across blocks. Shared-group comparisons retain joint survival-product
moments in the ratio-of-expectations contrast; multiplying posterior means or
averaging realized reward/cost products would lose that dependence. The predictive
first-failure and all-match outcomes of the actual offered prefix determine the
expected posterior decision value directly. The empirical prior remains fixed
within the decision, and an observation updates only reached depths of its own
group. Future entry prices retain their acceptance-dependent geometry. Identical
physical operation coefficients cancel symbolically before missing-price lookup;
matching numerical estimates alone do not establish identity. Continuous service
learning uses the physical estimator's explicit normal approximation, conditioned
on each possible prefix observation. Persistent service bias is not reducible
information. This economic rule remains advisory, not an optimal or calibrated
exploration guarantee.

Only uncertainty the candidate's round resolves counts: its own service and its
own independently generated proposals, or a same-block tail no incumbent round
or unverified proposal reveals. One round resolves
only its share of each: about 1/(k + 1) of a posterior built from k observations
(tail trials past the narrow prefix; the candidate's previous rounds for service).
The value is that one round's knowledge gradient, the expected improvement of the
decision after it, so a burst of rounds inside one short request is not justified
by uncertainty those rounds cannot remove. A component the candidate's current
round would measure, that the cost model cannot forecast and the incumbent's own
round would not measure, is priced by one round; components the two share, or the
incumbent also lacks, are priced by running the incumbent. Exploration is
therefore rare when evidence is clear and ends by itself. Before a regime has any
evidence the configured action stands; the prior alone never overrides it.
After that initial evidence, a different generated group may be selected using
its own posterior or to price a candidate-only unknown current component. No
forced coverage round, shared-block prior, or periodic revisit is required. A candidate the posterior already favors is the Bayes action, not
exploration, and becomes the incumbent. Configured initialization remains
meaningful; it is not silently changed to make a benchmark favorable.

Acceptance evidence and the economic incumbent are load-owned and advisory: a new
request starts from the load's current incumbent, and output-grant clipping never
replaces it. Acceptance and physical service evidence are shared only within the
same loaded model, drafter, device, and program preparation; separate loads never
share them. Checkpoints retain numerical drafter
obligations without learned statistics. Fixed budgets collect no controller timing.
Existing completion boundaries supply feedback without GPU fences, logits transfers,
or catch-up operations performed solely for learning.

Qualification compares fixed action pairs, adaptive selection, and the same fixed
policy with instrumentation disabled. Report startup graph costs, controller CPU
cost, draft/verify/repair service, throughput, and output agreement separately.
Removing forced exploration eliminates its extra rounds; inexpensive host arithmetic
alone still establishes no throughput improvement. No adaptive default is enabled without this evidence.

Adaptive budgeting separates physical generated proposals from the prefix offered
for verification. One action comparison prices current service and the following
entry implied by each accepted prefix. Shared physical components cancel before
uncertainty is compared. The same comparison owns economic promotion and temporary
information collection: an ordinary incumbent completion also teaches its service,
so a candidate's information value is measured against that completion. Shared
component observations use their net utility coefficient and are counted once.
Acceptance censoring, deterministic later resolution of unverified tails, and
change detection retain their existing committed-state ownership. Missing service
coverage can justify a prepared action only when the incumbent's own completion
would not cover it. No decision launches extra measurement work.
