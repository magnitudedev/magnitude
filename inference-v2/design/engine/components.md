# Engine component assembly

These component types own the engine contracts, dimensions and performance bindings.
[Identification](../components.md) defines stable identities; [performance](../performance.md)
defines parameter origins, two evaluations and evidence handling. The
[catalog](../performance/catalog.md) indexes these authoritative definitions.

## Assembly

```text
ENGINE:INFERENCE:MAG:STANDARD
├── admission · SCHEDULING:ADMISSION:MAG:FIFO
├── batching · BATCHING:ASSEMBLY:MAG:READY_COMPATIBLE
├── execution · EXECUTION:DEVICE:MAG:ASYNC
├── generation · GENERATION:PLAIN:MAG:TARGET
│   ├── sampling · GENERATION:SAMPLING:MAG:POSITION_KEYED
│   ├── target · MODEL:QWEN35:MAG:LAYERWISE …
│   └── execution · EXECUTION:DEVICE:MAG:ASYNC ↗ execution
├── memory · MEMORY:ACCOUNTING:MAG:RESERVATIONS
├── prefixes · CACHE:PREFIX:MAG:CHECKPOINTS
└── scheduling · SCHEDULING:SERVICE:MAG:TIME_SHARING
    └── prefill · SCHEDULING:PREFILL:MAG:CHUNKED
```

## Component definitions

All records inherit platform parameters and evidence rules from the performance
system. Theoretical composition uses `JOIN`, `L` and the linked reusable derivations.
Implementation estimates use selected execution regions and matched local/parent
observations under [execution estimation](../performance/derivations/resources.md#execution-estimation).
Percentages below are formulas, not current qualification claims. Time/footprint
ratios are multiplied by 100 for display; zero floors yield no percentage.

### `ENGINE:INFERENCE`

**Contract.** Complete request admission, service, generation, publication and resource lifetime over the
selected model graph.

**Parameters.** Workload: arrivals, prompts, outputs, model/artifact identities, external readiness,
stops/cancellations and initial residency; configured admission/memory/share limits.
Architecture: selected model/state/generation contracts. Platform: the inherited capacity
profile.

**Composition.** Compose admission, service, compatible execution groups, model/generation work, prefix
restoration, state and publication through the [workload
relaxation](../performance/derivations/service.md#workload-relaxation). Neural work is
supplied once by the selected models; join required live information for feasibility.
Responsiveness and aggregate throughput can move independently, so retain all three
dimensions.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `ENGINE:INFERENCE/RATE` | Committed output tokens per elapsed second over the specified complete workload interval, including prompt and restoration work. | `u/L_service(W,makespan)` is the upper rate; efficiency `100*observed_rate/upper_rate`. |
| `ENGINE:INFERENCE/TTFT` | Specified population statistic of arrival-to-first-publication seconds. | `L_service(W,J_TTFT)` minimizes that same statistic; efficiency `100*L/J_observed`. |
| `ENGINE:INFERENCE/GAP` | Specified statistic of seconds between nonempty publications while generation-ready; record burst sizes and exclude requests without two publications. | `L_service(W,J_GAP)` minimizes that same statistic; efficiency `100*L/J_observed`. |

**Implementations and controls.**

#### `ENGINE:INFERENCE:MAG:STANDARD`

- **Implementation:** One execution-owner composition binds request admission, time-shared service, model/state dependencies and generation.
- **Reference / validation:** Matched upstream generation for neural outputs; finite request-trace oracle for
  lifecycle/service; compare real request outputs and completion boundaries.


### `SCHEDULING:ADMISSION`

**Contract.** Admit ready requests in FIFO order within active and memory limits; respect blocked
consumers/cancellation.

**Parameters.** Workload: ready arrivals, prior FIFO obligations, active capacity, required memory, releases
and reclaimable state; readiness/consumer constraints fixed.

**Composition.** Bind [time sharing and
admission](../performance/derivations/service.md#time-sharing-and-admission). Supply
capacity-release/restore feasibility from model and state dependencies. Immediate admission
permits a zero waiting-time floor; bookkeeping estimates remain visible separately.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `SCHEDULING:ADMISSION/LAT` | Seconds from externally ready arrival to admission at the specified FIFO workload. | `L_wait=max(0,t_free-arrival)`, with earliest feasible `t_free` in the relaxed problem; efficiency `100*L_wait/T_wait` only for a positive bound. |

**Implementations and controls.**

#### `SCHEDULING:ADMISSION:MAG:FIFO`

- **Implementation:** FIFO waiting-request selection gated by active capacity, memory and readiness.
- **Reference / validation:** Hand-derived queue/capacity traces and scheduler tests; supply arrivals/releases without model
  execution.

### `SCHEDULING:SERVICE`

**Contract.** Time-share eligible prefill/decode service with bounded debt and resumable rounds.

**Parameters.** Workload: already admitted requests, externally supplied model/state demands and publication
obligations. Configuration: decode share, chunk/duration limits, phase separation and any
proved finite share-error bounds.

**Composition.** Bind [workload relaxation](../performance/derivations/service.md#workload-relaxation) and
[time sharing and
admission](../performance/derivations/service.md#time-sharing-and-admission). The admitted
workload isolates service ordering from admission. During sustained contention, the refinement
`max(L_P/(1-s),L_D/s)` applies only under its declared conditions. Total throughput, prompt
responsiveness and existing output responsiveness can trade off.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `SCHEDULING:SERVICE/RATE` | Committed output tokens per second for the supplied admitted workload. | `u/L_service(W,makespan)`; observed rate / upper rate. |
| `SCHEDULING:SERVICE/TTFT` | Specified statistic of admission-to-first-publication seconds in that workload. | `L_service(W,J_TTFT)` / observed same statistic. |
| `SCHEDULING:SERVICE/GAP` | Specified statistic of generation-ready inter-publication seconds, with burst sizes recorded. | `L_service(W,J_GAP)` / observed same statistic. |

**Implementations and controls.**

#### `SCHEDULING:SERVICE:MAG:TIME_SHARING`

- **Implementation:** Completed physical service time updates decode debt; bounded overshoot and contention resets control phase selection.
- **Reference / validation:** Independent debt arithmetic and exhaustive small service traces; scheduler tests cover
  contention entry/exit and completed-time accounting.

### `SCHEDULING:PREFILL`

**Contract.** Allocate an aggregate prompt-token allowance among compatible FIFO peers; respect memory and
shorter tails.

**Parameters.** Workload: granted aggregate prompt-input allowance, row histories and compatible peers.
Configuration: selected model graph, memory limits, FIFO chunk and tail rules.

**Composition.** Bind [prompt work and
grouping](../performance/derivations/service.md#prompt-work-and-grouping) to the actual prompt
model graph. Consecutive chunk pair counts conserve semantic attention work. Additional weight
reads need a memory-boundary proof; independent groups enter the relaxed schedule.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `SCHEDULING:PREFILL/EXEC` | Elapsed seconds to complete the fixed granted prompt workload; `u` is newly consumed prompt inputs. | `L_service(W_granted,completion)`; `L(D_prompt)` is a resource refinement. |

**Implementations and controls.**

#### `SCHEDULING:PREFILL:MAG:CHUNKED`

- **Implementation:** Shares an aggregate token allowance among compatible FIFO peers; equal-width chunks batch and shorter tails form separate groups.
- **Reference / validation:** Direct token conservation and per-request advancement oracle; compare chunked versus unchunked
  target state.

### `BATCHING:ASSEMBLY`

**Contract.** Regroup ready continuations by operation/model/state/query compatibility; reuse state and
split memory-infeasible groups.

**Parameters.** Workload: supplied ready set, model/query/state/conditioning/output compatibility and memory
feasibility. Boundary: legal groups produced, distinct from completion of their model work.

**Composition.** Bind [prompt work and
grouping](../performance/derivations/service.md#prompt-work-and-grouping). Local descriptors
may be unnecessary; sorting and history copies are not contract requirements. The effect of
grouping is assessed through the parent’s `L_ready` and selected execution estimate, with
shared weights and independent row state.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `BATCHING:ASSEMBLY/EXEC` | Elapsed seconds from supplied ready set to legal groups. | Initial `L=0`; if `n*s` descriptor bytes must cross a boundary, use `max(0,n*s-S)/B`. No fabricated percentage for the zero-floor case. |

**Implementations and controls.**

#### `BATCHING:ASSEMBLY:MAG:READY_COMPATIBLE`

- **Implementation:** Ready continuations regroup by operation compatibility; stable state is reused and memory-infeasible groups split without rebuilding proposals.
- **Reference / validation:** Enumerated legal partitions; batched generation versus independently advanced rows, including
  unequal acceptance and membership changes.


### `EXECUTION:DEVICE`

**Contract.** One execution owner submits lazy work and retains leases until completion; scopes/spans
combine obligations.

**Parameters.** Workload: supplied graph, output roots, leases, device/host dependencies and required
completion boundary. Configuration: selected execution arrangement and allowed observation
semantics.

**Composition.** Bind [evaluation algebra](../performance/derivations/resources.md#evaluation-algebra) to the
graph plus required boundary demands. Use [execution
estimation](../performance/derivations/resources.md#execution-estimation) for actual
submission/completion/retirement. No mandatory per-layer/token host fence is assumed; lazy
graph construction is not completed neural work.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `EXECUTION:DEVICE/EXEC` | Elapsed seconds through graph completion and safe retirement of required resources. | `L(D_graph)` with only independently proved boundary/dependency refinements. |

**Implementations and controls.**

#### `EXECUTION:DEVICE:MAG:ASYNC`

- **Implementation:** MLX async submission under scopes/spans; pending executions retain leases until completion or a safe drain proves retirement. If forward preparation fails before execution, committed work may retire before one unchanged retry; execution failures remain terminal.
- **Reference / validation:** Fake completion backend and lease-lifetime oracle, then equivalent synchronous MLX execution;
  execution tests exercise failure and retirement.

### `MEMORY:ACCOUNTING`

**Contract.** Reserve allocations before use, retain in-flight obligations and reclaim eligible prefixes
before rejection.

**Parameters.** Workload: reservation/release transaction, physical allocation identities, sharing, last-use
obligations and budget. Configuration: reclamation eligibility and owner lifetimes.

**Composition.** Use [required live union](../performance/derivations/state.md#required-live-union) for
feasibility: `M_required(t)=|union(live necessary allocations)|`. Reservations are not extra
physical bytes; in-flight buffers remain charged. Bookkeeping may fuse into its owner, so a
positive time floor is not presumed.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MEMORY:ACCOUNTING/EXEC` | Elapsed seconds for the specified reservation/release transaction. | Initial `L=0`; only proved externally required resource demand refines it. Capacity correctness remains an exact constraint. |

**Implementations and controls.**

#### `MEMORY:ACCOUNTING:MAG:RESERVATIONS`

- **Implementation:** Budgeted reservations with explicit release and eligible-prefix reclamation before rejecting an allocation.
- **Reference / validation:** Independent live-allocation union and budget event trace; exercise sharing, replacement peaks,
  failure and release.


### `CACHE:PREFIX`

**Contract.** Namespace-aware prefix trie with complete generation checkpoints, leases and bounded retention.

**Parameters.** Workload: request trace, compatible constructible checkpoint prefixes and required anchor
exclusions. Configuration: namespace/state compatibility, byte/entry budgets, leases and
reconstruction obligations.

**Composition.** Bind [prefix reuse](../performance/derivations/service.md#prefix-reuse) to checkpoint/state
definitions, including shared physical backing. The offline/fractional optimum is an
explicitly optimistic relaxation; restoration/pressure costs remain in enclosing engine
objectives.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `CACHE:PREFIX/REUSE` | Reusable prompt inputs actually skipped for the specified trace and cache budget. | `H_max` from prefix-reuse optimization; efficiency `100*H_observed/H_max`. No ratio when no reuse opportunity exists. |

**Implementations and controls.**

#### `CACHE:PREFIX:MAG:CHECKPOINTS`

- **Implementation:** Namespace-keyed compressed token trie with checkpoint leases and configured retention policy.
- **Reference / validation:** Linear longest-compatible-prefix search and tiny offline retention oracle; prefix-retention
  tests and restored-versus-replayed outputs.


### `KV:STORE`

**Contract.** Store logical histories in a shared paged arena, preserving identity independently of physical
placement.

**Parameters.** Architecture: producer/head/key/value geometry and encoding. Workload: required retained
positions, shared prefixes/producers, checkpoint obligations and lifecycle observation
boundary.

**Composition.** Bind [required live union](../performance/derivations/state.md#required-live-union) to the
unique required materialized KV ranges. Append and branch are independently identified
operations. Page/slab padding and fragmentation remain excess storage under the logical
contract.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `KV:STORE/MEM` | Retained physical KV backing bytes at the specified boundary; shared backing once. | `M_KV_min=|union(required encoded KV ranges)|`; efficiency `100*M_KV_min/M_retained`. |

**Implementations and controls.**

#### `KV:STORE:MAG:PAGED`

- **Implementation:** A shared paged arena separates logical history identity from physical placement, growth and relocation.
- **Reference / validation:** Dense logical KV arrays and a unique-allocation ledger; page/placement tests across growth,
  reuse and relocation.


### `KV:APPEND`

**Contract.** Append new K/V through logical page/run placement without changing protected prefixes.

**Parameters.** Architecture: KV geometry/encoding. Workload: new values, logical positions, protected
prefixes and required visibility/persistence boundary.

**Composition.** Bind [append and visibility](../performance/derivations/state.md#append-and-visibility). The
neural producer may supply new KV internally. New required payload enters the boundary demand;
protected old history does not imply a compulsory copy.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `KV:APPEND/EXEC` | Elapsed seconds for new entries through required visibility/persistence. | `L(D_new_KV)`; positive write demand only at a required persistence boundary, otherwise a view append may have zero floor. |

**Implementations and controls.**

#### `KV:APPEND:MAG:CONTIGUOUS_RUNS`

- **Implementation:** Maps row positions to arena pages/runs and writes the new K/V region while preserving protected history.
- **Reference / validation:** Dense concatenation oracle; inspect new payload and peer histories across page boundaries.


### `KV:BRANCH`

**Contract.** Share protected prefixes and detach affected backing on mutation.

**Parameters.** Workload: protected logical history, immutable backing, branch independence and required
observable metadata. Mutation belongs to its later append/update.

**Composition.** Bind immutable sharing from [append and
visibility](../performance/derivations/state.md#append-and-visibility). Reuse protected
payload; subsequent copy-on-write is estimated from the selected representation and does not
prove a universal whole-prefix copy demand.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `KV:BRANCH/EXEC` | Elapsed seconds to create an independently mutable history view. | Initial `L=0`; refine only for a proved required metadata transfer. |

**Implementations and controls.**

#### `KV:BRANCH:MAG:COPY_ON_WRITE`

- **Implementation:** Shares protected immutable page backing and detaches affected storage when later mutation requires independence.
- **Reference / validation:** Independent dense cloned histories as a semantic control; branch, mutate and release in
  different orders.


### `STATE:RECURRENT`

**Contract.** Own recurrent matrix/convolution images and accepted-boundary reconstruction obligations.

**Parameters.** Architecture: matrix/convolution geometry, state encoding and update equations. Workload: live
states, required checkpoints, advanced/accepted positions, lifecycle boundary and memory
budget.

**Composition.** Join recurrent/convolution information using [required live
union](../performance/derivations/state.md#required-live-union) and [restoration
cases](../performance/derivations/state.md#restoration-cases). The [hybrid
state](../models/architectures/qwen35.md#stateqwen35) composes this with attention. More
snapshots can reduce repair while increasing retained bytes; transient peaks and
advance/creation costs remain enclosing obligations.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `STATE:RECURRENT/MEM` | Retained physical bytes of live recurrent/convolution state and required checkpoints at the specified boundary. | Required materialized union `M_min`; efficiency `100*M_min/M_retained`. |
| `STATE:RECURRENT/RESTORE` | Seconds to accepted-state readiness, including repair deferred before next use. | `L_restore(initial,accepted,obligations,budget)`; efficiency `100*L/T`. All legal strategies constrain the bound. |

**Implementations and controls.**

#### `STATE:RECURRENT:MAG:CHECKPOINTED`

- **Implementation:** Recurrent/convolution images and transition records support accepted-prefix selection or reconstruction.
- **Reference / validation:** Independent recurrence from a saved prefix; reconcile first/interior/final accepted boundaries
  and compare restored state and next output.


### `GENERATION:PLAIN`

**Contract.** Consume the anchor, sample a target output and preserve the next anchor/state boundary.

**Parameters.** Architecture: selected target model/state and sampler. Workload: anchor/history, sampling
constraints/position keys, output allowances, stopping/publication and readiness.

**Composition.** `D_plain=JOIN(target,sampling,required_state,publication)` using [generation
rounds](../performance/derivations/service.md#generation-rounds). Target and
[sampler](#generationsampling) may fuse without externally materializing vocabulary logits;
the required output boundary determines demand.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `GENERATION:PLAIN/EXEC` | Elapsed seconds for a complete plain advancement interval; `u` is committed outputs. | `L(D_plain)`; committed-output upper rate `u/L`. |

**Implementations and controls.**

#### `GENERATION:PLAIN:MAG:TARGET`

- **Implementation:** Target advancement followed by position-addressed sampling and state/anchor publication.
- **Reference / validation:** Independently stepped target sampler with the same position keys, stopping and constraints.


### `GENERATION:SPECULATION`

**Contract.** Draft, verify, accept target-sampled matches and reconcile each request before publication.

**Parameters.** Architecture: selected target, proposal method/attached model and state contracts. Workload:
depth `d`, target query width `d+1`, acceptance/route/stop distribution or explicit optimistic
relaxation, allowance, readiness and memory.

**Composition.** Bind [generation rounds](../performance/derivations/service.md#generation-rounds) to
draft/verify, [sampling](#generationsampling), [acceptance](#generationacceptance) and
restoration. MTP selects the [Qwen head](../models/architectures/qwen35.md#modelqwen35mtp);
head layer count differs from proposal depth. Draft dependencies, borrowed weight sharing and
deferred catch-up remain explicit. Other methods need their own draft graph/state binding.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `GENERATION:SPECULATION/EXEC` | Complete draft–verify–repair/publication interval per committed output under the fixed joint workload. | `U=E[u]/E[L(D_round)]`; with no acceptance law use the labeled optimistic `(d+1)/inf_k L(D_round(k))`, bounded by allowed outputs. |

**Implementations and controls.**

#### `GENERATION:SPECULATION:MAG:TARGET_MATCHING`

- **Implementation:** Proposal generation, target-sampled prefix matching and per-request accepted-boundary reconciliation.
- **Reference / validation:** Plain target continuation with identical logical sampling positions; test zero/full/partial
  acceptance, stop and divergent batched progress.


### `GENERATION:SAMPLING`

**Contract.** Apply request history/constraints and sample by logical output position, independent of
batching.

**Parameters.** Architecture/configuration: sampling policy, numerical and random-position semantics.
Workload: eligible vocabulary, logits/history/constraints, required log probabilities and
boundary residency.

**Composition.** Use [sampling and acceptance](../performance/derivations/service.md#sampling-and-acceptance)
and the local neural equations. Arbitrary greedy logits require considering eligible
candidates; full sorting is not generally compulsory. A fused head/sampler does not force an
external vocabulary tensor or host fence.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `GENERATION:SAMPLING/EXEC` | Elapsed seconds for `u` sampled positions through required token/logprob readiness. | `L(D_sampling)` with boundary data and justified comparison/normalization demands. |

**Implementations and controls.**

#### `GENERATION:SAMPLING:MAG:POSITION_KEYED`

- **Implementation:** MLX policy filtering and argmax/categorical selection, with random keys derived from seed and logical output position.
- **Reference / validation:** Explicit distribution/greedy equations and position-key controls; sampling tests with
  policies, ties and regrouping.


### `GENERATION:ACCEPTANCE`

**Contract.** Select the consecutive matching nonterminal prefix and its target bonus token.

**Parameters.** Workload: width `d`, proposal/target samples, terminal rules, accepted prefix `K` and required
count/bonus output boundary.

**Composition.** Bind [sampling and acceptance](../performance/derivations/service.md#sampling-and-acceptance).
The first mismatch/terminal fixes the accepted prefix. Inspect only necessary proposal/sample
pairs; rejected suffix work and cumulative-product arrays are not theoretical requirements.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `GENERATION:ACCEPTANCE/EXEC` | Elapsed seconds per processed round through accepted count and bonus readiness. | `L(D_acceptance)`; conditional comparisons inspect `min(K+1,d)` pairs, with no positive floor inferred solely from a fused interface. |

**Implementations and controls.**

#### `GENERATION:ACCEPTANCE:MAG:PREFIX`

- **Implementation:** MLX match/terminal masks and cumulative prefix products derive the accepted count and target bonus token.
- **Reference / validation:** Scalar prefix scan; exhaustive short proposal/sample/terminal combinations.


## Qualification and attribution

Existing controls cover lifecycle, state, sampling and service behavior; not every
boundary has dedicated performance measurements. The engine/service objectives
include supplied model work; local bookkeeping dimensions do not claim ownership
of another copy of that work. Preserve observation boundaries when estimating costs.
Current scores follow the [assessment rules](../performance.md#stable-compositions-and-evidence),
including implementation/child fingerprints and transitive invalidation. A correct
control does not establish a numerical ceiling or measured efficiency.
