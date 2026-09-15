---
applies_to:
  - inference-v3/src/ops/**
  - inference-v3/src/engine/**
  - inference-v3/performance/**
  - inference-v3/src/session_bench/**
  - inference-v3/tests/session_bench/**
  - inference-v3/tests/ops/**
  - inference-v3/tests/models/**
  - inference-v3/tests/platform/**
  - inference-v3/tilelang/tilelang/metal/transform/**
  - inference-v3/tilelang/src/backend/common/codegen/codegen_c_host.cc
  - inference-v3/tilelang/3rdparty/tvm/src/runtime/metal/metal_module.mm
  - inference-v3/tilelang/testing/python/metal/test_metal_submission_regions.py
---

# Formula-defined execution and measurement

## Purpose

Stable mathematical boundaries make optimization comparable even when kernels,
fusion, storage and execution schedules change. The objective is efficient
execution relative to justified resource ceilings, not merely parity with an older
engine. V2 is reference evidence, not the performance ceiling.

## Ownership

- Engine owns architecture composition, blueprints, artifact interpretation,
  admission, logical prefix state, acceptance and reclamation policy.
- Ops owns formulas, operations, representations, complete physical execution,
  resource lifetimes, characterization and shared measurement facilities.
- TileLang owns the portable kernel language, target resource resolution, kernel
  scheduling/autotuning, compilation and native execution/timing.

Resource budgets reflect the resolved compilation target and explicitly selected
execution device. Missing resource information is unknown, never a claim of zero
capacity. Ops authors matrix work tiles and reduction steps; TileLang selects
instructions and infers layouts from the actual program. No instruction catalogue
or synthetic compiler probe gates construction. Native composition and binding
are execution-interface requirements.
Plan and measurement identity includes compiler provenance, target configuration
and resolved schedules.

Engine never imports TileLang; ops never imports engine. Public ops contracts do
not expose TileLang/TVM values. Numerical work, including import conversion, uses
portable TileLang. Source I/O is not numerical kernel computation.

DeviceTopology is a non-live inventory including memory domains and relationships.
Host discovery reports every endpoint executable by the process's existing transfer
runtime and its physical memory backing. CUDA and HIP discovery follows the installed
Torch build on Linux and Windows; Metal uses the operating-system API. An integrated
accelerator references the host memory domains and host physical budget; a discrete
accelerator contributes its own device-memory domain and budget.
Compiler-specific capabilities such as legal matrix instructions and kernel resource
usage remain the compiler target's responsibility.
DevicePlan selects resources and MemoryConstraints without allocating them.
Ops DeviceRuntime is the sole live owner, including reservations, completion,
characterization caches and shutdown. Engine's plan implements an ops-owned
read-only protocol rather than creating a second mutable configuration.

Comparable evidence includes the live runtime identity as well as the non-live
device plan. Ops includes OS build, transfer-runtime and compiler provenance;
TileLang exposes native SDK/driver versions through a thin read-only interface.
Metal's driver is OS-owned, so its OS build is essential. CUDA/HIP must report
their driver version rather than silently reuse evidence with incomplete identity.

## Formula and operation

A formula is a Python-defined numerical and effect contract, with typed ports,
shape/precision/representation rules, independent reference semantics and useful
quantities. Composition preserves occurrences, parent/child relationships, shared
dependencies and versioned state effects. Static control uses known facts;
data-dependent control uses explicit traced constructs. There is no string DSL.

An operation is one physical implementation of a formula. Its body may compose
children or implement the parent directly. Shape, representation and capability
branches belong inside that definition. There is no competing implementation
registry, cost-ranked cover, unknown-cost preference or ops autotuning layer.
TileLang owns kernel schedule tuning; retuning is an explicit development action.

An operation encompasses source reads, allocations, staging, transfers, numerical
conversion, kernels, dependencies, synchronization and temporary release. It
expresses that work through shared ops facilities. DeviceRuntime owns the actual
allocations, execution and lifetime mechanics; an operation does not implement a
private allocator, source profiler or backend launcher. The engine does not hide
recurring I/O outside the operation's execution boundary.

Blueprint residency choices change physical execution and bindings, not target
values or formula semantics. Drafting adds separate formula invocations and an
acceptance protocol; it does not silently change the target formula.

## Physical execution

Planning proves readiness, precision/effect ordering, compatible layouts, alias
safety, lifetimes and capacity. It does not estimate latency to choose between
implementations. Compilation and warm invocation are separate; warm invocation
does not trace, plan, compile or retune.

Formula boundaries are not host submission barriers. Ordered authored TileLang
work composes into maximal native entrypoints, with immutable operands partially
bound. A required host observation, streaming dependency or documented native
limit may split a unit. Fusion means eliminating an intermediate inside one
kernel; several kernels behind one host entrypoint are not fusion.
Safe contiguous device-launch regions also share the native queue handoff and
compute-pass lifetime where the execution adapter supports it. Argument-frame
preparation may enter that region; arbitrary host callbacks, external effects and
returns remain explicit boundaries. A scoped owner closes the pass on both normal
and exceptional exits before returning the borrowed command buffer to its owner.
Per-kernel timing may require separate passes and must identify that instrumentation
rather than claiming it observes an unchanged ordinary submission path.


Device launches enter the native submission queue in program order, preserving
argument preparation, control flow and individual kernel observations. A queue
crossing does not commit a command buffer or wait for GPU completion. Arbitrary
host callbacks remain on the submitting thread. An error stops later launches,
reaches that original thread and preserves completion ownership for earlier work.

A parent operation may absorb a child's output publication and its own residual
addition. It must still round the child result to the child's declared dtype
before widening and adding the residual. An isolated child measurement returns
the original child value; exposing that intermediate prevents its elimination.
Formula hierarchy and evidence identity do not change merely because this storage
and dispatch boundary is fused away in the parent.

Grouped attention may reuse compact K/V staging across query heads sharing one
KV head. That storage choice does not narrow probabilities, online softmax state
or output accumulation: those remain FP32. Value operands widen on consumption,
and the isolated and composed attention paths share one authored streaming body
and physical-capacity calculation. Native shared staging is not a claim that all
arithmetic has the storage dtype or that reduced storage guarantees lower latency.

Wide-head persistent prefill may stream bounded coordinate slices while keeping
its complete query-row cohort. Its resource plan prices the actual sliced arena,
not a whole-head allocation. QK finishes all coordinate slices before each FP32
softmax update; PV consumes all value slices before advancing history. The
probability fragment remains with its query-row owners, and immediate matrix
subfragments use statically selected coordinates so an enlarged register tile
does not require dynamic fragment-array addressing. Logical head width still
controls normalization and codec interpretation; physical slice width does not.

Persistent attention receives prepared logical queries, keys and values after
the model's normalization and positional transforms. Ops owns the KV codec,
packing, history-query transform, append, history traversal and partial merge;
model equations supply geometry and visibility without selecting codec kernels.
The current batch attends through its dense prepared values. Committed history
is consumed directly from its representation in bounded on-chip tiles. Current
rows must not be read back from their freshly encoded destinations, and packed
history must not acquire a dense full-prefix shadow or reconstruction buffer.
Both sources participate in one FP32 online softmax and return the model's
logical value basis before any nonlinear output gate.

A persistent KV representation is an immutable semantic identity covering codecs,
widths, metadata precision, packing, codebook and rotation conventions. Independently
aligned code and metadata planes belong to one backing resource; claims, copies,
completion pins and reclamation cover that entire bundle. Read visibility separates
committed history from current causal rows and reserved write destinations. A state
advance becomes visible only after both attention consumption and persistence finish.
An explicitly reserved append promises that its destinations are disjoint from
the committed-history intervals consumed during that advance. Its reservation
owner establishes this precondition. Only this declaration permits a component
to schedule producer persistence alongside a logical pre-advance read; arbitrary
overwrites retain their ordinary ordering. The component owns preparation,
attention and append together, so linear resource versions remain valid at its
external boundary and completion joins the output with the next cache state.
Compression memory savings and attention latency are qualified independently;
chunked prefill comparisons include every chunk's packed-prefix traversal and append.
Matrix query tiles derive their traversal extent from valid row visibility,
including partially padded tiles. A zero-visibility final physical row cannot
suppress preceding valid queries, and padding never enables an unmasked fast path.

Sequential execution may retain a bounded set of output backings. A backing becomes
reusable only when its retained owner holds every remaining allocation lease,
including completion pins and checkpoint/state views. A busy set causes fresh
allocation; it never permits overwriting an escaped result. Idle backings remain
charged to the device budget and are explicitly reclaimable. Closing a compiled
owner releases its claims without invalidating results retained by callers.
Native view reuse changes binding work, not tensor math or state publication.
Validated resource view geometry is immutable and may be shared by independent
leases. Forking a lease preserves that geometry; constructing a different view
still validates its bounds and alignment against the backing allocation.
An invocation may transfer its integer control fields in one dense word record.
Its public typed materialization preserves every signed-index and random-draw
bit, owns its output storage, and remains inside the measured invocation. Fields
shared across layer consumers need materialization only once; mutable model state
retains its ordinary resource and completion ownership.
Report preparation and submission-through-completion time separately so reducing
host preparation cannot masquerade as faster numerical kernels.

Kernel composition uses public Python-native TileLang construction, not generated
source, AST fabrication or direct TIR manipulation. Physical storage planning
respects alignment and live intervals; ABI views obey actual allocation bounds.
`OperationContext.kernel` receives a TileLang `@T.macro` or a host callable
composing such macros. Device loops belong inside the macros; ops derives the
typed ABI from formula ports and does not provide a second kernel-language tracer.

Reserve physical source, staging, conversion, output and temporary capacity before
allocation. Shared backing is charged once; aliases and leases do not allocate
again. Retain resources until every submitted consumer completes. Cancellation
stops future work and drains submitted work before reclaiming its storage.
Physical completion does not imply logical acceptance of a prefix advance.

A failed composite launch may already have submitted earlier kernels. Its failure
carries completion ownership; enclosing invocations and streamed tile scopes keep
their resources pinned until that completion drains. A launch error is not proof
that no device work was submitted.

Gated-delta recurrence publishes a separate following-state allocation, leaving
the prior checkpoint untouched; its operation must not invent an in-place alias.
An optional static single-sequence length is a declaration that packed offsets
are exactly zero and that length. Engine may specialize complete unpadded
single-sequence prefill only after deriving those facts from the actual admitted
rows. The geometry participates in executable identity; partial and packed
invocations retain runtime offsets. Priming covers both full and padded classes,
and repeated invocations reuse their compiled executables.
Pure arithmetic can read resource-valued ports. Those read hazards are derived
from the actual ports even when the primitive also accepts immutable tensors,
so a later independent writer cannot overtake an earlier delayed consumer.

Streaming operates on bounded source regions, preserving packed alignment, logical
indices, accumulation and valid tails. Prepare executables outside the tile loop.
Prefetch/overlap requires real capabilities, dependencies and reserved capacity.
Whole-weight transient loading is not a substitute for bounded streaming.

Root weight bindings explicitly group derived source regions for cache eviction.
Artifact bindings can be described from container metadata without opening a live
device or reading tensor payloads. Their source owner outlives all binding users;
describing a binding does not import it or acquire a runtime cache claim.
Unloading a provider releases its runtime cache claims; existing compiled and
completion leases remain valid. Reopening an artifact rebinds its current source
handle, while same-geometry conversion programs can be reused. Source fills use
caller-owned staging (`read_into`), including concatenated and zero-filled regions,
so composite reads do not conceal extra proportional host allocations.

Engine grows KV slabs with retained prefixes, not sessions or fixed maximum
histories. Engine decides which reclaimable prefix/model cache claims to release;
ops reports capacity and completion-pinned storage. Allocation failure does not
authorize eviction of active prefixes or implicit unloading of an active model.

## Measurement boundary and attribution

The runner identifies a formula occurrence and invokes its production operation.
Prepared Lab configurations reuse the engine's ProgramDefinition and invocation
argument names, not another benchmark model. Root inputs may be captured lazily. Intermediate inputs may derive from independent
primitive references or from executing the production prefix once; that choice is
recorded. Production captures provide the starting values, never the correctness
oracle for the selected implementation.
Configuration selection passes typed prepared objects and opens only the selected
worker-owned device. Switching drains and closes that owner before opening another.
An isolated artifact fixture may supply synthetic boundary inputs, but must label
that condition explicitly. It retains the production trace's formula, weight roles
and physical bindings. Independent artifact decoding prepares reference values
only; it is not an alternative numerical execution path.
Shared runtime instrumentation observes source reads, transfers, reservations,
submissions and completion. Useful units come from the formula; implementations
do not duplicate metric formulas or report their own estimated work counters.

The complete invocation interval includes all recurring I/O, conversion, kernels,
required completion and transient cleanup. Initial resident import, compilation,
fixture setup and reference preparation have separate phase times. Returning from
an asynchronous submit is not completion. A failed or unfinished observation is
not a valid successful latency measurement.

For streamed projection, report the enclosing invocation wall time, source bytes
returned, transfer API bytes, host spans, available native device timings and
reservation baseline/peak/end. The formula supplies useful projection work and
outputs. A resident projection shares the formula but has different residency
conditions and performs no recurring source read.

Source API traffic is not physical disk traffic: OS caching may satisfy a read.
Transfer API bytes are not proof of a bus copy, particularly with unified memory.
Host submit/wait durations are not GPU durations. Hardware traffic and device
timings require actual counter/timestamp support and explicit provenance. A metric
without such support is unavailable, never a fabricated zero or renamed host timer.

Each physical event is recorded once. Attribution may link it to composed formula
occurrences without duplicating it in totals. Parent elapsed time is measured
directly; isolated child times and overlapping stage spans are not additive.
Isolated measurement is a normal operation on any supported complete formula
subtree. It reuses the production operation definitions and captured/reference
boundary inputs, not a benchmark-specific numerical implementation. A subtree
request measures its parent and children independently. The parent retains its
actual fusion; a child's isolation may introduce publication and transfer costs.
The stable hierarchy displays both without presenting isolated times as actual
contributions inside the parent. Fusion is not a reason to leave the tree empty.
Allocation peaks come from the live reservation ledger, not logical tensor sizes
or the runtime's historical high-water mark. Baseline and additional demand are
distinct; report each physical constraint as well as aggregate unique backing.

## Ceilings and characterization

One directly derived memory bound is the fresh escaping output footprint under
the invocation ABI: follow primitive alias effects, count shared backing once,
and exclude borrowed inputs/state. Compare it with observed reservation increase,
not lifetime peak. It excludes scratch and transfer staging and is not a latency
prediction. Alias declarations must be preserved and cannot invent input mutation.

Measurement, numerical qualification and resource analysis are independent results.
A completed timing remains available when useful-work analysis or compatible device
characterization is unavailable. An explicit exploratory protocol can time a
numerically failing implementation, but cannot qualify it as correct or best-correct.
Unsafe state mutation and incomplete execution remain execution failures. Qualified
resource analysis connects formula-derived demands to compatible characterization
and the actual complete-operation duration. Conventional contraction work is separate from vector, integer, special
function and comparison work. Primitive semantic rules own those quantities;
implementations do not duplicate them. Concrete control values determine visible
attention, selected experts and sampling policy. Finite algorithm-dependent work
ranges remain ranges; missing semantic rules or resource evidence fail model
qualification rather than silently selecting a different objective.

The modeled roofline divides demands by measured resource references, adds demand
within a shared resource pool and takes the maximum across constraints under ideal
overlap. It is not a prediction of this implementation's latency or proof of an
absolute hardware limit. Its assumptions, precision, probe identity and source
conditions are part of the recorded evidence. Ratios above 100% remain visible and
challenge the reference/model; they are not clamped or called super-optimal.

Resource probes must expose throughput rather than application staging overhead:
matrix probes reuse operands on chip and memory probes use coalesced element
streams. Their counts still come from ordinary formula semantics and boundary
traffic, not handwritten benchmark counters. The best measured resource rate is
the empirical ceiling reference; matching a small probe's launch-dominated latency
is a performance prediction, not a roofline. Storage bandwidth is byte-based and
does not require matching the stream's arithmetic dtype. A populated tree with
large above-reference results fails calibration qualification.

Resource calibration explicitly records sustained warm-up duration to avoid
comparing a cold low-power GPU probe with a long-running model. Ordinary edit
measurements do not inherit this setup cost. A changed conditioning protocol creates
new calibration evidence; an absent additional warm-up preserves ordinary series
identity. Native operand fragments and sufficient independent work avoid measuring
matrix staging/recurrence latency as device capacity.

Prefill MoE composes homogeneous tiled routed and shared contractions in one
submission. Their packet interpretations, reduction widths and scratch legality
are independently specialized; a persistent heterogeneous worker must not force
one branch's geometry or code onto the other. Both use the same authored tile
bodies, retain grouped routing and matrix arithmetic, and share the final weighted
combine/residual publication. Prefill routing preserves batched matrix reuse;
decode may fuse per-row projection with selection. Kernel count alone does not
justify fusion that destroys those properties.

Ideal boundary traffic counts unique accessed backing, concrete selected ranges
and escaping writes, not allocated cache capacity or every internal intermediate.
Packed storage and source encoding are distinct. Source demand unions the required
encoding-aligned spans by immutable source snapshot. Cache/traffic assumptions must
remain visible; API traffic cannot be relabeled physical DRAM or disk traffic.

Packed contraction decodes coefficients at their declared precision and preserves
the formula's numerical contract. Operand precision is a complete resource
tradeoff: extra correction products can outweigh smaller shared tiles when native
arithmetic rates are similar. Silently discarding coefficient precision is not a
scheduling optimization. Activation range, conversion error, correction work and accumulation
error must be accounted for together. Exact code/coefficient contraction remains
appropriate where the target cannot realize the qualified operand representation.
Neither whole-weight expansion nor a changed tolerance establishes a faster
same-contract implementation. Packet owners reuse coefficient work and row cohorts
reuse weight tiles; measured operation time includes every correction product.

Hierarchical affine source coefficients are resolved to direct FP32 metadata
during import while codes retain their packed widths and interpretation. Ops
owns this execution representation; the artifact codec owns source decoding.
The prepared allocation replaces the old representation rather than retaining
both. Streamed imports remain bounded and expose conversion/publication bytes
separately from source bytes. Reservation and evidence identities include the
larger execution footprint.

Grouped prefill resolves routed input rows before channel-tiled contractions.
Packed matrix contractions stage original activations, exactly represented small
integer codes and original FP32 coefficient pairs. They reconstruct only the
immediate FP32 matrix operands and retain one FP32 accumulator across the complete
reduction. Quantization groups select metadata; they do not reset the accumulator
or require output-wide coefficient correction or activation-sum workspaces.
This common contraction serves resident, streamed, ordinary, parallel and grouped
projections. It does not round reconstructed weights to a 16-bit dtype. Matrix
work tiles and reduction steps are authored by Ops; TileLang selects instructions
and infers layouts during normal compilation. Decode's packet-vector arithmetic
is separate. Gather work and staging remain charged inside the operation boundary,
not permanent caches or hidden setup. Grouped down contractions consume their
contiguous activation rows without redundant route lookup.

Structural device facts come from supported interfaces. Ops runs bounded reusable
portable probes to characterize applicable arithmetic, memory and transfer paths.
Source-path characterization belongs to the source/path and cache conditions, not
a GPU bandwidth field. Cache by resource identity, runtime/compiler/probe version
and relevant conditions; refresh explicitly, not on every measurement.
Characterization is an explicit agent request, which loads compatible cached
evidence or runs bounded probes. Ordinary measurements never initiate calibration;
an absent compatible profile is reported as unavailable. Ordinary operation edits
do not rerun calibration or move its denominator. Explicit characterization of a
source-backed configuration also probes its actual streamed source paths. Memory-source evidence is never substituted for file-source evidence.

Ops owns the byte-source protocol and source provenance: source identity, snapshot
revision, kind, optional location and composed backing sources. Engine artifact
adapters implement that protocol without defining a duplicate one. File provenance
identifies the opened snapshot, not merely a filename which might be replaced.
Prepared source imports are keyed by source provenance as well as logical values;
equal weights on different streaming paths must not silently reuse the first path's
reader. Compiled numerical code may still be shared. A file source is not asserted
to be a physical disk: filesystem/page caching and remote backing remain distinct.

Formula-derived lower bounds must state their residency, precision and resource
assumptions. Distinguish theoretical peaks from measured sustainable ceilings;
an empirical probe maximum is not a proof of optimality. Compare matching units,
paths and conditions. Do not restore a latency simulator or fallback selector.

Bounds are not restricted to rates: a formula may have a throughput ceiling, a
latency floor or a minimum storage requirement in the appropriate units. Record
whether higher or lower is better, the modeled value and its assumptions. Runtime
baseline/peak and incremental reservation demand are separate metrics; a peak that
includes other cached preparations is not the operation's isolated storage minimum.

## Persistent development system

The agent owns execution: workload plus scope selects what to run. A persistent
measurement worker owns fixture/reference reuse, isolated measurement, dependency
invalidation and durable observations. Benchmark requests and their recipes remain
owned by the existing benchmark system; a formula fixture materializes the chosen
boundary for those inputs, or explicitly identifies synthetic inputs. There is no
second workload renderer. Enclosing request records, production forward observations
and isolated formula measurements publish into the same evidence store.
Independent workers and readers may open the same history file concurrently.
First-open WAL initialization handles transient SQLite lock upgrades within a
bounded deadline; it must not downgrade journaling or swallow permanent errors.
Selection uses Formula objects and typed occurrence handles, never string paths
or dynamic attribute proxies. The engine's typed hierarchy is derived from actual
formula occurrences, not a second benchmark-only model tree.

Comparable history is keyed by formula semantics/revision, shapes, precision,
representation, inputs/artifacts, state, residency/cache conditions, device and
measurement protocol. Operation implementation fingerprints belong to individual
observations, not the history-series key. Changing an operation marks affected
occurrences and dependent ancestors stale; it preserves historical measurements.
Moving a display node does not discard a comparable series. Shared dependencies
remain a DAG rather than falsely independent tree work.

The primary TUI is read-only and starts with a stable model identity. It displays
already published engine performance, condition-specific histories, formula
observations, ceilings and provenance. Opening or refreshing it never opens a
device, constructs a fixture or queues measurement. The agent executes and publishes;
the user does not have to select a workload to inspect a model. Unmeasured conditions
stay unmeasured and current-code freshness is unknown until verified. Detailed
formula selection is pinned to the selected execution, not a later result silently
substituted from the same series. Existing developer measurement clients use the
same worker; they do not define the model overview's behavior.

Model identity groups evidence; it does not make timings interchangeable. Artifact,
numerical contract, concrete workload/state, hardware, host, protocol and scope
qualify comparisons. Implementation revision belongs to each observation. Histories
follow execution timestamps rather than import order. Different hardware contributes
evidence about the same formula and implementations, with separate resource rates
and measured curves. No averaging device timings or assuming normalized efficiency
transfers between devices. Paired deltas require the same host and explicit shared
pair identities. Failing numerical candidates remain visibly unqualified.

External engines retain their native timing boundaries and token counts. Formula
mapping identifies an equivalent contract, declared differences, or an enclosing
region, with versioned supporting evidence. Opaque or fused regions stay combined;
missing attribution cannot be replaced by a guessed split. Reference engines supply
observations and hypotheses, not an upper bound on performance.

Remote execution is ordinary agent coordination: copy the request and necessary
inputs, invoke the same headless tools over SSH, and copy a portable evidence bundle
back. Import validates schema, integrity, references and immutable identities before
atomically publishing records; it does not execute imported code. Duplicate imports
are harmless. Compiled source artifacts travel by content hash. Large model blobs
remain separately provisioned. There is no scheduler, remote daemon or automatic
host selection. Independent investigations may run on the two M4 Pros; comparisons
within one investigation remain paired on the same host.

Each prepared configuration records its actual composition and links measured
occurrences to their exact comparable series. Recorded browsing needs no model,
fixture derivation or live device. Those recordings are display/provenance data,
not deserialized executable handles or a second model definition. A new live
configuration establishes its input conditions on measurement; do not substitute
another recording's conditions merely because formula names and shapes match.

The development target is under five seconds from editing a small operation to a
checked, persisted, visible observation, including affected compilation. Keep the
worker/resources alive, reuse independent fixtures/reference and compile only
affected code. Use bounded sampling. Report setup, compile, check, sample and
publication times separately. An unchanged warm cache hit does not qualify the
changed-operation target.

Complete-operation host time and native kernel execution time are separate
observations. Where supported, the same sample may include bounded native
timestamp intervals for every numerical dispatch on the production stream.
Native endpoints preserve interval unions and overlap; summed kernel durations
are not GPU busy time when work overlaps. Neither is complete-operation wall time.
Attribution checks the compiled dispatch count and symbols against captured native
events, retaining origins and one exclusive formula owner where justified. Dynamic
or mismatched launches keep valid durations with attribution unavailable. Inclusive
formula views may share fused events, while totals count every event once. Formula useful-work rates state which denominator they
use. Native instrumentation is opt-in, versioned in the measurement protocol,
and never requires replay or a second implementation. Unsupported counters are
explicitly unavailable; overflow, incomplete execution and invalid counters do
not produce successful partial timings. Observation never implicitly commits
or synchronizes the execution stream.

Persistent executable keys include both Python lowering content and native
compiler-library content. A native rebuild must not load a binary produced by
the preceding compiler under a newly recorded compiler identity. Compiler edits
require a fresh process; operation edits use the live worker's bounded refresh.

Persist request starts and terminal job outcomes separately from performance
observations, so failure before fixture/series construction and unfinished work
remain visible without inventing numerical evidence. Queue time is distinct from
active work. A measurement client may acknowledge a result after rendering; only that receipt
measures request-to-visible turnaround. Worker completion and read-only browsing
must not fabricate a display acknowledgement.

Operation refresh reloads original authored Python definitions and their import
dependants, not the live device owner. Executable provenance follows loaded code
and referenced helpers, including constants imported inside operation functions,
not every file in the kernel package. Immutable Python code analysis may be cached
within explicit bounds; imports, globals and definition dependencies must resolve
against the installed source revision. Failed refresh must
not continue with a mixture of new and old definitions. Edits to conversion code
invalidate affected prepared imports and derived cache claims as well as numerical
entrypoints; immutable fixture/reference semantics and unrelated programs remain
reusable. Structural changes to a live owner require reopening that configuration,
not changing the class of already-owned resources in place.

The measurement worker owns one bounded cache of boundary references and native
preparations. Fixture indexes are non-owning; visiting more formula nodes must not
retain an unbounded collection of decoded weights and intermediate tensors. Retain
immutable references across operation edits while replacing affected executables.
An individually oversized boundary may run but is not kept beyond its request.
A retained production boundary records upstream code dependencies, precise inputs
and starting state. Editing the selected implementation preserves independent
upstream work; editing a captured dependency invalidates it. Explicit disk replay
uses a fixed historical input snapshot with provenance, rather than pretending to
be a recomputation of the latest upstream code. Snapshot contents and tensor schemas
are validated without deserializing executable code; immutable artifact bindings
must resolve to matching values. Mutable resources are reset before every invocation.
Unsupported alias layouts fail explicitly rather than silently copying independent
values. Disk retention does not imply device residency after the worker exits.
Live configurations explicitly set positive preparation-count and reference-byte
budgets. Large attention references can therefore remain resident for repeated
operation edits without making the default cache unbounded. Formula effects
determine conditioning: read-only resources stay resident, while written resources
receive fresh replicas before each check/sample. Failed or cancelled execution
retires the live preparation, since a faulty implementation may have corrupted an
input declared read-only. The immutable reference remains reusable.
Read-only resource verification is exact physical-byte comparison, not another
decoded FP32 numerical pass over unchanged state. Written resources still use the
formula's final reference and declared numerical protocol.

Streamed embedding operations gather only the encoding-aligned source row groups
needed by a bounded token chunk, reuse one prepared import/kernel geometry, and
scatter to the original output positions. Duplicate tokens share imported groups.
File and composite sources fill caller-owned staging storage; source composition
must not hide proportional temporary copies from operation memory accounting.

## Conformance

- Resident and streamed executions preserve the same formula/reference result;
  their observations accurately identify different physical conditions.
- I/O, temporary peaks, conversion and completion cannot disappear outside the
  complete measurement boundary; aliases do not inflate allocation counts.
- Kernel restructuring retains comparable formula history and invalidates exactly
  its dependent observations, without claiming summed children are parent latency.
- The agent can discover production scopes without compilation, measure typed
  occurrences, retain/replay boundaries and cancel safely through one execution API.
- The model TUI displays published evidence with no execution callbacks, workload
  prerequisite, fixture derivation or live device.
- Paired experiments alternate explicit prepared candidates on identical boundary
  inputs; compilation stays outside samples and mutable state resets every time.
- Historical imports preserve chronological latest/best results and reject immutable
  conflicts atomically. Cross-host and unpaired timings cannot become paired deltas.
- Session evidence reuses the benchmark's actual recipes and exact native counters;
  absent formula attribution and token equivalence remain explicit.
- Unavailable analysis does not erase valid timing; numerical failures are retained
  only as unqualified evidence and cannot enter best-correct results.
- Fast feedback is demonstrated with an actual changed operation. Whole-model
  prefill/decode/serving gates remain necessary for integration, not each edit.
- A populated production formula hierarchy with meaningful resource models and
  independent parent/child measurements is required before declaring this system
  complete or using it to resume kernel tuning. Infrastructure tests alone do not
  establish that capability.

The active implementation work plan determines validation gates. The current
replacement is written in full before its consolidated validation gate; the future
rapid measurement workflow is not permission to validate partial migration blocks.
