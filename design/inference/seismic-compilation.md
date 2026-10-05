---
applies_to:
  - inference/seismic/**
  - inference/solver/**
  - inference/engine/**
---

# Seismic compilation

The ordinary compiler artifact progression is:

```text
CheckedModule -> LogicalEntry -> RefinedCandidateFamilies<T>
             -> CandidateDomain<T>
             -> preparation-owned candidate executables
             -> CandidateEvaluator -> SelectionPolicy
             -> exact finalization -> PreparedKernel<T, H>
             -> WorkflowGraphDraft<T, E> -> BoundWorkflowGraph<T, E>
             -> AdmittedRun<T, E> -> SubmittedRun<T, E> -> Completion
```

Preparation retains its domain and candidate executables throughout evaluation.
Finalization publishes an immutable result and may leave preparation alive for
explicit continued search. There is no other semantic layer, no
realization subsystem, no independently sealed strategy, dataflow,
placement, kernel, consequence, or occurrence artifact, no reference
fallback, retry compiler, compatibility route, greedy selector, or backup
implementation. Private algorithms may normalize, infer, schedule, or emit,
but they never introduce a public or cross-crate artifact that restates the
program.

An explicitly selected top-level native implementation is a separate compilation route, not another
compiler progression:

```text
CheckedModule -> LogicalEntry -> InvocationContract
             + embedded native source + authored launch
             -> NativeKernel<Entry> -> NativeGraphPlan
             -> BoundNativeGraphPlan -> ReadyNativeGraphRun
             -> Metal completion
```

Native invocation reuses the checked entry contract and public tensor runtime but constructs none of
`RefinedCandidateFamilies`, compiler kernel IR, `CandidateDomain`, `SelectionPolicy`, `ExecutableVariant`,
`PreparedKernel`, or portable workflow artifacts. It has no solving, duration model, candidate
selection, retry, or fallback. Its only search is tuning, fast enough to run at program
preparation. Entry-wide declarations use a time-bounded local search of the author's declared parameter domain, minimizing one
weighted device-measured cost (`Σ weight × median`) over the points it times. The consumer
describes its points in ascending estimated cost and supplies their inputs through a source that
the tuner asks for a point's inputs only when it needs the point, with the time building them may
take; the source refuses a point whose building is predicted not to fit. One tuning call spends
the consumer's whole allowance: it builds the points in order, running each point's reference and
validating and measuring the defaults there, admitting a point while that work stays within the
plan's admission time, each point predicted from the previous one: its building and validation as
measured, its invocations at the defaults' measured device time scaled by their cost ratio. A
census (`Strategy::Census`) admits only the required points (the cheapest point when none is)
and returns the defaults' measurements, from which the consumer divides its time among units; a
later search of the unit given that census (`Strategy::Censused`) reuses those measurements and
admits further points within its admission. A point
the consumer marks required may take the plan's required time instead; when the required points
cannot fit, the tuner keeps the defaults without searching (`Unaffordable`). A point not admitted
is not timed and folds its weight into the largest admitted point of its class (or the largest
admitted point). Every candidate must pass numerical validation at the timed points before
repeated timing. A choice other than the defaults is then validated at the points not timed,
whose inputs are built only for that; a choice that fails there, or whose validation there would
not fit the allowance, gives way to the defaults, which must pass there too unless they are
themselves the reference (a native-default reference, against which they pass by construction).
Every chosen configuration passes at every point. A parameter read only by launches inactive at every timed point keeps its
default: nothing measures it, and no validation runs its code.
Each parameter's values are ordered numerically; neighbours differ by one step in one parameter. The search
evaluates the defaults, then, as its time allows, one start per other value of each `form` parameter and any
additional start configurations the consumer names. Each form start is the admissible configuration with that value nearest
the defaults, unless the consumer supplies an admissible start for that form value; the hint then uses that form's first
measurement slot. From the cheapest start of each form, cheapest form first, it repeatedly forms every
unvisited neighbour of the current configuration as one parallel batch, measures each, and moves
to the best while it improves by more than ε; a form's own defaults can be far from its best, and
a single descent from the entry's defaults rarely crosses into another form. Then, at each local
minimum, it restarts from the unvisited configuration farthest from everything visited. It stops when the space is
exhausted, R consecutive restarts found nothing better, or its exploration time ends: the consumer's
allowance less what confirming the finalists found so far (each estimated from its own measured
samples at the timed points) and validating the choice at the untimed points (estimated from the last timed point) will
cost. Time is checked before each
configuration. Configurations the device cannot form or run cost +∞. Of the K cheapest
configurations, those within 10% of the leader's cost (the spread at which confirmation already
rejects a configuration's samples as unstable; farther behind, noise cannot reverse the ranking)
and the defaults, the reference of the confirmed costs, are then re-measured with more samples across every timed point, alternating round by
round, and ranked by those full-workload costs; the defaults rank first unless the leader beats
them by δ. Defaults with no measured rival have nothing to rank against and are not re-measured.
Given an evaluator's costs and where its time ends, the search is deterministic, so a recorded
evaluator can replay it. Measurement is device time: a
point's calls are placed once and calibrated so one sample covers a minimum device time. Each
sample completes before the next is submitted, since a device queue overlaps independent work and
back-to-back samples would each include the other's. A kernel's first pass pays its first-use costs
and is never a sample; a later point's calibrating pass of the same kernel is one when it already
covers a steady sample (2 ms), and a point whose sample is that long takes one sample rather than
the requested count, because a sample's fixed jitter is then a small fraction of any ranking margin.
A device idle for more than 20 ms since its last timed work (forming a batch of configurations
idles it for tens to hundreds of milliseconds) runs at a lowered clock, which read a
configuration's time up to 4× too long; it is kept busy until its speed stops changing before its
next sample.
Partition projects every admissible configuration's parameter values once and decides each point's
active launches once per distinct value of what the launch conditions read. The consumer then prepares the chosen
specialization explicitly. The first initialized timed execution also supplies numerical observations,
consumed before any output storage is reused. Pooled result buffers start each observed
execution from the same zeroed storage as a fresh call, so missing writes cannot inherit a
preceding candidate's correct output. Reference execution, initialization, readback and
comparison are outside the device timing interval. A cold execution remains calibration only;
an already steady first sample is retained. A failing candidate is excluded immediately.
The numerical reference is explicitly chosen by the caller: the required portable source
construction under Exact with zero search, or the declaration default native specialization.
There is no automatic fallback between references. Native-default agreement cannot detect
shared implementation defects; neither kind of observation establishes compiler applicability. Native tuning applies the
caller's bounded or exact policy to every result and writable input subject using the compiler's
comparison rules. Discrete state stays exact. Mutable cases provide initialization for equivalent
pristine reference and validated candidate invocations; timing repetitions that follow are
batched without re-initialization, identically for every configuration. A case may declare the
leading rows each writable parameter's entry writes; validation then observes exactly those rows
of it, and observes an undeclared writable parameter whole. Callers keep at least one case per
tuned entry whole so writes outside the declared rows are rejected.
The default receives the same validation as every candidate. The first fully passing candidate
establishes the timing anchor; if none passes, tuning returns a failure with the observed exclusions.
A default that failed validation or measurement cannot be selected as an implicit fallback;
a default that passed both but whose timing re-measurement was unstable stays in the ranking,
costed at its fastest sample at each point (a time it reached; a sample cannot read faster than
the kernel runs), so a confirmed leader replaces it only by beating it at its best by δ, and
otherwise it remains the choice: timing noise is not a numerical verdict, against the default or
for it. A default whose re-measurement failed outright remains the choice. The defaults' measurement and evidence from
admitting the points carry into the search.
Seismic does not reuse stored results: the consumer keys and stores them, by what a result is
valid for (device, implementation, numerical policy, served shapes), and names what case inputs
are generated from, so reuse never reads them back. A timing key alone never establishes
numerical agreement. Old Boolean-only records cannot authorize selection.
A consumer may tune the same native entry for disjoint served workload classes as separate
units, then bind each prepared choice into the corresponding exact graph class. Each unit has
its own search identity, measurement points and validation; the prepared choices share the
entry's checked invocation contract.
The distinct public handle makes direct-only use structural.
Launch-scoped declarations use a factored search. The checked declaration gives each parameter
its launch ownership, including explicit entry-parameter reads by kernels that are absent from
launch geometry and activity conditions. Active launches, shared parameters and joint `where` restrictions
determine the groups that must be measured together. The tuner measures every candidate in each
group at the points where that group contributes work, evaluates boundary parameters that move
points between launches, then confirms each group's shortlisted candidates against its defaults
before assembling one choice per group. An independent launch group outside the consumer's
served points keeps its declared default. Each launch is formed up front as one program holding
all its code variants and held for the whole run, so a candidate forms nothing and supplies its
runtime geometry. The search runs within the consumer's allowance, less what confirming the
assembled choice and validating it at untimed points will cost; when that time ends, unmeasured
groups keep their defaults, an interrupted group's already formed candidates are measured and
ranked, and the tuner skips further group confirmation. The assembled choice is
remeasured and validated as a complete invocation against the selected reference before selection. If the default fails, a passing seed must first be found; unmeasured group defaults cannot themselves authorize a result. The assembled
candidate and the passing seed are measured in shared sample rounds, so clock drift
affects both together.
Native programs are formed uniformly. A caller renders a program: source text and the entries it
needs, with any launch constants. Code variants of a launch are template instances of one source,
so they share its text and differ only in their entry. Each backend's toolchain compiles a program
in one pass into executable entries. Each opened device owns one program former, which keys a
program by the toolchain identity and its text, serves it from a live program of the same text
that holds every requested entry, and otherwise has the toolchain compile it; independent programs
compile concurrently. What is kept between processes is each toolchain's choice: CUDA keeps the
CUBIN with each requested instance's lowered linker name, Vulkan keeps each entry's sealed
SPIR-V, and Metal keeps nothing, since the OS caches its compiles. A kept entry that no longer
loads, validates or holds the requested entries is compiled again and replaced. CPU kernels are
compiled into the binary and render no programs.
Seismic performs no file I/O for artifacts and knows no cache locations. An embedder that keeps
them between processes passes an `ArtifactStore` when it opens a device; the store sees opaque
namespaced, content-addressed bytes.
NVRTC is an owned dependency of the CUDA backend, loaded from exactly one directory: the
installation's `runtime/` beside `bin/` (`<executable>/../runtime/`), where the release, the local
development installation and every test layout place it together with its builtins library of the
same release. `SEISMIC_NVRTC_DIRECTORY` replaces that directory only for engine development outside
an installation layout. There is no search-path fallback: a missing NVRTC is a typed
toolchain-unavailable failure, never a lookup elsewhere.
For launch-scoped Metal and CUDA formation, each launch is its own program: the same rendering
serves preparation and the factored tuner, so a tuned choice prepares from the programs tuning
formed or CUDA kept.
For keying the embedder's
own records of tuning results, Seismic exposes a device tuning identity that includes the Metal OS
build, the CUDA driver and NVRTC release, or the CPU's detected instruction-set tier and CPU library
version, and an implementation digest over an entry's declaration and the source rendered for it (on
CPU, the compiled implementation's digest of its asset, its source root's CPU library files and the
CPU library version).
A CPU device executes on one worker pool per process: one participant per physical performance core,
the submitting thread among them, shared by every CPU device the process opens. A native submission
is one pool job whatever its launch count: the participants of each launch claim its work items in
contiguous chunks of about a quarter of a participant's share, so each participant streams adjacent
weight rows while later chunks still balance the load, and meet the next launch's participants at a barrier, waiting by spinning briefly, then yielding, then
parking. A launch uses at most as many participants as it has work items, and its participant count
is a tuning axis. The measured interval of a CPU submission starts when it holds the pool, so a
submission never measures another's work.
Standalone native calls use the same checked entry without creating a graph. Their scalar-result
slots and scratch are the prepared kernel's invocation workspace, which reports both.
Every non-empty scratch arena (the standalone arena and every graph workspace that places
anything) reserves a fixed range at its start for `sync` scratch. Nothing else is placed there;
within one graph each node's sync buffers are disjoint from every other node's, and never
lifetime-aliased. Arenas are zeroed when allocated and kernels restore their counters, so the
range is zero whenever a launch starts, whichever call or graph used it last. Reserving it in
every non-empty workspace, not only in graphs that declare sync scratch, is required: graphs of
one family share a workspace arena, and a graph without sync scratch placing ordinary buffers in
the range would leave counters nonzero for the next graph that uses them.
A native graph composes checked native entries, owns the shapes and lifetimes of
its graph-local mutable tensors, host-uploaded input tensors, intermediate results, and exported
outputs, and reports its exact storage charge and the liveness floor its workspace cannot go
below. Graph-local storage is placed by liveness, largest buffer first at the lowest offset no
buffer of overlapping lifetime holds. Native graphs are prepared for admitted model dimensions and
physical launch classes before the engine becomes ready. Workflow families may bind one shared
execution arena: a device runs one submission at a time in order, so graphs that keep no value in
workspace between runs share it; a family with a prewritten port owns its workspace. Request-dependent external state and resident tensors
are joined to checked ports while constructing an owned run, before submission. A submitted run
does not discover an absent tensor, incompatible shape, representation, or alias.
The checked entry is also the source of tensor-port and result-leaf extents for
metadata-only planning; the representation registry supplies canonical storage
bytes. Such tensor facts alone do not establish a graph storage bound or backend
formation.
For a graph whose nodes have complete checked storage facts, one parameterized
program owns the topology used by both resource certification and executable
preparation. Its backend-free resource projection takes a structural regime's
admitted classes as a union of slices, each a product of finite dimension
values. Every buffer's capacity is the exact maximum of its checked size over
those classes: a tensor's size depends only on the dimensions its shape reads,
and a scratch buffer's only on the dimensions its charge and guard read, so
each is evaluated at the combinations of just those dimensions. Guards are
evaluated exactly. Seismic places the capacities once per regime and
certifies their offsets; an exact graph seals into one certified layout only
when its checked topology and every buffer fit. The certificate charges the
declared maximum over finite native scratch tuning choices, and production
reserves that charge even when its selected choice uses less. A missing
scratch choice or an empty tuning domain is unsupported. Resource evidence
does not establish that the device can form the native implementation.
Consumers bind entries through prepared kernels or exact checked element
bindings while following one composition.
Direct native workflow nodes execute in their checked dependency order within one Metal command
buffer, or one CUDA stream submission, per workflow submission. Sealing validates each node
against its entry contract once and fixes its argument words, launch geometry and the storage
region and offset of every buffer, each at the one native buffer alignment (256 B, the alignment of
device allocations, so every placed buffer satisfies what kernels assume of a standalone call's); a
run supplies only its storage regions and the tensors bound to external ports, and attaching
checks those bindings alone. Per-run values live in storage, never in launch arguments: host-written
inputs go to the run's upload region, which is host-visible memory (Metal shared storage, CUDA
mapped pinned host memory), so writing it is a plain host write that never waits for queued
device work. A slot takes its upload regions in rotation, so a fixed sequence of runs binds the
same storage at the same position. Attached runs of different plans may be queued on a sequence
and submitted together as one unit (one Metal command buffer, one CUDA stream submission); the
device runs them in queue order exactly as if submitted one by one, and a queued run holds its
storage rather than its tensor handles, so output leases it reads can be recycled and rebound by
later runs of the same sequence. On CUDA a submission's launches are fully determined by its
plans and the addresses it binds: the device keeps the instantiated CUDA graph of each (plans,
bound addresses) key (least recently used dropped beyond a bound) and a submission with a known
key is one graph launch. A launch of a module that declares a programmatic dependency
(`SEISMIC_PROGRAMMATIC_DEPENDENCY`, compute capability 9.0 and later) follows the node before it
by a programmatic edge: it starts once every block of that node has triggered
(`griddepcontrol.launch_dependents`, which a kernel issues only after its own wait) or exited, and
each of its kernels waits (`griddepcontrol.wait`) before it reads anything but weights or what
launches before its predecessor wrote, and before it writes anything, so at most two launches run
at once and a kernel's first weight loads overlap its predecessor's tail. Standalone calls and
launch-detail traces launch individually. Graph
nodes publish no scalar results and never touch a kernel's scalar slots. A submission holds all
referenced storage through its completion. An allocation's host access orders after only the
newest submitted device use (host writes) or write (host reads): a device's native submissions
run on its one queue (one CUDA stream, one Metal command queue of serial compute passes) and
complete in commit order, and each submission commits and records its fences under the device's
order lock, so the newest fence of a kind completes after every earlier one.
Standalone native calls retain their own submission boundary. Tensor-result native calls from
different prepared entries may also form one ordered native batch. Each call keeps its checked
arguments, results and scratch alive until the batch completes; shared scratch is safe because
the device executes its launches in submission order. No scalar result crosses this batch API.
A read-only host mapping may back a shared input tensor on a capable backend.
One mapped device region owns one allocation and may supply several tensor views.
That allocation retains the mapping, and submitted work retains the allocation
through physical completion. Host writes and writable device bindings to these
tensors are refused. A backend without the mapping capability
uses an owned upload instead.
One state slab is one owned allocation with disjoint, aligned component regions.
Each component view shares that allocation and carries its lifetime through
submitted work. Slabs and their address table share one hazard domain, so a
submission records one fence for the table and all its slabs regardless of
slab count; direct host access and release of any slab observe that fence.
A fixed address table names slab allocations by stable index;
its entries contain the component region's device address and are changed only
after prior uses of the table complete. On Metal, each slab remains in the
command queue's residency set until its allocation is released. A queued
run keeps its slab placement bound until it receives a submission fence, and
the fence then orders table writes after physical completion. A slab
slot may be reused after release, but its old allocation
remains charged until every view and submitted use releases it; removing a slot
alone gives the owner no memory credit.
Canonical upload tensors can be filled from a host reader in bounded chunks.
An incomplete read leaves the tensor unpublished; the caller cannot bind it
as a native input until the whole physical byte range has been written.
A device's submission trace (measurement only, one active at a time) records every native
submission's host encode interval and its device interval on one host clock. At launch detail
each launch is encoded in its own timestamped encoder (Metal) or between recorded stream events
(CUDA), still within the same single submission; that attributes device time to entries but
changes the device work, so launch-detail steps are never production timings.

## Principles

1. One fact, one owner, one representation. A fact that affects legality,
   selection, resources, numerics, layout, or execution is owned by exactly
   one artifact; other layers consume it by reference or as a derived value.
2. Invalid compiler states are unrepresentable where the semantic category is
   known: private fields, opaque scoped ids, typed builders, non-empty
   collections, refined enums, consuming transitions. Validators do not
   compensate for open structs.
3. Alternatives are closed: an implementation contains its schedule, kernels,
   transfers, storage topology, constraints, and numerical transfer. Prediction is
   derived from that closed structure; factories do not own timing estimates.
4. Candidate evaluation owns search, realization requests, retention, and
   selection. Its only semantic output is a non-empty retained candidate set
   plus a total deterministic invocation-to-candidate function. The output
   does not reveal whether evaluation was analytical or measurement-based.
5. Native compilation and reconciliation are demand-driven after the domain
   is sealed and before a coordinate is retained. They consume kernel-affecting
   choices, do not redesign, and cannot reject a retained candidate later for
   a planning fact.
6. Runtime executes; it does not prove. It never discovers an inconsistency
   between compiler artifacts.
7. Errors model reality; panics model bugs. A large family of panic sites is
   itself an architectural defect.

## Artifacts and authority

| Artifact | Owns | Must not own |
|---|---|---|
| `CheckedModule` | source semantics, types, effects, canonical bodies, lowering declarations, top-level native asset references and launch expressions, capability requirements, stable identities | target decisions, schedules, allocations, native code bytes |
| `LogicalEntry` | monomorphized entry semantics, `CallSchema`, `EntryDomain`, canonical operation graph, provenance, the entry's expression arena | placement, algorithm selection, native limits |
| `DeviceDescription<T>` | immutable device-wide compatibility, capabilities, hard limits, memory rules, toolchain modes, numerical environment and target facts | compiler registrations, native handles, measured rates, selected plan |
| `CompilerRegistry<T>` | compiler policy: structural factories, lowering registrations, launch rules and emitted-intrinsic coverage | device observations, native contexts, analytical coefficients, executor state |
| `RealizationRegistry<T, H>` | opaque native handles keyed by preparation-local formed-instance identity and canonical request reuse; candidates share their reconciled ordered native set through materialization | domain membership, performance models, planning decisions |
| `RefinedCandidateFamilies<T>` | one universal structural family, optional optimized families, finite axes, construction report and one `ClosedExecutableIr<T>` per family | native handles, performance models, solver state |
| `CandidateDomain<T>` | invocation domain, every structural family, finite hierarchical choices, canonical coordinates, and one authoritative constraint relation | native descriptions or handles, evaluator identity, scores, search policy |
| `SelectionPolicy` | one total deterministic `SelectionFunction: Invocation -> CandidateIndex` whose non-empty operand list solely owns candidate IDs and order; finalization resolves those IDs against preparation-owned retained candidates | entry metadata, diagnostics, native handles, evaluator method, estimates, measurements, solver services |
| private preparation | structural domain, reusable executable candidates, native resources, invocation context and resource accounting | search strategy, implicit retention decisions |
| `ExecutableVariant<T, H>` | one materialized structured schedule, opaque native kernels, guard/layout evaluators, binding table, identity and assessment | candidate alternatives, performance estimates, logical program, solver state, public kernel enumeration |
| `PreparedKernel<T, H>` | call schema, target domain, non-empty covered portfolio, deterministic selector | compilation logic, uncovered domain, inter-call scheduling |
| `BoundWorkflowGraph<T, E>` | selected variants, closed output descriptors, dependency topology, access hazards, lifetimes and complete symbolic resource requirements | reservations, allocations, submission |
| `AdmittedRun<T, E>` | one whole-graph reservation transaction, physical bindings, persistent leases, access permits and opaque submission ownership | binding, selection, replanning |
| `SubmittedRun<T, E>` | native completion owner plus every retained admission resource | allocation, policy evaluation, early resource release |

Executable kernel, schedule, storage and representation definitions have one
shared IR owner. Closure derives each launch's local layout, scratch reservation and ABI
requirements together from its normalized structure and immutable target rules. Candidate
families cannot supply or replace separate resource tables. Reservations over lexical loop
indices take the maximum across those iterations; execution retains the exact per-iteration
layout. Runtime branches reserve both possible arms without evaluating content predicates.
Every invocation-determined allocation reservation, including storage acquired
at a reached schedule action, must fit the target's fixed per-allocation and
index limits throughout the accepted invocation domain. A callee's imported proxy of
a caller-owned view is not a reservation: the caller's own reservation carries that
obligation, so the callee never restates the caller's reached geometry. Execution-dependent
reached sizes and aggregate available capacity retain their planned runtime
capacity-failure behavior. The IR's coordinated
construction API owns scoped identities and child import; consumers cannot mutate
closed artifacts or rebrand handles. Scalar and quantity
bindings carry exactly one realization: an invocation scalar, a natural expression,
a private exact host quantity, or a published fixed-width slot. Calls carry that realization unchanged. A callee constructs its result
publications with the caller's destination symbols; import transfers local slot ownership
without changing expression identity. Forced destinations must have the same symbol, sort
and storage type. A fixed-width scalar publication retains its source dtype;
private `Index` and `Integer` publications use exact natural or integer quantity slots,
including range endpoints. The publication kind determines expression sort, SSA type,
byte decoding when applicable, and final-result category. Naturals never pass through source U32 storage. Cached native artifacts retain only
physical result kinds; each executable launch owns its semantic destination slots.
Private mathematical quantities remain exact through reached host evaluation and
loop carries. Conversion to a fixed-width kernel or publication slot occurs only
at a typed boundary with a proved representable value; a host calculation is not
silently performed as an I32/U32 kernel operation. Reached source failures retain
their source failure identity, and host work remains part of whole-entry cost.
Scalar bit reinterpretation transports complete equal-width numeric payloads.
Vector reshaping and Boolean packing require separately defined lane and value
semantics; equal storage byte counts alone do not admit a bitcast.
Kernel arguments and schedule expressions therefore continue to refer to
the same value after import, without a parallel symbol-substitution protocol.
Refinement constructs those artifacts without timing. The independent estimator
reads them through a backend vocabulary contract that requires no native service.
Native formation/reflection uses a separate service contract and explicit live
context; immutable Metal target facts contain no device handle. The preparation
orchestrator prepares each candidate on demand before it can be measured or selected.
Each successful native formation has a distinct preparation-owned instance. A
request-cache hit reuses that instance, while equal digests or reflected facts
alone never merge separately formed handles. Evaluation and finalization retain
the exact instance they assessed; sharing a different instance requires checked
work and objective equivalence under the same execution scope.
The publication manifest binds the complete reflected description to the
ordered formed-instance sequence. Separate fresh outcomes may share a semantic
implementation digest while retaining different reflection and handles.
Within one preparation, request and candidate caches key materialized family
instances and exact ordered active physical-choice/value pairs. Structural and assignment digests
remain semantic labels, not equality tests that can merge different resident
requests after a hash collision.
Checked function labels hash a domain-tagged whole-module semantic hash and
definition index. The module hash covers the compiler-semantic and registry
versions and every length-delimited source path and text, so a changed callee
file changes a caller's label as well.
The checker semantic version changes when this identity scheme changes, so an
older checked bundle cannot claim the same versioned module identity.
Portable construction choices compare an exact checked-program subject
(canonical source set and validated compile-time element bindings), source
definition ordinal, body label and mapping. The lowered semantic program
retains that subject, and each lowered function retains its source ordinal.
Root resolution checks the exact subject and ordinal before choosing an owned
function ID; call locations also retain their source ordinal, node path and
occurrence. Coordinates remain restartable across independent checks of the
same source and bindings. Compiler-semantic and registry versions are fixed by
the current build; cross-version persistence needs an explicit versioned
subject. A digest may accelerate lookup but cannot replace exact equality.
Native request identity also includes every explicitly supplied descriptor field,
including whether a field was omitted and its default took effect. Matching
reflected limits do not establish that two requests have the same formation
route or retained image. A separately formed instance remains distinct even
when its request fields, reflected facts, or executable text agree.
Completed construction paths retain separate structural families even if their
digests agree; the domain never substitutes one family's checked constraints or
outcome relation for another on that basis alone.
The solver indexes these families by their distinct registered candidate
entries, so equal structural digests cannot eliminate one during enumeration.
When an image is retained as native-formation evidence, the retained handle
preserves the exact submitted payload and accounts for its host storage.
For a source-only formation route, the handle retains the exact submitted
source and entry while treating the compiled native image as unobserved.
Native reflection and publication use the same handle on which post-load
configuration was applied.

Only the checker and the validated bundle decoder construct a checked module.
Preparation owns its structural domain, expression arena, native compiler/context,
artifact registry, reusable candidate executables, and resource accounting. The
evaluator inspects the domain throughout search and requests candidates by canonical
coordinate. Repeated preparation of a resident coordinate retrieves the same
candidate. Deterministic rejection, temporary resource deferral, and infrastructure
failure remain distinct; deferral never poisons candidate identity.
An explicit fresh-formation request may evaluate a second native outcome for
the same physical coordinate. It creates new formed-instance identities and a
new preparation candidate without replacing the ordinary cached admission.
Its native work consumes the applicable preparation budget, and selection
refers to the particular retained outcome.
The selection policy carries exact preparation candidate IDs in selector order;
finalization checks each ID against its corresponding retained executable.
The published executable retains the same ID for later observation binding.
Equal semantic labels cannot exchange two fresh outcomes in a policy.
Feedback observation requests carry that preparation candidate ID alongside
the executable, so measurement consumers can bind samples to the exact
formed outcome rather than grouping them by a semantic digest.
If formation fails after earlier kernels of the same candidate succeeded,
preparation charges those successful formations before returning the failure;
the retained request entries remain available for a later retry.

Native realization consumes the domain's checked candidate selection directly. Its family and
coordinate are not independently supplied and checked again. Cache identities are derived from
that selection; they do not establish its validity.

The invocation contract owns the entry/module identity, shared call schema, dimension inference,
parameter validation and invocation domain together. Preparation compiles it once and shares it
with executable bodies and published kernels. Invocation validation accepts this contract rather
than an independently supplied schema. Candidate observation obtains the contract from the
candidate executable itself.
The same invocation boundary validates each actual tensor descriptor's representation,
rank, axes, affine stride footprint, byte range and alignment before schedule execution.
Physical IR owns the concrete footprint rule used by compiler admission and workflow
binding. The bound allocation must contain that validated range; execution retains the
actual descriptor, including noncanonical strides.

Candidate lowering fixes one implementation's choices and compiles self-contained
expressions while the domain remains open for further search. Candidate bodies do
not contain evaluator scores or final evaluation provenance. Shared native resources
and the exact executable body are retained through trials and ordinary execution.
Compilation produces an execution-safe candidate, which cannot enter a selection policy.
Construction-owned numerical applicability derives its accepted scope and explanation together
and directly constructs the selectable candidate. A required source construction is applicable
through its actual operations and selected children; an alternative without an established
whole-entry relation remains unresolved. There is no unresolved selectable candidate, operation
that installs an independently supplied guard and assessment, or empirical promotion of scope.

The evaluator owns retention and its total invocation decision. Checked policy
construction verifies preparation-scoped candidate identity, general coverage,
applicability, and resource fit, including selector storage. It never silently trims
a portfolio. Finalization packages that exact policy with already prepared native
resources and the call contract, without compiling or making another selection.
Published kernels own their resources independently of preparation and remain
immutable when explicit continuation produces a replacement result.

Analytical evaluation owns its model and solver dependencies; feedback owns a
controlled observer implemented by runtime. The observer runs requested candidates
through the shared selected-executable admission/completion path, using private
inputs and reset state. It does not choose candidates or grant numerical eligibility.
No fake total policy or second execution representation is introduced for trials.

Workflow planning resolves external and producer-result arguments before binding. One native
binder validates those arguments against their entry contract, selects the policy variant or
checks the requested candidate's execution scope, and derives its effects and resources.
The bound node's fields are private to workflow binding and planning; other production callers
cannot assemble selection and descriptions separately. Tensor descriptors preserve their actual
device identity through external arguments, outputs and views. Binding never substitutes the
destination device for an input tensor's device. Ordinary calls and isolated candidate execution
feed the same post-binding insertion operation.
Controlled observation compares its requested metadata with a read-only projection of the
admitted execution's actual bound values before submission. It does not infer and validate the
same arguments again through a separate observation path. A mismatch drops the unsubmitted
admission and its resource ownership.
Insertion owns dependency edges, access hazards, output resources and lifetimes. The planner is
parameterized by its execution result and error, not by one particular policy implementation.

## Identities and expressions

Every semantic identity is an opaque arena index with a crate-private
constructor; region-local identity carries its region. Stable identity is
content-derived and is the only identity that crosses a bundle boundary or
enters a cache key.

One hash-consed typed expression DAG per entry (`Nat`, `Int`, `Bool`,
`Scalar<T>`, `Duration`) drives solver constraints, partial evaluation,
applicability guards, layout, geometry, allocation sizes, numerical bounds,
and modeled duration. Its free symbols are call dimensions, call scalars, target
constants, finite decisions, loop binders, and schedule scalar slots. There
is no string symbol, sentinel, or second formula language. Integer semantics
are mathematical; runtime representability restricts the target domain
rather than wrapping.

The one DAG has two non-interchangeable authority wrappers. `PlanningExpr`
contains only finite decisions, exact Boolean/table/linear structure, and
target constants accepted by the complete solver adapter. `InvocationExpr`
is the total checked evaluation language for call-dependent products,
division, remainder, alignment, folds, guards, geometry, and layout. Planning
expressions embed into invocation expressions; invocation expressions never
enter the solver. Raw solver assignments are private and become
`FeasibleAssignment` only after direct evaluation of every immutable planning
constraint.

## Machine contracts and capabilities

Analytical model implementations are trusted backend code. The service-vocabulary macro generates
its enum, complete enumeration and stable-name match from one declaration. An empty marker trait
does not establish exhaustiveness or macro provenance and is not part of model admission.

Catalog discovery only enumerates unopened physical devices. Opening a Metal
device creates the production service/queue immediately and derives the
device legality description. Hardware characterization is a separate,
analytical-evaluator dependency: a fixed, versioned probe manifest is compiled
once by a narrow native adapter, acquired as one aggregate raw-observation
bundle, and interpreted by a pure certifier. Candidate-domain construction,
non-analytical evaluation, direct native calls, and runtime execution do not
depend on characterization.

The device description is assembled once from backend revision, hardware identity
and device-wide limits, driver and toolchain versions, dtype support, the
numerical environment, and the static capability registry. It contains no
fact whose truth depends on a particular compiled function or pipeline.

Every numeric performance fact is either queried, derived by a sound documented
physical rule, or measured by a fixed primitive probe on the exact opened
device. The target-closed Metal cost program owns the finite fact vocabulary.
The renderer and analytical demand traversal consume that same program; neither
may reconstruct performance-bearing work from a broad semantic class. Measured
facts bind their raw observations, probe and method identity, endpoint,
environment, uncertainty, model version, and complete target identity into the
certified-profile identity. A separate stable compatibility identity contains
only legality/codegen facts and keys native artifacts; timing evidence does not
invalidate reusable native code. Prepared selection is never reused under a
different evaluation identity. Candidate implementations are never benchmarked
to create analytical facts. There are no calibrated corrections, fitted
candidate curves, arbitrary weights, guessed defaults, copied values from
similar hardware, nominal-peak shortcuts, or unknowns represented as zero.

Characterization constructs a complete immutable profile or no profile. Its
successful type has an infallible, exhaustive fact projection and contains no
optional required parameter in the installed service vocabulary. Acquisition
failures are aggregated across the fixed batch; certification is pure and
replayable from the retained raw bundle. `ExecutionProfileParts<T>` owns the
exact `Arc<DeviceDescription<T>>` from which its observations were acquired,
and `AnalyticalEvaluationContext<T>` consumes those bound parts with one model
definition. A concrete analytical evaluator cannot be installed until this
profile exists. The production profile boundary is structurally closed for the
built-in service vocabularies, but its physical formulas and evidence remain
unqualified until the independent estimator and characterization gates pass.

Kernel-affecting decisions are fixed before native formation. Demand-driven compilation
produces an unusable `NativeKernelCandidate`; reconciliation consumes it and
authoritative reflection to construct `NativeKernel`. Its contract records
the actual ABI, launch domain, pipeline/function limits, static local memory,
register and spill usage where exposed, cooperative requirements, numerical
mode, service footprint, and compatibility identity. Unknown legality or
selection facts are not represented as zero and prevent admission of that
native implementation. The active evaluator requests formation only for exact
canonical coordinates, and reconciliation completes before any such coordinate
enters `SelectionPolicy`.

Native artifacts contain ABI ordinals and physical types, never semantic schedule-slot identities.
An executable launch owns the mapping from native result ordinals to its actual scalar destinations,
just as it owns argument bindings. Native-code deduplication therefore cannot carry another launch's
result destinations into execution. Every backend publishes through the launch's mapping; it does
not recover semantic destinations from cached native metadata.

Host-written launch input bytes remain owned by their native reader until completion.
Metal binds and retains the same physical buffer range in its submission state. Before
rewriting overlapping bytes it completes those readers; independent ranges remain
issuable. Logical allocation ordinals are not physical identity across executable variants.
Reuse consumes the existing reserved storage rather than allocating unplanned versions.

Every emitted command, physical primitive, memory relation, synchronization
operation, and intrinsic is visible in the target-closed cost program or in the
closed workflow lifecycle model. The Metal physical vocabulary includes every
finite execution regime required by its formulas; it has no optional, default,
catch-all, or unassessed branch. A backend that cannot close the program or
construct all of its required facts cannot install the analytical evaluator.
Capabilities are typed
intrinsic families; a backend advertises a signature only when the same
registration provides its typed lowering, resource rules, and native
emission. Registration is sealed at compiler initialization; an inconsistent
registry is a startup panic. Native compilation is forbidden from returning
an unsupported-capability or resource result for anything the profile
represents.

Primitive measurements retain their workload identity, timer resolution, raw
observations, ordering, digest, endpoint, environmental controls, compilation
time, execution time, and total acquisition time. Qualification freezes the
profile before measuring separate held-out candidate families. Qualification
observations never feed back into the profile or evaluator. Accuracy and
ranking criteria are versioned and selected before held-out evaluation from
measurement noise and candidate decision sensitivity; the architecture does
not prescribe fixed percentage thresholds in advance.

## Implementations

Refinement factories, portable and backend-specific, receive a semantic
function, pure target rules, the shared arena, the precision policy, and
core-owned builders. A factory may decline before construction; once
construction begins it returns a closed candidate family or a real preparation
error. Calls are resolved during construction: every applicable child family is
spliced under an explicit finite decision, composing constraints, lifetimes,
transfers, provenance, and effect ordering. No call survives into a schedule.
`RefinedCandidateFamilies` owns one universal family, optional optimized
families, their finite axes, and an honest construction report. It owns neither
performance models nor native handles. Structural identity excludes timing
facts; the later evaluation identity invalidates predictions.

Kernel construction requires only the backend intrinsic vocabulary and its
numerical/resource rules. Native compilation and execution services are separate
requirements. Prediction consumes the closed IR and a read-only execution model;
its service algebra has no compiler or device dependency. Native realization,
prediction, and solver admission remain distinct operations in preparation.
Native realization occurs on demand within evaluation before execution or retention.

Factories use a sealed refinement-rule API. They cannot fabricate raw schedule,
storage, synchronization, numerical-transfer, or demand nodes. Each rule
consumes semantic obligations and produces locally valid executable structure;
only a draft with no remaining value, event, output, lifetime, numerical, or
demand obligations can close.

Kernel IR is typed by value category and representation; branches own their
joins and repeats own their carries with identical typed schemas. Global and
launch-local storage are different types; native launch bindings accept only
global views. Materialization is a compiler operation derived from use, never
source ceremony. Every tensor producer binds its actual stored or computed realization in the
lowering environment. Computed realizations capture already-bound operands and preserve declared
rounding boundaries. Consumers read this value; they never reconstruct an omitted producer from
source syntax. An addressable use materializes that same realization before consuming it. Use and
effect analysis chooses storage without replacing the produced value or changing snapshot timing. Allocation topology (representation, alignment, symbolic
bytes, lifetime, alias facts, reuse decisions) is owned by the implementation
and every resource expression derives from it once. View contiguity follows
canonical representation strides and survives composition; it does not imply
a zero offset or ownership of the entire backing allocation.

## Planning and coverage

Every compile-time decision is a finite explicit domain owned by one
implementation. A codegen decision changes emitted instructions, static local
memory, numerical mode, ABI, or native resources and is enumerated before
native formation. A launch decision changes only runtime geometry within one
closed native launch domain and may remain in the planning model. Invocation
dimensions stay symbolic. Target limits and
numerical admissibility are hard admission constraints, represented in the solver
for analytical search and enforced before selection under every evaluator.
The solver exports Boolean structure exactly, including disjunction,
negation, and reified comparison.

Modeled duration is the result of the target-closed physical execution model
over the same structured schedule, exact operation program, allocation
topology, geometry, memory/address relations, path/cohort facts, native
realization bounds, workflow lifecycle, and certified device profile as
execution. Construction accounts for dynamic launch multiplicity,
dependencies, issue resources, concurrency and residency, cache and memory
transactions, overlap, barriers, atomics, submission, synchronization and
completion. A factory cannot assign or omit duration. Proxy lexicographic
counters, empirical candidate calibration, arbitrary weights, hard-coded
timing guesses, and nominal peak formulas are forbidden. A missing regime
prevents cost-program or profile construction; it cannot become a successful
partial estimate. The model propagates correlated evidence and uncertainty and
never describes a prediction as physical proof. Data-dependent control,
addressing or contention uses a sound all-path relation or rejects closure; the
compiler never invents branch probabilities, cache-hit rates, retry counts, or
expected input distributions.

Qualification criteria are selected and frozen before held-out evaluation.
They must establish useful candidate ordering for the declared domain and
bound the cases where modeled differences cannot justify an ordering. Held-out
measurements validate the model and its uncertainty; they never fit correction
coefficients or candidate-specific behavior.
Metadata such as constants, views, and allocation
declarations cannot form launch boundaries, and structured control stays
within a launch unless a real execution or synchronization boundary requires
otherwise.

Coverage is constructional. `CandidateDomain` requires one universal family
whose type admits no residual choice, whose numerical transfer is admissible,
and whose legality is total over the independently derived invocation domain.
Coverage entailment applies after that domain accepts an invocation; it does
not erase possible evaluation failures from executable implication predicates.
Optimized families have a different type and cannot impersonate it. Refinement
and candidate evaluation own distinct budgets and typed search diagnostics.
Prepared results retain optimization completion and resource accounting, not a
coverage marker. Completion describes the evaluator's actual modeled search;
it does not assert exhaustion of every implementation the target can express.
Budget exhaustion preserves the complete domain already constructed and the
universal prepared policy, while reporting exactly which search scope was not
exhausted. It never turns a partially evaluated domain into success. Optional typed preparation ranges direct feedback optimization effort without
narrowing the accepted invocation domain or asserting a workload distribution.

Selection at invocation validates the call against the schema and target
domain, applies the evaluator-produced `SelectionFunction`, and verifies that
its `CandidateIndex` names an applicable retained candidate. The common
function is an opaque immutable `Invocation -> CandidateIndex` program: its
representation contains no score, duration, measurement, evaluator name,
report, or feedback scope. The analytical evaluator privately compiles its
minimum modeled-duration decision into that program; another evaluator may
construct the program differently without changing the policy or runtime
contract. Retained and materialized candidates contain no performance model.
An invalid index or false selected
guard after validation contradicts private construction and is a panic.

## Native formation and runtime

Each backend forms and reconciles native kernels on demand before candidate retention.
A frozen plan selects only closed kernels and binds one native schedule over
the shared structured step type; there are no backend schedule mirrors and no
late physical/native comparison. Native errors are toolchain, malformed
output, device loss, cache, and toolchain resource exhaustion only.

Runtime execution is workflow-based. Runtime state is generic only over target
family `T` and native executor `E`. The native compiler is a preparation-local
service constrained by `NativeCompiler<T, Handle = E::Handle>` and does not
enter prepared or runtime types; the analytical model is erased inside
`AnalyticalEvaluationContext<T>`. `WorkflowGraphDraft::bind` derives all
inter-call hazards from semantic event manifests, evaluates prepared policies,
closes output descriptors, and retains unresolved may-alias relationships as
binding obligations. `BoundWorkflowGraph::admit` atomically acquires one
whole-graph reservation set and constructs `AdmittedRun`. Only that owned value
can submit; submission constructs `SubmittedRun`, and terminal completion
releases resources.
`execute_variant` is the sole constructor of `ExecutionEnvironment`; runtime
cannot inspect a variant's schedule, enumerate its native kernels, or construct
an environment. A backend submission receives only the selected native handle
through `ExecutionEnvironment::kernel_handle` while executing a closed command.
There is no device-wide lock held across execution and synchronization.
`Kernel::call` is a synchronous one-node workflow convenience; production
model execution prepares at least one complete decoder-step workflow. Runtime
never infers placement, repairs a plan, retries selection after execution, or
interprets the portable body.

The explicit native route has the same constructional boundary. Its reusable workflows are built
from generated entry arguments, results, and checked shape expressions. Seismic derives graph-local
scratch, host-uploaded inputs, intermediate and exported output storage; validates external port
descriptors and alias rules; and owns bounded concurrent slots. An engine may orchestrate ordered
decoder-block workflows and state transactions without authoring numerical storage between their
checked boundaries. It supplies model topology, resident tensors, owned state claims, and request
values, but does not restate numerical tensor shapes in a separate scratch recipe or resolve named
intermediate buffers during submission.

## Public integration

`seismic-build` checks sources at build time, emits a versioned checked
bundle, and generates typed bindings (`Args`, `Results`, entry handles).
Rust consumers import only `seismic` and `seismic-build`, provide tensors and
ordinary parameters, prepare with `for_device`, and `call`. Dynamic consumers
load immutable checked source snapshots through the public `seismic` API and
prepare discovered entries against the same runtime. Only genuinely
polymorphic element representations are compile-time bindings.

The Python frontend is a thin in-process adaptation of that public API. It
accepts authored Seismic source and explicit resident tensors; it does not
trace Python, infer a computation graph, duplicate checking, or interpret source
as an execution fallback. Dynamic owned arguments are move intents until whole
invocation admission succeeds. After commitment every alias of that host handle
is invalid, including on submission failure; separately tracked views prevent
ownership transfer. Workflows retain the ordinary bind/admit/submit boundaries.
Feedback sessions own their preparation context independently of published kernels.

Filesystem consumers share source collection and native-asset capture. Checked
bundles contain source and asset snapshots under the core version and checksum
contract. Generated bindings embed the captured native bytes; reopening a path
cannot silently change an existing module. Host testing uses the numerical owner's
comparison and bounded checked-source oracle on private invocation snapshots.
Unsupported or resource-limited observations never count as passing checks.

The engine build identity covers its Rust implementation, Seismic, the Seismic
standard library and solver implementations, authored kernel and native helper
sources, and the workspace dependency lock and toolchain selection. A change to
those inputs invalidates worker compatibility and the assessment environment identity.

For the explicit direct-native route, runtime renders the canonical registry descriptors of those
compile-time bindings and of every tensor ABI leaf into the Metal source prefix before compiling
the pipeline. The rendered source therefore changes with representation bindings without adding a
second layout authority or runtime representation branch.

## Failure taxonomy

Source, bundle, target, preparation (`NoApplicableImplementation`,
`NumericalPolicyInfeasible`, `TargetDomainUnrepresentable`,
`SolverResourceExhausted`, `NativeCompilation`), invocation, and execution
errors are the complete typed taxonomy. No variant means two compiler phases
disagreed. Permitted panics are: inconsistent static registry, out-of-arena
private id, violated FFI precondition by Seismic code, unrecoverable poisoned
lock, solver witness contradicting the immutable model, and the prepared
kernel coverage invariant.

## Identity, caching, telemetry

Native cache keys are module semantic hash, entry, backend/compiler version,
stable compatibility identity, native-kernel identity, precision policy identity,
implementation/variant identity, and native toolchain identity. In-process
prepared portfolio keys additionally include the per-open execution-profile
identity. Runtime dimensions never create preparation keys. The cache stores checked bundles and executable variants; decoding
validates version, hash, target identity, and binary integrity. Telemetry is
OpenTelemetry at preparation, native compilation, selection, allocation, and
execution; it carries no legality fact back into planning.

## Acceptance criteria

- No struct literal or public constructor can fabricate a checked module,
  logical entry, implementation, native kernel, frozen plan, prepared kernel,
  prepared workflow, or admitted run.
- Every id is opaque and scoped; no map is keyed by a bare region-local
  number.
- Every runtime and solver formula references a node of the entry arena.
- Only directly evaluated `FeasibleAssignment` values freeze, and freezing
  performs no native compilation or legality decision.
- It is impossible to construct a prepared kernel with an uncovered
  target-domain point.
- An unreconciled native candidate cannot enter planning or execution.
- Device descriptions contain no pipeline-specific fact; concrete Metal and CUDA
  resource/launch behavior comes from each native-kernel contract.
- Only an admitted workflow can submit, and its selected variants,
  reservations, buffers, and native objects have one owned lifetime.
- A prepared native workflow is complete for its admitted model geometry and launch classes;
  submitting it cannot fail for a missing intermediate, incompatible tensor shape, or storage
  capacity that was already reserved.
- Backend crates contain one native schedule type instance and no plan
  mirror; runtime crates contain no compiler-consistency branch.
- The engine imports only `seismic` and its generated bindings.
- The forbidden symbols of the superseded architecture (proposal, recipe,
  algorithm label, placement enum spanning ABI and local storage, sealed
  value joins, encoded plan mirrors, workload envelopes, capacity classes,
  defect taxonomies) do not exist.


## Feedback preparation

Feedback and analytical evaluation consume the same preparation-owned domain and
executable candidates. Feedback owns point populations, proposal operators,
observations, confirmation events, and its final exact-point decisions. The
structural domain is not a timing model. Its retained checked source semantics
borrow the same expression arena for reference execution; extending preparation
does not clone an arena or create another semantic authority.

Generated entry bindings accept `PreparationOptions` and supply typed invocation
range builders. A range limits investigation, never ordinary invocation validity.
Requested method, scope, protocol, seed and operational limits participate in
prepared-cache identity. Feedback preparation does not acquire an analytical
profile. The published result's evidence identity is computed after evaluation.

Generated entry bindings expose `start_feedback` (or `start_feedback_with` for
representation parameters), returning a `FeedbackPreparation` and its first
kernel. An explicit `FeedbackPreparation` owns continuation state. Additional preparation
preserves candidates and comparison event numbers, revalidates the observer's
measurement environment, and returns another immutable kernel. Previously
returned kernels do not retain the observer or mutable search state. The
preparation borrows its opened device while each kernel owns its execution
resources independently. One-shot preparation uses a policy cache; an explicit
campaign owns its original domain and never reconstructs continuation from a
cached kernel. Feedback cache hits revalidate the observation environment.
Diagnostics
are separate from the two-field selection policy.

The controlled observer uses deterministic dense and resident packed inputs,
bounded reference execution, private state reset and ordinary selected-executable
admission. Packed construction consumes canonical registry plane layouts and
validates finite decoded values and input assumptions through its reference recipe.
Integer and Boolean content recipes sample their discrete representable domains;
they never obtain diversity by truncating a continuous floating-point distribution.
Recipe changes change observation compatibility identity.
Checked-source analysis identifies content-dependent checks, control and addresses;
these cases require successful bounded reference replay before candidate submission.
Their timing evidence remains conditional on the restored contents. Unsupported
allocation/view geometry, contended atomics and external conversion-only input
recipes are explicit limitations. Device admission precedes submission; the measured interval
starts at submission and ends at native completion. Host result decoding and
checking are preparation costs outside that interval. The observer verifies that
ordinary argument inference reproduces every requested invocation binding,
including floating-point bit identity. Experiment memory admission accounts for
native/input/restore payloads, live oracle-owned backing and view maps, and
comparison snapshots separately. Oracle reservations precede allocation and are
released with the last shared owner, including outcomes retained after execution.
This budget covers requested vector/buffer payloads; allocator overhead and
semantic environment metadata are not represented as an invented byte multiplier. Screening nominates a
candidate; fresh interleaved comparisons at two fixed content seeds authorize
promotion. Exact-point cases always retain a general full-domain default.

Discovery retains experiment intents in bounded age-ordered rounds. Read-only
artifact inspection determines the exact missing native units before each visit;
shared-setup batches contain only already pending experiments and execute serially.
Measured combined formation costs apply only to the same missing-unit set; unknown
costs stay unknown and remain eligible for exploration. Screening and confirmation
have separate observation-cost histories. Finite candidate enumeration is admitted
from measured affordability with a confirmation reserve, not candidate count alone.

Feedback continuation preserves deferred work separately from permanent candidate
rejection. Extending time does not enlarge byte ceilings or discard native-cache
identity. A native-formation ceiling cannot prevent use of an already resident
artifact. The runtime derives feedback phase time allowances from the requested
campaign duration; it does not impose the analytical evaluator's short defaults.

Invocation navigation uses a persistent bounded witness cursor. Supported coupled
integer constraints contract intervals soundly; unsupported expressions remain
unknown and use authoritative point validation. Empty-region proofs never follow
from failed random probes. Applicability-boundary probes retain broad coverage.
Invocation navigation covers typed scalar bit patterns, including legal exceptional
floats, and collapses proven integral parameter equalities before sampling. The
authoritative domain remains the admission test. Confirmed disagreement adds
subdivision work without deleting the broad coverage traversal. Refinement visits
use bounded rounds; screened donors are bounded per point, with current incumbents
and active comparisons protected. Inconclusive confirmation retains the incumbent
and stops at a finite sample ceiling. Reports identify the timing endpoint and the
conditional independent, identically distributed timing assumption.

## Lexical native obligations

Native reflection supplies obligations at each launch. The schedule owns lifting
those obligations through choices, runtime branches and repeats. A repeat-local
symbol cannot escape into an entry applicability predicate. Every executed repeat
iteration must satisfy its body's requirements; inactive choices impose none.
Runtime-data branches conservatively require both alternatives where entry-level
metadata cannot decide their condition.

Universal launch chunking uses the quotient and remainder of the original block
count, with a logical base owned by the repeat binder. Its last chunk uses exactly
the remaining blocks. It does not introduce a partial subtraction into native
geometry. Launch bounding happens before allocation, lifetime, resource, numerical,
and identity closure. Sealing construction consumes its mutable builders; a mandatory
normalization transition then consumes that sealed structure before allocation analysis
is available. These are successive ownership states of the same IR, not copied programs.
Imported bounds must satisfy the receiving target before they are retained; incompatible
normalization is rejected, never silently reused or wrapped in another chunking loop.
For constant positive participant count P, grid limit G and largest native natural I,
the chunk cap is min(G, floor(I/P)). Its construction records that count, cap and natural
width. Receiving targets check those premises before allocation analysis. Symbolic
optimized geometry retains ordinary per-axis obligations and receives no chunk bound.
Grid and physical-index constraints consume the actual geometry or its proven envelope;
no chunking flag exempts a launch from either constraint.
Closed executable schedules cannot be rewritten afterward. The schedule also owns
a proven per-axis grid envelope for uniform scratch reservation across repeats:
the minimum of the original group count and chunk capacity. Exact tail geometry
controls execution; physical participant indices address chunk-local scratch,
while the logical base affects semantic indexing. The kernel constructor issues
the logical-index binding when it emits that indexing operation. Launch insertion
requires the same kernel, extent and one-dimensional mapping; a raw ABI ordinal
cannot stand in for that binding. Structural import remaps its kernel ownership.
Padded lanes clamp their offset to the remaining extent before adding the base,
so their inactive sentinel cannot overflow before the bounds check.
Closed geometry bounds follow
from expression structure and reflected limits, never from sampled invocations.

Participant-local call lowering distinguishes portable families from backend-only
helpers. Portable calls inline the sealed reference body within the segment;
backend helpers require a unique supported body with checked applicability.
Alternative-body choices remain at ordinary call-splicing boundaries. Slices
preserve omitted trailing axes, and a reduction with a scalar semantic result
is evaluated and bound as a scalar at its definition. Scalar reductions use scalar
publication; tensor reductions use tensor publication. A sum initializes its
identity without reading an element, including when its input is empty.

Participant-region control returns its requested scalar yields and safety status
through the IR branch/loop result contract. Lowering never recovers values from
a branch-local construction map after the branch ends. Ordinary node traversal
is iterative; recursion follows structured control only. Admission-predicate
short-circuit control likewise uses an explicit evaluation stack, preserving
left-to-right partial-expression behavior without stack growth per conjunct.

Source-check status is one dense-u32 allocation and one kernel binding per
participant segment, with an independent indexed word and diagnostic site for
each check. Initialization, atomic failure publication, completion reads, and
failure reporting use that same index contract. Adding checks does not consume
one native argument binding per check. The IR's explicit branch-construction
tokens preserve kernel ownership, lexical arm order, dominance, and matching
result schemas; the closure API delegates to that same construction authority.
Metal emits nested blocks through a work stack, preserving IR order and joins
without recursive native emission per check continuation.
