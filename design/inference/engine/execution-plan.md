---
applies_to:
  - inference/engine/executor/**
  - inference/engine/kernels/**
  - inference/engine/state/**
  - inference/engine/src/execution.rs
  - inference/engine/src/planning.rs
  - inference/docs/tuning.md
---

# Numerical execution plan

Before allocating persistent device memory, a device-aware planner resolves one immutable plan
for enabled components, exact weight representations and tied storage, ordered checked-entry
program slots, method capabilities, service policy, and all device resources. The ordered program
topology is both the complete requirement set and the construction recipe. Program construction
attests every exact slot and returns typed callable groups; execution never queries a handle map or
coarse coverage class after readiness.
An optional drafter rejected during semantic admission may remain in the opened package.
Its unbound artifact contributes no weights or residency to the plain execution plan; an admitted
drafter definition still requires its matching component.

Metadata-only model assessment derives resident weight bytes and slab-rounded history and recurrent
bank bytes from this same model load plan and state layout, for a one-conversation workload at the
lesser of supported context and 100,000 tokens. It does not read weight payloads or open a device.
The recurrent fit charge includes the bank slabs needed for the accepted bank, one in-flight
successor, the lookahead successor when lookahead is enabled, and the pristine seed. State startup
includes one history slab per history domain and the zero-seed bank slab per store; later slabs
are heap claims.
Those exact model terms alone do not establish fit: prepared graph resources, workspace, startup
transients and the device's stable fit capacity are added as upper bounds derived from the same
header-only program plan, and so is the worker's host memory (the host tables, the graphs' sealed
plans and bindings, and its runtime, vocabulary and prepared programs priced by costs measured on
the backend; see [engine memory](memory.md)), so a fit result never undercounts. The graph-resource bound builds every
graph the load prepares, so a kernel call outside its kernel's domain fails the assessment exactly
as the load's kernel preparation fails, and both classify it `Unsupported`. Assessment results
are complete: Fits, DoesNotFit or Unsupported; there is no unconfirmed fit.
Speed assessment runs nothing on the device: each model's decode speed at each requested depth is
computed from its header-derived byte, history and launch demand over the device's memory
bandwidth (see [performance estimation](../../icn/performance-estimation.md)).
The composition root prepares complete Seismic workflows for the admitted model geometry and
finite launch classes, imports the target component, allocates the storage reported by those
workflows, and publishes readiness only after those steps succeed. The engine does not maintain a
second numerical tensor-shape description.
The largest service token allowance bounds a launch's rows and request slots in either phase.
Every request consumes at least one row, so prefill can admit more request slots and output rows
than the decode allowance while still staying within its token budget. The selection bound, the
decode allowance, bounds the rows a launch selects a token for and the requests a drafter launch
drafts for; every vocabulary-wide buffer is sized by it. A decode round never meets it, the
scheduler admits no prefill round beyond it, and execution refuses any launch beyond it or any
logits demand the load does not export.
Device assessment uses the allocation domain's capacity (for a dedicated device, the memory its
driver lets one process use; see engine memory), bounded by
applicable process limits and Metal's recommended working set, less the domain's planning reserve;
a dedicated device's load also fits its staged uploads in host RAM less the host's planning
reserve. Admission of new holdings uses fresh available memory observations for each used domain
and must leave headroom above the planning reserve (see engine memory). Already charged
allocations are excluded from observed availability and are not subtracted again.
The native execution path is backend-neutral: the host names a device (a backend or an exact
selector) or asks for automatic selection, which takes backends in the fixed order Metal, CUDA,
Vulkan, CPU and uses the first device, in Seismic's enumeration order, of the first backend with a
usable device (available to Seismic, meeting its backend floor, with established memory backing;
software Vulkan devices are not accelerators). Selection never ranks devices by fit or speed, so
preview, assessment and load select the same device; a numerical worker re-resolves an exact
selector in its own catalog and refuses a missing or ambiguous match. The path then executes on
the opened device's backend,
and every native entry is prepared from that backend's declarations. Preparation reports every
entry the program plan needs that lacks an implementation for the backend together, and every
error names the path and the backend.
A load and a prepare-only job share this one preparation (device selection, planning, opening the
device and preparing programs with the kernel cache), so both produce and look up identical tuning
keys; the prepare-only job ends there, while a load continues with state planning, graph sealing,
resource allocation, weight import and warm-up. The prepare-only job forms every program the load
that follows it prepares: each searched unit's canonical artifact and every entry without tuning
parameters, concurrently and before its optional search; a promoted choice publishes its own
artifact. That load therefore finds each program in the toolchain's store (and the driver's
caches) rather than forming it, one entry at a time, itself.
A prepare-only job that runs to its end records its load in the kernel cache, keyed by the
manifest's content and the tuning version; a later load of the same manifest in any process finds
that record before it opens a device and starts no further prepare-only job. The key excludes what
names the load only within a process or a boot: process-local artifact identities (the components'
files and tensors stand for them) and the device selector (a Metal registry id is assigned at
boot). The device's tuning identity, known only once it is opened, keys every unit's record, so a
load on a device without records still meets an unprepared unit and reports it. The key also
excludes where the host keeps its cache and trace, and fresh tuning.
Native entries with declared tuning parameters are tuned on the opened device during this
preparation, on the first preparation for each tuning key, whether by a load or by the prepare-only
job that follows a catalog installation: each such entry registers a tuning case
that supplies static values from model geometry, weighted tuning points over the shape classes that
entry serves (every row class of its graph path, crossed with served history lengths for attention,
including the empty history that exercises the fresh-only path; a unit whose layers all keep a
window of history shorter than the context is timed at the lengths below its largest window and at
that window, the history every longer prompt gives it, instead of lengths its layers never see, and
a search start sized from history (the attention decode forms' encoded-history slice) is sized from
that window too;
selection-row classes up to the selection bound for the readout, each retaining its own step-time
share; the prefill chunk classes divide theirs by the octaves of prompt length each serves, the
largest launch's class holding every octave from its rows to the served context, since a longer
prompt is prefilled in launches of that many rows), rotations over real resident weights of distinct layers for weight-streaming decode rows,
control tables packed by the batch builder. Every tuned entry applies the load's single α policy
to each floating result and writable-state subject through the derived error-ratio admission rule.
Production preparation forms the declaration default native specialization as its fallback artifact without creating numerical evidence. Production tuning compares each candidate on the device against the declaration's native default, the canonical kernel itself, under the admission limits below; fidelity to the portable checked body and model-level quality are evaluation's job. There is no portable or host-CPU reference. Reference observations and fidelity qualification share the original first-load allowance. Startup qualification remains a local empirical check, not a portable semantic proof or whole-model quality qualification. Every alternative candidate must pass the witnesses of each timed point before continued performance sampling; the selected winner must also pass every untimed case. An unmeasured fallback, including a terminal zero-demand unit, remains explicitly unqualified. A qualified terminal candidate requires the selected artifact's completed native-default evidence for every immutable case, including untimed cases. An explicit canonical-fallback disposition instead pins the actual formed declaration default; cold and warm loads can use it without claiming numerical qualification. Reuse resolves that same default and checks both specialization and artifact identity plus the served contract. Missing evidence never implicitly creates fallback authority. A known rejected default cannot be used as fallback. Unqualified fallback measurements never supply admitted service forecasts, and neither fallback reuse nor failure reopens the tuning allowance. Zero-demand reuse performs no numerical work. The first suitable timed
execution supplies its validation outputs. Integer, Boolean and packed-code state remains exact.
Every tensor an entry writes in place is case-owned, and its case declares the rows the entry
writes; real serving state is never bound. Those rows are restored before each reference and
validated invocation, and validation compares every result element and every written row. The
rest is input the entry only reads, so no long history is re-uploaded or read back per candidate.
One case per unit, the one with the least state, restores and compares its state whole, so a
write outside the declared rows is rejected. Timed passes after a
candidate's validated invocation are batched and run on the state they leave, the same way for
every configuration, so timing never pays a host reset per invocation. Numerical witness sets and performance rotations are distinct: all declared witness cases qualify the artifact, while only the declared workload rotation contributes sampling and confirmation cost. Input variation used for numerical coverage does not imply independent memory streams; real weight-stream rotations remain part of the performance workload. An expired validation is incomplete work within this load, never a cached rejection or a further-load allowance. Matching evidence from earlier in the same search can be
reused, but timing equivalence alone is not numerical evidence. Whole-model numerical and raw
output/parser regressions are development and release qualification, not another startup gate or
a search over kernel combinations. Local bounded policies do not claim a mathematical bound on
whole-model accumulated error.
An entry that declares parameters without a case fails preparation; there are no
engine-side default parameter values. An entry prepared again with identical element bindings
and static values reuses the load's first tuning result; its rotations already span the weights
of several layers. Disjoint row-class workloads may prepare distinct configurations of the same
entry contract; each is tuned and validated on the classes it serves, and the exact graph class
binds its prepared configuration.
Row partitions follow a native form's supported geometry rather than a model identity. In
particular, an affine Metal form that shares a history tile across four rows has a distinct
four-row objective throughout its supported head geometry; shorter and wider row bands cannot
displace that choice. The partition changes neither the served numerical contract nor the
first-load allowance.
Entry-wide
declarations are tuned during one first-load tuning allowance (60 s per preparation when
served, on any device; evaluation may set an explicit budget, which keys the stored result). Counting identifies units, served geometry, launch frequency and final stored
choices without forming programs. Canonical programs required by a fresh load form inside
the same allowance, including zero-demand units; their formation is reported separately.
Census, representative discovery, candidate planning,
optional formation, qualification, timing, confirmation and receipt publication otherwise use
this one allowance. No independent optimizer or later load adds time.

The served numerical contract and the timed performance objective are distinct. The former
contains every declared row/history/ownership domain and remains immutable. Candidate eligibility
requires exact-artifact numerical witnesses throughout that contract, including zero-demand and
untimed points. The formed canonical default remains an explicitly distinguished unqualified fallback;
its usability does not qualify an approximate alternative or establish artifact fidelity or model quality.

The workload explicitly weights prefill and decode at 4k and 64k context equally, capped
and deduplicated at the load's supported context limit, for one sequence. These are policy
choices rather than inferred traffic frequencies. Each prepared draft width contributes an
equal part of its decode metric, including plain decoding when prepared. The graph supplies
actual logical launch counts, including draft and verification work, independently of weight
ownership. Alternative implementations receive only the launches they serve.

The objective is the weighted mean of relative service time `Σ term_weight × time / baseline`.
Baseline totals are measured once over the declared tunable workload. Consequently a unit's
point weight is `Σ term_weight × launch_count / baseline_term_time`, and its contribution is
exactly additive under this objective. There is no logarithmic objective or inferred probability
of speculative acceptance. Fixed work outside the tunable workload is not measured by this
objective; whole-engine evaluation establishes end-to-end improvements.

History-dependent performance points are empty history, 4k and the layer's longest retained
history. Intermediate launch demand is split linearly between neighboring points, with window
limits applied first. This is an explicit representative-workload approximation, not a proof
that every intermediate history has linear latency. Every candidate is measured at every
selected point: no timing or ranking is transferred from a short history to a long one.
Numerical history witnesses remain intact, including intermediate histories with zero
performance demand; timing representatives do not define numerical scope.

Baseline discovery measures every demanded unit before candidate search begins. Unknown
units share discovery by elapsed effort, with declaration order breaking ties. Incomplete
discovery is reported explicitly; it never contributes invented latency or zero service to a
candidate objective. Once discovery completes, the measured baseline term totals normalize
point weights, which remain fixed throughout the search.

Search effort follows measured remaining service. Each available piece is ranked by its
virtual finish time `(effort + price) / (share × opportunity)`, smallest first. `effort` is the
unit's completed search work, including compilation even when it overlaps device work;
`share` is its measured fraction of the baseline objective. For exploration, `opportunity`
is the incumbent's remaining cost relative to its default; for a pending conversion it is
the observed lead. This is an explicit allocation policy, not a probability of discovering a
faster configuration. Neither model identities, reference-operation counts, nor the spread of
previous winners predicts a candidate's value. A unit explores distinct forms before refining
one. Resolved promotions take priority over further exploration so established gains
reach serving before the allowance ends. Other stages impose no global barrier after discovery.

Prices use this load's observed operation costs. Unobserved operations use the conservative
admission forecast until observations exist; coverage costs are shared only within the same
entry. Every admission still protects the original deadline, numerical coverage and publication
obligations. A chosen kernel must establish a lead through matched measurements of the full
performance cohort and satisfy its complete numerical scope.

The runtime owns candidate generation, artifact qualification, retained measurements and complete
pending-work quotes. The allocator owns admission, elapsed charging and deadlines. A quote states
complete pending execution and any incremental terminal obligation admitting it can create.
Completed artifact witnesses and formed programs are removed from pending work rather than
charged again. The allocator does not independently calibrate a runtime quote or price program
formation. Quotes refresh after every step or canonical dependency seal.

Unknown quotes cannot waive baseline discovery or numerical evidence. Known candidate work
is admitted only when execution, current terminal obligations and the added obligation fit.
An admitted transaction uses the original hard deadline; adapter overhead does not create a
second forecast-sized deadline. The runtime protects its conclusion reserve, and the allocator
protects the other units' reserves and publication.

Ordinary and factored search produce complete prepared candidate artifacts during exploration.
Factoring reuses keyed point measurements and generates assembled whole specializations; it does
not introduce another winner or finalization authority. The measured incumbent's legal neighbors
lead useful refinement while representatives of disconnected admissible components remain
reachable. Numerical allowance expiration leaves incomplete witnesses pending within this load;
actual numerical or execution rejection is retained as a verdict.

Finalization confirms and ranks only retained whole candidates using the same deterministic
weighted cohort and evidence as search. It does not assemble unseen artifacts or reserve work for
hypothetical group combinations. Reserves derive from actual candidate obligations. The first
rival adds one winner-coverage reserve, held separately from confirmation; further rivals do not
each reserve another copy. Earlier timing samples may survive expired confirmation, but a retained
leader still needs complete numerical coverage before it replaces the default. Canonical
prerequisites are validated as a finite acyclic dependency relation before scheduling. Sealing a
prerequisite publishes its final artifact and ends its exploration; dependent alternatives are
qualified and repriced against that artifact, not a provisional default.

A search's bytes are admitted like its time, under the room its device's platform granted when it
was opened. A point whose input build the device's memory limit refuses is priced at the build's
demand and offered again; a unit whose work cannot fit stops for want of room with its terminal
choice published. Neither fails the preparation, so a preparation that ran to its end is recorded
as finished and later loads read its records without searching.

Deadline checks apply before each interruptible operation and between submissions. A reference
call is a call of the canonical kernel, priced by its measured service. Other admitted quotes remain forecasts; submission granularity bounds their
risk, and receipts report actual overruns honestly. A reserve
never grants extra time. At the end of the allowance every unit saves its terminal choice,
including a limited or unmeasured default. Later loads consume that choice without reopening
search. A terminal choice that fails its same-key served contract is a preparation error,
not a cache miss granting another search. Served maps include the codec-derived plane
dimensions used by actual attention graphs. Incomplete evidence and negative dependent outcomes grant no further allowance.

Shared history planes live
for the whole tuning, so units reading the same history build it once. A launch-scoped declaration admits its points the same
way, and its factored search runs within the same window. A slower device or build tunes fewer
points, candidates and units, never longer; each unit's report gives its budget and its time.
The engine owns every cache, under a directory the host names (`--cache-dir`; without one nothing
is cached). It holds the program artifacts Seismic keeps (CUDA CUBINs, Vulkan SPIR-V), through
the device's artifact store, one directory per toolchain namespace, and one tuning result per
tuning key. The key is a digest over what a stored result is valid for: the tuning version, the
device and toolchain identity (Metal OS build; CUDA driver and NVRTC release), the unit, the
implementation digest (declaration and rendered source, never the directory they were built
from), the precision policy, the explicit load budget, and the labels of
the served shapes its choice was validated at. How it was searched is not part of the key (search
settings, tuning time, budget shares, workload weights, the points timed), so improving the
search never invalidates a result that is still correct. The tuning version changes only with the
maintainers' approval, when tuning has improved enough to justify retuning every model, or when
stored results can no longer be read. Every search that ends is stored, so a hit prepares the
stored choice with no measuring, validation or input construction: its key pins
everything that validation depended on. A stored choice is still checked against what is not
measured: the declaration's domain and, for a unit that serves shapes its points do not take, that
its formed kernel launches at every one of them. A tuning point's rows are one request's, so they
see one history span, while a served launch's rows see up to the store's span class (the next
power of two above `MAX_HISTORY_SPANS`), and the fused attention entries' launches take group
memory per span. Those units name every point's rows at every span class up to it, and the search
excludes a candidate that cannot launch there as it is formed. A parsed terminal result that fails either
check is an explicit preparation error; its choice cannot reopen search under the same key. Tuning inputs are generated test data and resident
weights, not a fixed corpus, so numerical evidence is identified by the case's structure, policy
and reference rather than by input bytes.
Persisted numerical policies and search weights round-trip exactly; serialization must not
change eligibility or invalidate an otherwise identical objective.
Keys are content addresses for that structural and policy identity. Each unit's terminal result,
including an unmeasured default, is published atomically as selection concludes. Incomplete
search scope does not serialize a continuation or authorize more work on a later load. An
unreadable record cannot authorize a nondefault choice; a parsed same-identity result that fails
the served contract is a preparation error, not a search restart. Stored candidate measurements
retain their configuration, point, comparison and coverage identities separately from the policy
that selected a winner. A changed α can reselect without execution only among candidates whose
retained evidence completes the required scope. Stored results are local measurements; nothing
is shipped. Tuning progress (milliseconds of the tuning time spent, reported when the census
begins and after each step of a unit; nothing when every unit is stored), retained search scope
and stop reason, total tuning time and how many units were searched or stored are reported before
readiness, or before a prepare-only job reports that it is prepared. The load also reports its
target weight import in resident bytes; no tuning or
preparation occurs after readiness. Two development
measurement tools, enabled only by the evaluation harness and never by a served engine, change this:
the executor's `pinned-tuning` build feature records the configuration chosen per entry (entry,
bindings, static values) and replays exactly those configurations in a later run, so two runs can
be compared bit for bit; its `tuning-survey` feature replaces the search of the named entries by a
survey that forms, measures and validates every admissible configuration and writes every sample,
so a search's choices can be judged against the whole space.

Numerical eligibility and speed ranking have separate authorities. The [native tuner](../../../inference/docs/tuning.md)
retains the default's outputs per measured point; each candidate's initialized execution supplies
both timing and error observations on identical inputs. Every nondefault candidate uses the
same rule: the error ratio must not exceed the load's α. The profile
carries only α; there are no declared deviation classes or separate defect-admission threshold.
Rounding-scale comparisons are inspection signals, not selection authority. The leader must
also pass every remaining required witness, with its coverage reserved inside the allowance.
If it fails, another retained contender can be checked only while that work fits; incomplete
coverage retains the actual canonical default. The [precision system](../../../inference/docs/precision.md)
defines the error model and the relations evaluation measures. Neither local agreement nor the policy
alone establishes whole-model quality; independent same-artifact evaluation remains required.

The tuner admits the actual native artifact against the native default and ranks admissible artifacts on the serving workload. Agreement with the default checks native agreement, not fidelity to the portable checked body or shared defects. Development closure must separately exercise an independent checked portable body at representative geometries before claiming that relationship. Compiler assessment, empirical native agreement and whole-model quality evidence remain distinct; see [Numerical precision](../seismic-numerical-precision.md).

Concrete optional computations retain their physical behavior. INT8 activation projections on Metal's gate/up and down projections
past 64 rows, on tensor operations with Q4_K or Q8_0 weights, quantize each activation row to int8
per 32 columns in a pre-pass and multiply the weights' stored codes on the int8 tensor operation,
folding each 32-column block under its activation and weight scales; the weights stay exact, and
every other weight format or device runs the exact form in the same launches. Its transient
scratch is the int8 operand with its scales and block sums (rows x K x 1.2 bytes) and the weights'
decoded block scales and biases (weight rows x K / 32 x 6 bytes), per call. Paired-token projections on Metal's gate/up and down projections, the plain projection of post-norm
sublayer outputs, the recurrent projection's qkv and z segments, the attention projection's
segments and the attention output projection past 64 rows, with Q4_K, Q5_K, Q6_K, Q8_0, iq4g32,
q4g32s (GGUF Q4_0), q5g32s, q5g32 or iq4g32s (IQ4_NL blocks held as stored) weights on devices
without tensor operations, round each activation row to
integer codes per 32 columns under a gain that bounds every sum for any row of the weight format,
and pack two rows into one F32 matrix operand against the weights' stored codes minus the centre
of their range, so one multiply-accumulate carries two products; the two 16-bit sums of each run
of columns under one weight scale (a 32-column block; 16 columns for Q6_K, whose rows are coded
per run) are split from the accumulator and folded in F16 under the row and weight scales, each
held over one power of two per row (applied to the F32 result) so that no input can overflow the
fold. The segments of one entry share one operand, coded under the tightest of their formats'
bounds; a segment whose format needs another column order runs the staged exact form. The weights
stay exact and the pair's top sum is exact; its low sum carries the accumulator's rounding, so a
row's result depends on the row it is packed with: the complete computation is row-dependent. Its transient
scratch is the packed operand with the rows' scales and code sums (rows x K x 2.1 bytes) and the
weights' block scales and biases (weight rows x K / 32 x 4 bytes), per call; other weight formats,
and devices with tensor operations, run the staged exact form in a launch of their own. Direct and coissued attention and the CPU's activation-quantized projection implementations are ordinary automatic backend candidates under the entry's comparison policy. Backend scheduling and registered physical representation facts select them without an extra feature profile; their numerical qualification against the native default remains mandatory. Reassociated rounded projection is also a normal candidate.

Paired-token projection is an ordinary automatic tuning candidate, admitted by the same measured
error-ratio rule as other forms. Native representation and device guards determine where it is
legal, and every selected artifact retains its complete numerical qualification obligations.
A result may depend on its
launch's shape class and prepared configuration, and different shape classes agree within the
gate's tolerance, not bit for bit. Speculative verification is therefore statistically, not exactly, equivalent to
plain decoding; acceptance over the logits a verification produced remains exact.

The resource plan authorizes persistent weights and startup state slabs, including the permanently
pristine recurrent zero seed, the device's one workspace arena, outputs that outlive launches,
structural retention slots, optional component residency, and the
qualification/startup peak. Persistent allocation follows planning. Qualification scratch is
released before readiness. Execution receives plan-issued leases and cannot allocate general
scratch outside the plan.
Per-request state and feature tensors require fitting heap claims. Retained checkpoints release
their physical charge when their final owner drops them.

Seismic composes native checked entries into prepared workflows for decoder blocks and other
numerical units. Block graphs share a plan only when their complete computation agrees,
including decoder-boundary work: widening the entry into residual streams and mixing the exit
streams for readout. Equal sublayer geometry alone does not permit sharing those boundary graphs. Its checked entry contracts derive graph-local mutable scratch, host-uploaded
inputs, intermediate and result tensors' representations, extents, alias conditions, and lifetimes.
It reports exact storage charges and owns bounded concurrent activations. A device executes one
submission at a time, in submission order, and a graph's workspace holds only its own run's
values, so every family's graphs bind the device's one workspace arena, charged once at the
largest family's workspace whatever the families and launches in flight. Workspace is placed by
liveness: largest buffer first, at the lowest offset no buffer of overlapping lifetime holds.
Metadata assessment projects a resource schedule from the one parameterized graph program that
also constructs executable graphs. A regime is the set of admitted classes that select the same
structure: the row form (the attention-decode, recurrent-chunked and routed-decode predicates the
topology code itself branches on) plus any other class field that changes node order, edges or
exports. Seismic charges every port, result and scratch buffer of a regime the exact maximum of
its checked size over the admitted classes, evaluating each size expression only at the
combinations of the class dimensions it reads, never across whole graphs. It places those
capacities once and certifies fixed offsets that production reuses for every exact class of the
regime. A class whose topology or checked size exceeds its certified layout fails preparation.
The family charge takes independent workspace, output, and upload maxima across the certified
layouts and counts each distinct bound constant once. Zero bases remain symbolic while assessing
and sealing classes; neither stage materializes a host matrix per class. Binding allocates one
zero-filled device tensor per distinct element and shape, with the same byte charge and storage
identity as an explicit all-zero constant. A checked contract that cannot be evaluated fails
assessment rather than producing a fit estimate.
Each activation holds a fixed set of host-upload regions, allocated and charged with it: one
per graph run its lease keeps in flight at once (a target step queues its embedding entry and
every block before any completes). Upload regions are host-visible (CUDA: mapped pinned host
memory), so writing a step's controls never waits for the device, and they are taken in rotation,
so every step binds the same storage per block and CUDA replays each block's instantiated graph.
Activation never allocates; an activation that finds every region still in flight is a typed
failure, not growth.
Resident imports use one-shot destinations. Metal maps source-file windows shared by ordered
imports; the transient is bounded by the largest planned source tensor plus host-page rounding.
Other backends use a one-shot staged source upload. The engine charges Seismic's reported native invocation, intermediate, and result
storage; it does not author parallel tensor recipes or look up named intermediates during a
request.

The target and head workflows cover the admitted row and history-span ladders. Row classes are
powers of two up to 32 rows (decode, verification, concurrency) and multiples of 64 above, up to
512; span classes are powers of two through the per-model bound, the largest history domain's
`ceil(row limit / rows per slab) + 16`. Blocks whose sealed workflow would be identical apart from
their layer (same geometry, weight representations and shapes, and state layout) share one sealed
plan per class and bind their own weights to it; the composition root reports the class count,
sealed graph count and sealing time. Decoder numerical
pipelines share projection results through checked Seismic result edges. Dense feed-forward owns
its activation product; attention owns the normed Q/K/V projection and one fused entry that prepares
queries and keys (norms, rotary) in place, accumulates the stable softmax and gates the values
(decode row classes use the partitioned decode entry, larger classes the streaming prefill entry);
a gated delta mixer owns its normed segmented projection, then one state entry (row-sequential for
decode row classes, chunked above) that advances its bank in place and publishes the gated rows
(each head's recurrent output RMS-normalized, scaled by the recurrent norm and gated by SiLU(z),
formed once per row and head), then a plain residual output projection that repeats no prologue;
on a backend that declares the convolved step form, the decode row classes' projection launch also
convolves each channel once and publishes the successor windows, and the state entry advances from
those convolved channels with the same bits;
a state-space (Mamba-2) mixer owns its normed projected row (gate, convolved channels and time
steps together) and runs the checked step entry for decode row classes or the chunked entry
above them over its bank's window and state slabs, then its gated group norm and output
projection; a gated short-convolution (LFM2) mixer owns its normed `B⊙X` and `C` rows in F32,
then one entry for every row class convolves them over its bank's F32 window (its only bank
component, whose rows are also its tape), gates, and publishes the successor window, then the
output projection; a block may hold a lone mixer with no feed-forward, and its output is the mixer's
residual row; routed feed-forward owns normalized input, routes and scores (ranked once by probability), and the
shared coefficient, then either the per-choice expert and shared products (row classes within the
GEMV bound; a backend that declares the shared-route entry forms the shared product, which does not
depend on the routes, in the routing launch with the separate expansion's bits, and expands the
choices alone; every choice's product has its own one-row GEMV's bits, so a backend may stream an
expert that several rows choose once for all of them, as CUDA does for classes of two or more rows;
Metal keeps the per-choice form, whose repeated expert reads its cache already serves and whose
one-row GEMVs outrun a gathered multi-row one)
or, for larger classes, grouped tables and grouped expert outputs: choices grouped by
expert into tile-aligned blocks whose capacity derives from the class, the selected-expert count and
the tile rows, so no table is uploaded per step and no host readback sizes a launch. The general
routed form (every family without that gated shared expert) owns the normalized input, routes and
weights in one selection entry (softmax, sigmoid or square-root softplus scores; a selection-only
bias; slot-order renormalization; the post-scale and per-expert output scales folded into the
weights), and adds the routed sum to a base row that already holds the residual and any shared
expert's output. A zero base serves a routed sum that is normalized alone or projected out of a
latent width. Its grouped capacity counts at most min(E, M·K) receiving experts. Seismic prepares an
exact workflow for the selected physical batch class, so a small decode batch does not execute the
maximum class width. Draft head blocks use their declared dense or routed feed-forward geometry
and the same routed numerical composition as target blocks. Header assessment includes the head's
routed weights, program entries, and class-dependent workspace before residency begins. A causal
head's residual form determines its mix/scatter entries and its conditioning width and
element. Raw residual conditioning is exported before the target's exit mix and remains
F32 through selection, prefix-boundary retention, and chained head steps. Vocabulary
readout features are computed separately. Both exports and every mix/gather entry belong
to the same checked graph topology and resource plan used for execution. Linear
projection stages use cooperative subgroup reductions for the smallest row classes and subgroup
matrix operations above them. A projection entry that reads packed weights has one
accumulator-scale port per weight, whose static extent is 0 (absent, compiled out) or the extent of
the weight's resident second-level scale: per tensor for dense feed-forward, latent, post-norm and
vocabulary projections, per expert for up-only routed experts; a routed down weight's per-expert
scales join the selection's per-expert output scales in the combine weights. Blocks share a sealed
graph and a specialization only when their scale extents agree. A decode projection, normalization prologue
included, is one launch: each workgroup reduces its few rows' norms while staging them. Larger classes
normalize once per row into entry scratch, never per output tile. A monolithic entry that recomputes normalization,
projection, routing, or softmax for each output coordinate is not an admissible production program.
Target readout preserves every demanded feature row, and projects only rows it selects: the
feature and head entries each gather their hidden rows through a row table in their own
normalization prologue, so no copy or gather node precedes them. Logits are graph locals of the
selection that consumes them; a served readout's outputs are its features and compact
selections. Selection graphs exist with and without the shaping stage; shaping rewrites the
logits in place, and a step whose selected rows all leave the logits unchanged under shaping
(greedy or unit temperature without cuts, and no penalties) samples the projected logits
directly. Each selected row carries a constraint flag; only a step with a constrained row uploads
vocabulary masks, and an unconstrained row's mask is never read. Shaping applies a constrained
row's mask before its cuts, so top-k, min-p and top-p rank and renormalize only admitted tokens
and a row that admits a finite logit never samples an empty distribution; sampling applies the
same mask to unshaped rows. Exporting full logits is a load capability that served loads do not
have: a diagnostic load declares the rows it exports, exports the logits of every projecting
class, and selects only unshaped, since shaping must not rewrite logits it exports. The head and
separate draft keep their vocabulary logits as graph locals too, their drafting classes bounded
by the selection bound.
A readout always retains its canonical packed projection. Backends may offer lossless progressive
views of eligible Q4_0 or Q8_0 matrices as optional physical capabilities. Eligibility alone never
selects them. The packed projection retains the ordinary tuner's finalized configuration in every
class. Optional qualification depends on that choice concluding first within the same first-load
tuning minute; a default projection may price its census but cannot substitute for the finalized
canonical in qualification or serving. Whole-recipe qualification compares that packed
projection and selection against Top→Refine→Exact and selection, using the same normalization,
score transforms, masks and noise. Only complete, numerically certified classes with a measured
cost improvement select the progressive recipe. A class choice identifies the projection scope,
normalization scope and whether its canonical input was already prepared. Consumers sharing a
physical table do not share qualification when those semantics differ. Plane residency is merged
by projection scope; graph class selection uses the exact consumer recipe. Missing or unfinished
evidence selects packed; full-logit and indexed policies use packed. A recipe is offered only when
each of its kernels admits a specialization at the recipe's dimensions on the device, each kernel
bound to exactly the static dimensions it declares (the refinement levels read no normalization);
an inadmissible recipe keeps the head packed, the form assessment planned, and never fails a load. Later loads consume the saved class choices without
additional profiling. Their evidence binds the exact finalized packed specialization; a changed
canonical choice falls back to packed without reopening qualification. The progressive levels preserve every row whose exact score could win:
top codes establish a lower-bound threshold, refinement tightens it, and the exact pass scores
survivors while marking other rows −∞.
Before resource admission, planning finalizes these choices: canonical packed storage remains,
optional planes become resident only when a served class selects them, and the program and graph
resource plans are derived again from the retained capabilities. Transient candidate imports
belong to the tuning authority and never silently become resident bindings. Physical candidate
weights and row-independent fixed programs are prepared once per exact storage and program
identity within that preparation; row classes share those resources while keeping independent
activations, numerical evidence and qualification. The admitted plan is
immutable, accounts for every retained plane, and is the sole import and resource authority.
Optional views must fit the same checked assessment charge and stable allocation capacity as the
load. When their added charge does not fit, finalization keeps the canonical plan and drops the
optional callable handles before forming graphs; this grants no new measurement or search time.
Live allocation still requires the ordinary heap claims.
A draft head stored in the draft component that keeps no history (`HeadHistory::Target`, the
Gemma 4 assistant) is a separate program family of the head lane, not a variant of the embedded
head's graphs. It has its own residual width, decoder blocks of Shared attention plus dense
feed-forward, an own full-vocabulary readout table without a soft cap, and pre/post projections at
the target's width (`HeadInputProjection`, `HeadChainProjection`). One sealed graph per slot class,
step count and shaping runs a whole transaction: each pass gathers the target's scaled embedding of
its token and joins it with the conditioning row (the target's final feature, then the previous
pass's normalized output projected back to the target's width), projects into the head's width, runs
the blocks with each attention bound to the planes of the target layer it reads (resolved through
the target's own sharing), normalizes, projects the whole vocabulary through the head's table and
selects with the target's selection entries. Every pass is one row per slot at the anchor's
position that sees exactly the request's accepted target rows of every stored domain and appends
nowhere. Its decoder-block entries are prepared by binding beside the target's; its graph family,
bound constants and class workspace are charged in header assessment. Such a head plans no head
state store.
A separate draft (DFlash, DSpark, DFlash2) conditions on target taps instead of the final features. A
tapped block's workflow rounds the residual entering it, entering its feed-forward, or leaving it
(the exit tap is the last block's output) into that tap's column block of a draft-input buffer
the bound target workflows own (charged with their bound constants); the
readout's demanded feature rows are then the draft's fusion (one projection of the concatenated
taps) rather than the normalized final rows. The draft runs as the head lane's drafter, one sealed
workflow per class and transaction: each draft layer's attention appends the entry rows' context
keys and values (the fusion norm as its input norm). Context injection has a distinct
projection binding with absent query and gate segments, so it allocates and computes only
K/V outputs. It selects the cache codec's append-only entry when forming the graph; that
entry accepts only cache-publication inputs, with no query, gate, visibility, attention result
or attention/merge launches. Its K/V arithmetic is the ordinary projection and publication
arithmetic; when
drafting, one pass over each slot's block reads its domain's accepted and injected
rows plus fresh block rows according to explicit per-layer causal or bidirectional
attention. Sliding windows independently bound history and fresh rows, using strict
position distance less than the normalized window size. Layout metadata determines
whether proposal selection begins at the anchor row or the first mask row. The draft's owned or target-shared vocabulary projection and selection
read the proposing rows over the full target vocabulary. A reduced vocabulary requires an
artifact-declared projection and token mapping; token frequency is never a vocabulary boundary.
Separate drafts read raw embedding rows with only their explicitly declared input
scale. They use their bound vocabulary projection without inheriting the target's
logit scale or softcap. A draft-declared softcap is applied in the readout epilogue
before its own selection bias; absent metadata preserves raw logits. DSpark then chains its slots through the Markov bias.
Its fixed service confidence threshold is zero, so confidence cannot decline a
proposal; preparation and graph formation omit confidence evaluation and the
normalized features used only by that evaluation. DFlash2 runs every block-pass sublayer unfused:
the normed rows project to per-row coefficients of a grouped causal convolution that restarts at
each slot's block, its first half feeds the operator's plain projections and its second half
convolves the operator's output before the residual add; its proposing rows keep their top-k
logits, and one ordered step per proposal selects each slot's candidate from the predecessor and
successor codebooks, the anchor preceding the first. Its proposals are that greedy path for
greedy and sampled requests alike (the target's verification decides acceptance). Context
injection is the same for every variant. A draft graph class records its physical
proposal count; the selected class computes exactly that block size. Verification may
consume a shorter prefix independently. Graphs are prepared before readiness, never
compiled per round, and their resources participate in the same assessment as load.
Unequal physical block widths form separate execution groups because padding can
change noncausal predictions. Block rows append nowhere.
The planned separate-draft method records the sorted, distinct positive generated
proposal counts its configured policy may choose. Only those physical block
sizes are prepared: one for a fixed width, and for the adaptive default every
width whose verification fills a launch row class; verification
prefix choices add no block graph classes. A short output grant selects an
eligible configured action or plain decoding, never an unprepared physical size.
Entry/catch-up capacity remains bounded by the load's proposal cap, since the
previous action can have accepted more tokens than the current action generates.
Conditioning overlays are Seismic workflows with only external ports, sealed once per overlaid row
count and never per request. A step with conditioning queues, after its embedding entry, one
overlay run per contiguous range, binding the source span and the matching row view of the
embedding output; the device queue orders them before the first block. Overlays add no scratch or
result storage; the embedding result remains under its original output lease.
Causal target attention uses the cache representation for every earlier row and the dense
prepared key/value for the query's own row. Moving an earlier row between the current batch and
committed history must not change its codec representation. Host visibility includes earlier
reserved destinations when the target cache is compressed; publication of those destinations
completes before attention reads them. Prefill already separates preparation from attention;
multi-row decode uses an ordered append phase and suppresses its fused append. Single-row decode
keeps its existing fused append. Media rows with declared bidirectional attention retain their
dense fresh span, and separate drafters retain their declared block semantics.
History and recurrent state ports bind slab address tables, with each slab retained and ordered
as a used resource until submitted work completes. Every visible span is slab-contained. No
attention launch concurrently appends a row it reads through a visible span; a preceding ordered
preparation may publish reserved rows for that launch. Preparation does not commit tentative state.
An attention history is read whole (the host's visible spans and fresh rows) or selected by an
indexer (`HistoryReads::Selected`). A selected read's graph adds the indexer: an
`attention_project` of its query heads and raw key from the sublayer's own input under the same
norm, then `indexer_select`, which appends each row's raw key (with its coordinates) and the block
key it finishes to the indexer's Window and Block domains (`state-transactions.md`) and selects
each row's blocks: past `blocks` complete blocks the `blocks` best by score (ties to the lower
block) and the row's tail, every visible row before. The indexer's projection outputs its query
and raw key in the activation element, as every projection does (the projection families stage
2-byte operands); from them on the indexer is F32 (stored raw and block keys, prepared queries,
scores) and rounds nowhere, as the F32 reference computes it. A selected read's domains are causally
cached: a row's earlier launch rows are visible spans, its own row is fresh, and its prepared key
and value are read dense, never through the cache's quantization, as a dense decode row reads
them. Decode rows read the Segments form: `indexer_select` emits each row's read rows of its
visible spans as at most `C = blocks + 1 + 64` slab-contained `[start, end)` runs in key order (64
the visible-span class bound), and attention reads the row's own row fresh; a multi-row launch
appends its rows first in an ordered append launch and suppresses the fused append (the scored
decode entries' `R` is `C`). Prefill rows read the WorkList form: a per-row bitmap of `B` 32-block
words of `BR`-key blocks (`B` enough for every block of the served context; both static axes of
the prefill entries) that the prefill applies as a key mask (a block past the bitmap is not read),
and from which a list launch of the prefill entry builds, per query tile of the backend's own row
tiling, the key tiles some row of the tile reads; the attend launch walks only those tiles. The
list belongs to the prefill entry rather than to `indexer_select` because its tiles follow the
attend launch's query and key tiling, a tuned parameter of each backend's prefill that no other
entry may assume. A query tile lists at most its rows' tiles of the bitmap's `32·B·BR` keys (its
rows may belong to different requests), so the list scratch follows the served context, not the
history reservation. WorkList forms are the staged fragment forms, and their graphs list no
history tiles. Each selected block graph comes in two classes, a fifth class axis: one that scores
and one that does not. A launch whose rows all have at most `blocks` complete blocks takes the
class without scoring: `indexer_select` only appends, and attention binds the host's visible
spans and fresh rows as a read of every visible row does. The indexer's score scratch (`M · 32 · B`
F32 scores) is charged in the class's certified workspace, and the decode demand charges the
indexer's projection, its select call and its per-token block key and score traffic, and caps
attention's selected read at `(blocks + 1) · block` tokens.
An attention sublayer is one segmented query | gate | key | value projection, the fused entry and
the output projection, whatever its form: each optional part of the operator (interleaved or
separate gate, own keys and values, head norms, value norm, rotated pairs) is a static axis of
extent 0 or 1, so a form is a kernel specialization, never a different graph. An absent weight
segment binds zero rows of the query weight and an absent head norm zero rows of a unit norm row;
the rotary table (axis, frequency and amplitude per pair), the unit row, the score scale and the
gate function come from the operator. The per-row work a form adds is small beside the history the
entry streams. Wide heads (up to 512
columns, query groups up to 16) stay within each backend's workgroup memory (the 32 KiB floor on
Metal and Vulkan, 32-lane subgroups): decode splits a kv head's query group into register-resident slices that each
stream the history once, or in its grouped-query matrix form makes the group's query heads the rows
of tensor-core tiles that stream the history once per kv head (the prefill's tile body on CUDA and
Vulkan), and prefill scores whole heads but accumulates outputs one 256-column
window per pass, so a wide form is a specialization of the same entries. Few kv heads over long
history are parallelized across the history, never across a different graph: decode splits a row's
keys into tuned partitions, and prefill may split a tile's history keys across tuned partition
groups whose partial softmax states merge in fixed order in a second launch. Metal affine matrix
decode also declares row/head/column tiling: a threadgroup owns up to eight global query heads
and two query rows, with two column slices per row. Head bands publish disjoint global heads into
the same partition ABI, and only band zero appends fresh history for its kv head. Query storage
and the cooperatively staged K/V tile alias across barrier-protected lifetimes; partial-score
exchange occupies a separate region after K and V. Source memory predicates determine physical
eligibility. The associated checked computation binds the partial-score slice order; native MMA
and subgroup reductions still require fidelity comparison against that computation, rather than
an assumption of bitwise identity. The tuner must qualify and measure this family within the
original first-load allowance before selecting it. Metal's prefill also
declares a direct form, a tuned specialization of the same entries: a simdgroup keeps its queries in
tensor-operation registers and reads key and value tiles as device tensor operands, so nothing is
staged; K8/V4 history is first decoded into F16, with the staged decode's arithmetic, so both forms
multiply the same operands, and the form is chosen only where tuning measures it faster. Decoded
history follows neither the history reservation (an address-space ceiling from the device's bytes)
nor the context limit, and adds no scratch: the affine prefill entry takes the history row tiles its
launch's rows see (`history_tiles`: the distinct 256-row tiles holding a row of any visible span,
ascending, then -1), and a call that lists tiles is charged its partial outputs for the most key
partitions any configuration takes, which is what a call that lists none is charged; the direct
form decodes into what its own partitions leave (a window of about 16k x G / KV rows less M x G
per partition in use; the form's key partitions stop at half of the most). The listed rows are
taken in list order, tiles adjacent in the history adjacent in the window, and a tile the launch
does not see (between two requests' spans) reads zero rows. A class whose listed rows fit the
window is one round of a decode and an attend launch; a larger one repeats them (the entry's
`repeat` block), each round taking a window's worth of keys and splitting those into its own key
partitions, and a fold launch after each attend merges the round's split records into the one
state per row the window keeps, by the partition merge rule, so a row's rounds are to it what key
partitions are. For 128-, 256- and 512-column heads on simdgroup matrices the co-issue form reads
the same window in the same rounds: it pairs the attend launch's simdgroups, one forming Q K^T on
the matrix pipe and one P V as scalar F16 products accumulated in F16 over a step of keys, within
the BF16 tolerance of the default. A 512-column head is two 256-column windows there, a pair per
window: each pair scores and produces its own window's columns and the two score simdgroups of a
row group add their partial scores, so the head's work is two 256-column heads' (the staged form
walks its keys once per output window, forming the whole head's scores each time and decoding the
history in every threadgroup). A prefill attention graph over K8/V4 history therefore comes in
classes that list tiles (powers of two from the smallest class its decoded-history form declares up
to one request's worth, the domain's span limit in pages: 16 tiles for a form that decodes into a
window, whose launches do nothing different below it, and one tile for a form whose scratch and
launches follow the class) and one that lists none; all hold the same workspace, the listing classes admit the forms
that read decoded history, and the other admits only the forms that read history in place. Metal's decoded-history workspace retains enough storage per query row for its fold state and one
history round at every admitted launch class; increasing the query grid cannot consume that
reserve or make its window geometry negative. CUDA may instead stage the unique listed history
pages into operation-local F16 operands once, with a row-validity union mask that excludes gaps,
partial pages and unallocated rows. Each consumer row maps independently to that compact scratch,
retaining the original source-slab tile boundaries, per-query visibility, fresh operands and
softmax sequence. Its descriptor/mask, decode, prepare, attend and merge launches are one candidate
transaction; all scratch is heap-charged and the original in-place families remain available.
A launch takes the smallest listing class that holds the tiles its rows together see (so it
dispatches at most twice the rounds it needs; a round with nothing to do returns at once) and the
class that lists none when they exceed the largest; listing and unlisted kernels tune separately.
A device that has no form reading decoded history for the graph's heads has only the class that
lists none, which spares it tuning and forming kernels no form of which it can select: Metal has
one on tensor operations, and on simdgroup matrices for 128-, 256- and 512-column heads; CUDA
has F16 operand staging for the affine entry's32-,64-,128-,256- and512-column heads
(`planning::reads_decoded_history`). The tensor operations fact is the device's own probe
(`DeviceInfo::forms_tensor_operations` before it is opened, which planning and assessment read;
graph preparation fails if the opened device disagrees).
The draft head's and the separate draft's graphs list none.

An image encodes at most the load's image cell limit: the cells its declared resize admits, within
4096 (`MAX_IMAGE_CELLS`), and within the launch row bound only when the decoder's media rows attend
each other, since such an image is one launch. Otherwise prefill places an image's features across
launches, so the vision limit is independent of the launch row bound. The host bounds image resize to
the limit, and input installation refuses a larger image as a request error before reserving a vision
slot. The load prepares one vision graph per image cell class (the launch row ladder continued past
the launch bound); an image runs in the class covering its cells, its patch rows first and the
class's padding rows after. Every vision attention reads its keys as a span per row, its window's
rows or its image's, so no image row reads a padding row; padding inputs are zero and the published
features are the image's leading cell rows. The tower is one program of vision operators the projector description
composes (patch stem, row norms, projections with their epilogues, attention with optional rotary,
cell pooling or concatenation). A tower post-norm applies per patch before concatenation;
a projection norm then applies over the complete concatenated cell. Learned position tables
do not imply rotary attention. A form a backend's operators do not run is refused when the program is
prepared, never approximated. Seismic's recurrent workflow derives
the exact window and delta state contracts from its checked entries. Each recurrent block binds
the store's bank slabs and, per run, bank tables for its exact active request slots: each entry
that binds them (the state entry, and the convolved step form's projection for the windows) reads
the slot's accepted bank and writes only its successor bank, in place, within the block's ordered
submission. The engine does not dispatch state transfers around the block,
copy state between banks, or bind padded request state. A block with an n-gram input binds the
step's token rows as a device tensor whatever their source (the previous step's selection, or the
host's tokens, which such a model's entry copies into token rows it keeps bound for the blocks)
and evaluates its table rows on the device from them and its bank's token window (`ngram_rows`),
so its one graph per class serves both sources and its steps chain on device-selected tokens like
any other. Its stored table's
rows ([residency](residency.md)) are read behind a host gate: the block's graph waits at a gate
node until the table's filler has written the step's rows, reads them (`stored_rows`), then
signals that it has read them. A gate node is a Seismic timeline node sealed into the graph
(`TimelineNode::Wait` or `Signal`, at a position between its calls); each run binds every node to
a timeline and the value it waits for or signals, and a run with an unbound node is refused. A
timeline is a monotonic host-visible value: a shared event on Metal, a mapped host word the
stream waits on (`cuStreamWaitValue32`) and writes on CUDA, a timeline semaphore on Vulkan
(waited and signaled at queue-submission boundaries, so a gate divides its run's submission), and
a host condition on the CPU, whose run executes its segments in order across the gates. The
entry graph of a step on selected tokens copies the selection into host-shared storage and
signals the step's selection notice, from which the filler names the step's rows; a step on the
host's tokens is named when it is queued. Workflow activations cover
submitted concurrency, and retained outputs cover submitted and live request owners. The target's
residual stream between block graphs is one device pair, returned at the end of each submission:
every reader of it is queued in that submission and the device runs the next launch after it.
The target's stateless tail is the longest run of trailing blocks that publish no state:
attention blocks that read another layer's history (a Shared domain, which they never append
to) and write no draft taps. Preparation derives it from the plan, never from the family; a
model whose last block keeps history, a recurrent bank or a tap has none. A tail row's output
reaches nothing but later tail blocks and the readout, so a step runs the tail over the window
of the smallest row class that holds the rows its readout consumes (its feature, projected and
selected rows), placed at the first of them, with the window's rows of the residual, per-layer
rows and controls, and its readout reads that window; a step that reads no row stops before
the tail. The window's graphs are the sealed classes of its row count, so the tail adds no class.
A prompt chunk of a KV-shared model (Gemma 4 E2B and E4B) thus runs the tail over its last row
or not at all, while a drafter that reads every row's features keeps the whole batch.
Vision holds no activation or output until a request encodes an image, and it is outside the
shared workspace arena. Each image cell class has its own layout and storage: an encode claims an
activation (its own workspace and upload regions) and an output slot of its image's class under the
heap like any growth, and free ones are released when idle, so vision memory follows the images
actually encoded, never the largest class. Encoded images belong to their request, so the vision
output slots hold every live request's images; retained checkpoints share those
features and do not size the pool: an encode that finds every output pinned by retention releases
retention through the ordinary capacity release order. Source-weight upload uses the largest admitted encoded tensor as a one-shot startup
resource. Qualification holds one weight scope's fixtures at a time, at their resident
representation: a block's weights, the target's norm and output (the embedding table is
qualified on one row), or a head block's weights with its draft projection. Qualification and
upload are sequential, so the startup-only addition is their maximum. Prepared native argument,
intermediate, and result storage is charged from the prepared workflows and checked against their
actual allocations.

One program factory is selected at startup. A program returns its submission once the work is
queued: the native target program queues every graph run of a step without waiting, in Seismic
sequences of doubling length (1, 2, 4, … runs, the remainder submitted at the step's end), so the
device starts on the first run at once, each sequence is prepared while the device runs the previous
ones, and a step has about log2 of its run count submission boundaries (a backend may divide a
sequence further to bound each queue submission's device time, see
[Seismic compilation](../seismic-compilation.md)). One submission per run
leaves a device gap at every boundary; one submission per step starts the device only after the
host has prepared the whole step; both measure slower. It then returns a pending submission whose completion is observed off the owning thread, while other lanes may
return already ready submissions. Every submission owns its validated launch, state transactions,
workspace, and output until completion; the host waits on the device only where it reads a
result, such as a target step's selection. Finishing physical work returns the output and the
launch's reconciliation payload. A failure or drop releases the same owned resources through RAII.
While a round executes, the service handles work that does not depend on its result (control
commands, admission, and publication of the previous round's tokens, which follows the next
round's submission); it never submits a round on unreconciled output. Host constants of a sealed
graph (rotary components and frequencies, identity row maps) are uploaded once and bound statically with the
weights; runs write only per-step controls.
Before submission, the service resolves exact typed requirements against read-only availability,
applies retention eviction or live preemption while selection remains provisional, and takes one
owned reservation. That reservation contains every workspace, output, and state successor claim;
the service also holds exact publication-ring permits for the result. Launch construction consumes
those claims and performs no allocation. A capacity result after reservation is an invariant
failure. Device and invariant submission failures terminate the domain. Recurrent repair retains
its exact accepted prefix and successor claims through this ownership path.

The worker constructs one concrete program family for target, head, projection, vision, and state
maintenance. The executor domain and service owner are generic over that family, so each flight
keeps its concrete submission type through completion and reconciliation. The root seals the
generic owner behind its non-generic worker command interface before exposing the engine client.
Replacing the family changes program construction without changing admission, scheduling,
generation transitions, or publication.

## Acceptance criteria

- Every required entry corresponds to one ordered, attested callable slot.
- Every selected tuned kernel configuration has matching bounded-policy evidence for each shape class it
  serves; at a fixed shape class, changing a peer row's inputs leaves a row's results
  bit-identical.
- No independent kernel requirement set, semantic class set, or runtime handle query remains.
- No independent numerical tensor recipe or request-time role lookup remains; Seismic owns each
  prepared workflow's complete tensor contracts and reports its exact storage charge.
- Recurrent state is read and published in place by the block's checked entries for the exact
  active request slots; no bank is copied, and no entry writes an accepted bank or the zero seed.
- No block of the stateless tail runs a row its step's readout does not consume.
- Every persistent, workspace, output, retention, and startup-peak byte traces to the plan.
- Ready and pending submissions drive one executor lifecycle, and no round is submitted before
  its predecessor's output is reconciled.
- The same service owner accepts either family without lane-specific path branches or erased submissions.
- Finished work retains all state needed for exactly one reconciliation.

Load warmup exercises one target row and, when drafting is enabled, its state-only
priming row before publishing readiness. This imports and binds the enabled drafter
component during load rather than delaying the first request. Both numerical advances
are aborted afterward; the request leaves no accepted state. Optional components may
still be released under the ordinary memory policy and rebound when needed.

Dense expansion bindings include the semantic activation function as well as
operand representations and expansion width. Layers with different nonlinear
consumers cannot share one preparation/qualification group merely because
weights have the same shape. Native preparation reads that bound function for
both a dense sublayer and a dense branch beside routing; tuning uses the same
function that serving calls. The tuning identity includes the function for
paired and up-only expansion, so numerical evidence for SiLU cannot be reused
for GELU-tanh or another consumer.
