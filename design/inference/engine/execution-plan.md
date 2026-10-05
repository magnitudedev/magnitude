---
applies_to:
  - inference/engine/executor/**
  - inference/engine/kernels/**
  - inference/engine/state/**
  - inference/engine/src/execution.rs
  - inference/engine/src/planning.rs
---

# Numerical execution plan

Before allocating persistent device memory, a device-aware planner resolves one immutable plan
for enabled components, exact weight representations and tied storage, ordered checked-entry
program slots, method capabilities, service policy, and all device resources. The ordered program
topology is both the complete requirement set and the construction recipe. Program construction
attests every exact slot and returns typed callable groups; execution never queries a handle map or
coarse coverage class after readiness.
Metadata-only model assessment derives resident weight bytes and slab-rounded history and recurrent
bank bytes from this same model load plan and state layout, for a one-conversation workload at the
lesser of supported context and 100,000 tokens. It does not read weight payloads or open a device.
The recurrent fit charge includes the bank slabs needed for the accepted bank, one in-flight
successor, the lookahead successor when lookahead is enabled, and the pristine seed. State startup
includes one history slab per history domain and the zero-seed bank slab per store; later slabs
are heap claims.
Those exact model terms alone do not establish fit: prepared graph resources, workspace, startup
transients and the device's stable fit capacity are added as upper bounds derived from the same
header-only program plan, so a fit result never undercounts. The graph-resource bound builds every
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
Device assessment uses the allocation domain's total capacity, bounded by
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
resource allocation, weight import and warm-up.
Native entries with declared tuning parameters are tuned on the opened device during this
preparation, on the first preparation for each tuning key, whether by a load or by the prepare-only
job that follows a catalog installation: each such entry registers a tuning case
that supplies static values from model geometry, weighted tuning points over the shape classes that
entry serves (every row class of its graph path, crossed with served history lengths for attention,
including the empty history that exercises the fresh-only path;
selection-row classes up to the selection bound for the readout, each retaining its own step-time
share), rotations over real resident weights of distinct layers for weight-streaming decode rows,
control tables packed by the batch builder. Every tuned entry uses a bounded precision policy,
with explicit floating result and writable-state subject limits shared with the compiler policy
representation. Production tuning uses the declaration default native specialization as its
reference. This checks candidate agreement with the baseline, not independent agreement with
source semantics; kernel and model regressions must detect shared defects. Every candidate, including the
default, must pass every case before continued performance sampling. The first suitable timed
execution supplies its validation outputs. Integer, Boolean and packed-code state remains exact.
Every tensor an entry writes in place is case-owned, and its case declares the rows the entry
writes; real serving state is never bound. Those rows are restored before each reference and
validated invocation, and validation compares every result element and every written row. The
rest is input the entry only reads, so no long history is re-uploaded or read back per candidate.
One case per unit, the one with the least state, restores and compares its state whole, so a
write outside the declared rows is rejected. Timed passes after a
candidate's validated invocation are batched and run on the state they leave, the same way for
every configuration, so timing never pays a host reset per invocation. Matching evidence from earlier in the same search can be
reused, but timing equivalence alone is not numerical evidence. Whole-model numerical and raw
output/parser regressions are development and release qualification, not another startup gate or
a search over kernel combinations. Local bounded policies do not claim a mathematical bound on
whole-model accumulated error.
An entry that declares parameters without a case fails preparation; there are no
engine-side default parameter values. An entry prepared again with identical element bindings
and static values reuses the load's first tuning result; its rotations already span the weights
of several layers. Disjoint row-class workloads may prepare distinct configurations of the same
entry contract; each is tuned and validated on the classes it serves, and the exact graph class
binds its prepared configuration. Entry-wide
declarations are tuned within a fixed tuning time (60 s per preparation, on any device), in three
walks of the program. A count finds the model's tuning units (entry, element bindings, static
values and served workload), their launches per step and which have a stored result; nothing is
formed or measured. The tuning time starts with the census, which measures each unit that will
search at the points every candidate must pass (its cheapest point when none must), the fixed
cost of searching it, within the tuning time left. Keeping a unit's defaults costs nothing and is
always valid (they are the validation reference), so every other piece of work starts only when
its predicted cost fits. The unit's points are built and measured in ascending estimated cost
(rows, and for attention each row's rows plus visible history): a point's inputs are built only
when it is admitted (weight imports and generated activations each predicted from their size and
the rates measured so far, and refused when they would not fit), then its reference runs and the
defaults are validated and measured there. Each point is predicted from the last admitted one:
its building and validation as measured, its invocations at the defaults' measured device time
scaled by cost. Every point of at most 8 rows (the decode range, where implementations'
row-dependent code paths differ) is required: a unit whose required points do not fit keeps its
defaults without searching. The census keeps the inputs it built for the unit's search. Its
measurements give each unit's share of step time: every row class it measured holds its share of
the step, split among the units serving it by launches per step times the defaults' mean time
there. Each unit's budget is its share's part of the tuning time left among the units still to
search, so time a unit leaves goes to the units after it and an overrun is taken from them, and
tuning ends within the tuning time. The search reuses the census's measurements of the defaults
and admits further points while they fit a tenth of its budget; a point not admitted is not
timed and folds its weight into the largest admitted point of its history class. It validates
every candidate at the timed points,
ending exploration when what remains covers confirming its finalists (the defaults and the
cheapest candidates within noise of the leader, each predicted from its own measured samples)
and validating its choice at the points it did not time (predicted from the last admitted
point). Those points' inputs are built only to validate a choice other than the defaults; a
choice that fails there, or whose validation there would not fit, gives way to the defaults:
every chosen configuration passes validation at every served point. Shared history planes live
for the whole tuning, so units reading the same history build it once. A launch-scoped declaration admits its points the same
way, and its factored search runs within the same window. A slower device or build tunes fewer
points, candidates and units, never longer; each unit's report gives its budget and its time.
The engine owns every cache, under a directory the host names (`--cache-dir`; without one nothing
is cached). It holds the program artifacts Seismic keeps (CUDA CUBINs, Vulkan SPIR-V), through
the device's artifact store, one directory per toolchain namespace, and one tuning result per
tuning key. The key is a digest over what a stored result is valid for: the tuning version, the
device and toolchain identity (Metal OS build; CUDA driver and NVRTC release), the unit, the
implementation digest (declaration and rendered source), the precision policy, the admitted error
classes the entry declares with their envelopes (none for an entry that declares none, so
admitting a class retunes only the entries that have it), and the labels of
the served shapes its choice was validated at. How it was searched is not part of the key (search
settings, tuning time, budget shares, workload weights, the points timed), so improving the
search never invalidates a result that is still correct. The tuning version changes only with the
maintainers' approval, when tuning has improved enough to justify retuning every model, or when
stored results can no longer be read. Every search that ends is stored, so a hit prepares the
stored choice with no forming, measuring, validation or input construction: its key pins
everything that validation depended on. Tuning inputs are generated test data and resident
weights, not a fixed corpus, so numerical evidence is identified by the case's structure, policy
and reference rather than by input bytes.
Persisted numerical policies and search weights round-trip exactly; serialization must not
change eligibility or invalidate an otherwise identical objective.
Keys are content addresses for that structural and policy identity. Each unit's result is stored
as soon as its search completes, so a preparation stopped before it finishes keeps the units it
completed and a later preparation searches only the rest. Writes go through a
temporary file renamed into place; an entry that cannot be read or parsed, or whose configuration
the implementation does not admit, is a miss and is rewritten; opening the cache evicts the least
recently used entries beyond its capacity. Stored results are local measurements; nothing is
shipped. Tuning progress (milliseconds of the tuning time spent, reported when the search walk
begins and after each unit it searches; nothing when every unit is stored), total tuning time and how many units were searched or stored are reported before
readiness, or before a prepare-only job reports that it is prepared. The load also reports its
target weight import in resident bytes; no tuning or
preparation occurs after readiness. Two development
measurement tools, enabled only by the forward bench and never by a served engine, change this:
the executor's `pinned-tuning` build feature records the configuration chosen per entry (entry,
bindings, static values) and replays exactly those configurations in a later run, so two runs can
be compared bit for bit; its `tuning-survey` feature replaces the search of the named entries by a
survey that forms, measures and validates every admissible configuration and writes every sample,
so a search's choices can be judged against the whole space.

Numerics are speed first within one qualified tolerance. The only numerical requirement is the
precision gate: every kernel candidate, for every shape class it serves, is qualified per layer
against an F32 reference forward and end to end by logit top-1 agreement, mean KL divergence and
its tail against an external F32 reference forward of the same artifact (its weights dequantized
once to F32; no activation quantization). Reduced-precision activations, packed weights
dequantized inside a kernel to the activation element, changed accumulation order, and explicit
fast math functions are admitted on any backend when they pass. The gate is model-level and runs
outside the load: tuning times one entry on generated inputs and cannot measure it. A kernel form
whose error exceeds the per-dtype tolerances against its entry's default declares an error class,
and the engine keeps one envelope per class (the form's measured per-entry error with room for
the tuning inputs). A model's qualification admits classes; the host names the admitted classes
in the model policy at load, none by default, and a name no kernel declares is refused. Tuning
forms a configuration of an error class only when the class is admitted, and validates it against
the default under the class's envelope, so a defective kernel still fails while the form's
expected error passes. The first class is `int8_activations`: Metal's gate/up and down projections
past 64 rows, on tensor operations with Q4_K or Q8_0 weights, quantize each activation row to int8
per 32 columns in a pre-pass and multiply the weights' stored codes on the int8 tensor operation,
folding each 32-column block under its activation and weight scales; the weights stay exact, and
every other weight format or device runs the exact form in the same launches. Its transient
scratch is the int8 operand with its scales and block sums (rows x K x 1.2 bytes) and the weights'
decoded block scales and biases (weight rows x K / 32 x 6 bytes), per call. Without an admitted
class a row's result never depends on peer rows' values; a form that makes it depend on them is its own error class. A result may depend on its
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
numerical units. Its checked entry contracts derive graph-local mutable scratch, host-uploaded
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
layouts and counts each distinct bound constant once. A checked contract that cannot be evaluated
fails assessment rather than producing a fit estimate.
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
routed weights, program entries, and class-dependent workspace before residency begins. Linear
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
On a backend that declares the progressive head readout
(Metal, CUDA), a plan whose head admits it (a transform-free Q8_0 matrix of whole 32-value groups
without a second-level scale, under no logit softcap, with no separate draft) holds the head in
progressive planes (residency): every projecting class reads the head through the planes
(`readout_planes_rows`, the exact logits), and served loads also have certified selection
graphs, which serve every step whose selected rows, up to the backend's certified bound (one on
Metal, whose batched projection is arithmetic-bound, four on CUDA), are all unpenalized and uncut
(a temperature at most). Their levels score a row as the sampler does (`logit / temperature`
plus the row's own Gumbel noise, masks applied): `readout_top_rows` projects every vocabulary row
onto the codes' top four bits and keeps per row the largest lower-bound score as its threshold;
`readout_refine_rows` adds bit 3 to the rows whose upper-bound score reaches it and raises the
threshold; `readout_exact_rows` projects the remaining rows exactly and writes −∞ elsewhere (Metal's
top level and full pass are the projection library's GEMV, batched GEMV and GEMM over the
views' packets). Every row whose exact score can be the largest survives
each level, so the selection is the full readout's, while the head's low bits are read only for
the survivors (about 54% of the head's bytes on real text). An MTP draft head projects the planes'
leading draft-vocabulary rows: its drafting classes up to the certified bound run the certified
levels over them and the rest the full pass; a certified class serves every drafting row whatever
its shaping, since only verification decides what is emitted. MTP verification rows select
through the target's certified classes like any other step.
A separate draft (DFlash, DSpark, DFlash2) conditions on target taps instead of the final features. A
tapped block's workflow rounds the residual entering it, entering its feed-forward, or leaving it
(the exit tap is the last block's output) into that tap's column block of a draft-input buffer
the bound target workflows own (charged with their bound constants); the
readout's demanded feature rows are then the draft's fusion (one projection of the concatenated
taps) rather than the normalized final rows. The draft runs as the head lane's drafter, one sealed
workflow per class and transaction: each draft layer's attention appends the entry rows' context
keys and values (the fusion norm as its input norm, each row attending only itself); when
drafting, one non-causal pass over each slot's block `[anchor, mask, …]` reads its domain's
accepted and injected rows plus the whole slot block, and the target's vocabulary projection and
selection read the proposing rows. DSpark then chains its slots through the Markov bias and
declines a proposal below its confidence threshold. DFlash2 runs every block-pass sublayer unfused:
the normed rows project to per-row coefficients of a grouped causal convolution that restarts at
each slot's block, its first half feeds the operator's plain projections and its second half
convolves the operator's output before the residual add; its proposing rows keep their top-k
logits, and one ordered step per proposal selects each slot's candidate from the predecessor and
successor codebooks, the anchor preceding the first. Its proposals are that greedy path for
greedy and sampled requests alike (the target's verification decides acceptance). Context
injection is the same for every variant. The block always has the draft's trained width;
a load's proposal width selects its leading proposing rows. Block rows append nowhere.
Conditioning overlays are Seismic workflows with only external ports, sealed once per overlaid row
count and never per request. A step with conditioning queues, after its embedding entry, one
overlay run per contiguous range, binding the source span and the matching row view of the
embedding output; the device queue orders them before the first block. Overlays add no scratch or
result storage; the embedding result remains under its original output lease.
The fused attention entry appends each row's key and value at its destination while other rows read
history. History and recurrent state ports bind slab address tables, with each slab retained and
ordered as a used resource until submitted work completes. The entry resolves one slab base per
slab-contained part of a visible span (a span may cross slab edges), one per destination row, or
one per recurrent bank; its inner loops retain their direct component layout. This is ordered by construction: destinations are freshly reserved rows,
so no row of the batch sees one through its visible spans, and fresh rows are read from the batch's
own projections.
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
groups whose partial softmax states merge in fixed order in a second launch. Metal's prefill also
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
partitions are. For 128- and 256-column heads on simdgroup matrices the co-issue form reads the
same window in the same rounds: it pairs the attend launch's simdgroups, one forming Q K^T on the
matrix pipe and one P V as scalar F16 products accumulated in F16 over a step of keys, within the
BF16 tolerance of the default. A prefill attention graph over K8/V4 history therefore comes in
classes that list tiles (powers of two from 16 up to one request's worth, the domain's span limit
in pages) and one that lists none; all hold the same workspace, the listing classes admit the forms
that read decoded history, and the other admits only the forms that read history in place. A
launch takes the smallest listing class that holds the tiles its rows together see (so it
dispatches at most twice the rounds it needs; a round with nothing to do returns at once) and the
class that lists none when they exceed the largest; listing and unlisted kernels tune separately.
A device that has no form reading decoded history for the graph's heads has only the class that
lists none, which spares it tuning and forming kernels no form of which it can select: Metal has
one on tensor operations, and on simdgroup matrices for 128- and 256-column heads
(`planning::reads_decoded_history`). The tensor operations fact is the device's own probe
(`DeviceInfo::forms_tensor_operations` before it is opened, which planning and assessment read;
graph preparation fails if the opened device disagrees).
The draft head's and the separate draft's graphs list none.

Vision patch capacity is the admitted merged output row limit times the merge area; input validation
rejects a larger aggregate before reserving a vision slot. Vision attention sees every physical
patch row of its image (or of its row's window, a contiguous tower-row range the family supplies),
so its prepared workflow uses the exact admitted patch-row count; padding with additional patches
would alter real outputs. The tower is one program of vision operators the projector description
composes (patch stem, row norms, projections with their epilogues, rotary attention, cell pooling
or concatenation); a form a backend's operators do not run is refused when the program is
prepared, never approximated. Seismic's recurrent workflow derives
the exact window and delta state contracts from its checked entries. Each recurrent block binds
the store's bank slabs and, per run, bank tables for its exact active request slots: each entry
that binds them (the state entry, and the convolved step form's projection for the windows) reads
the slot's accepted bank and writes only its successor bank, in place, within the block's ordered
submission. The engine does not dispatch state transfers around the block,
copy state between banks, or bind padded request state. Workflow activations cover
submitted concurrency, and retained outputs cover submitted and live request owners. The target's
residual stream between block graphs is one device pair, returned at the end of each submission:
every reader of it is queued in that submission and the device runs the next launch after it.
Vision holds no activation or output until a request encodes an image; it then claims them under
the heap like any growth and releases them when idle. Encoded images belong to their request, so
the vision output pool holds every live request's images; retained checkpoints share those
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
ones, and a step has about log2 of its run count submission boundaries. One submission per run
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
- Every persistent, workspace, output, retention, and startup-peak byte traces to the plan.
- Ready and pending submissions drive one executor lifecycle, and no round is submitted before
  its predecessor's output is reconciled.
- The same service owner accepts either family without lane-specific path branches or erased submissions.
- Finished work retains all state needed for exactly one reconciliation.
