---
applies_to:
  - inference-v4/engine/model-executor/**
  - inference-v4/engine/model-kernels/**
  - inference-v4/engine/model-state/**
  - inference-v4/engine/src/execution.rs
---

# Numerical execution plan

Before allocating persistent device memory, a device-aware planner resolves one immutable plan
for enabled components, exact weight representations and tied storage, ordered checked-entry
program slots, method capabilities, service policy, and all device resources. The ordered program
topology is both the complete requirement set and the construction recipe. Program construction
attests every exact slot and returns typed callable groups; execution never queries a handle map or
coarse coverage class after readiness.
Metadata-only model assessment derives resident weight bytes and per-token history and recurrent
bank bytes from this same model load plan and state layout, for a one-conversation workload at the
lesser of supported context and 100,000 tokens. It does not read weight payloads or open a device.
The recurrent fit charge includes the accepted bank, one in-flight successor and the pristine seed.
Those exact model terms alone do not establish fit: prepared scratch, startup transients and the
device's capacity must be accounted for before publishing a fit result. If the exact resident
terms alone exceed stable capacity, assessment may reject fit immediately; passing that lower
bound never proves fit.
The same header-only program plan derives exact prepared native invocation storage and the
production qualification/import startup upper bound. These remain partial bounds until Seismic
graph resource pools have a backend-specific upper bound; assessment does not infer positive fit
from the partial bounds. Exact invocation storage strengthens the no-fit lower bound; the startup
upper bound does not.
Speed assessment measures shipped native defaults on synthetic device-resident inputs separately
from model loading. It retains the device and formed-program identity, workload geometry and raw
timings. A missing measurement or a measured pair that does not establish a physical cost supplies
no speed prediction; real-model validation cannot be used to fit a correction factor.
The composition root prepares complete Seismic workflows for the admitted model geometry and
finite launch classes, imports the target component, allocates the storage reported by those
workflows, and publishes readiness only after those steps succeed. The engine does not maintain a
second numerical tensor-shape description.
Device assessment uses the allocation domain's total capacity, bounded by
applicable process limits and Metal's recommended working set. Admission of new holdings uses fresh available
memory observations for that domain. Neither calculation subtracts a fixed
planning reserve; already charged allocations are excluded from observed
availability and are not subtracted again.
The native execution path is backend-neutral: the host names a device (a backend or an exact
selector) or asks for automatic selection, which considers accelerators only and treats several
fitting devices as an explicit ambiguity; the path then executes on the opened device's backend,
and every native entry is prepared from that backend's declarations. Preparation reports every
entry the program plan needs that lacks an implementation for the backend together, and every
error names the path and the backend.
Native entries with declared tuning parameters are tuned on the opened device during this
preparation, on the first load for each tuning key: each such entry registers a tuning case
that supplies static values from model geometry, weighted tuning points over the shape classes that
entry serves (every row class of its graph path, crossed with served history lengths for attention;
projected-row classes for the readout, each retaining its own step-time share), rotations over real
resident weights of distinct layers for weight-streaming decode rows,
control tables packed by the batch builder. Every entry is validated with one engine-wide
tolerance, a defect guard derived from an error model, not the precision gate: it admits every arithmetic option an
entry declares (down to q8_1 activations) with margin, and is looser than anything the end-to-end
precision gate could accept, so tuning never rejects a configuration the gate would pass; the gate
itself is the end-to-end qualification. Every tensor an entry writes
in place (recurrent state arenas, KV history, routing tables, selection outputs) is case-owned
state: its written region is restored before each configuration's validation run, and real state
is never bound. An entry that declares parameters without a case fails preparation; there are no
engine-side default parameter values. An entry prepared again with identical element bindings and
static values reuses the load's first tuning result. Entry-wide declarations still use a
configuration budget: a census counts the model's tuning units (entry, element bindings, static
values) and their admissible configurations, and shares a per-model budget among them. A
launch-scoped declaration instead searches every candidate of each independent launch group;
its boundary choices and group candidates do not spend that budget. A safety stop on the whole
preparation's tuning (a wall-clock limit for pathological machines) ends every search early with
the best completed choice, leaving unfinished groups at their defaults; it is reported as a
warning and its results are not stored.
On CPU, expensive projection cases screen candidates at a few representative rows with folded
row shares. The default and shortlisted configurations are still confirmed, ranked and validated
at every served row; the full workload remains the final objective.
The engine owns every cache, under a directory the host names (`--cache-dir`; without one nothing
is cached). It holds CUDA images Seismic formed, through the device's artifact store, and one
tuning result per tuning key. The key is a digest over the device and toolchain identity (Metal OS
build; CUDA driver and NVRTC release), the unit, the implementation digest (declaration and
rendered source), and the search definition (search version, budget, settings, point labels and
weights, screening points and folded weights, validation rule, sample time). A hit prepares the
stored choice with no forming, measuring or validation for tuning; its key pins everything
validation depended on. Keys are
content addresses, so nothing is invalidated: changed inputs give new keys. Writes go through a
temporary file renamed into place; an entry that cannot be read or parsed, or whose configuration
the implementation does not admit, is a miss and is rewritten; opening the cache evicts the least
recently used entries beyond its capacity. Stored results are local measurements; nothing is
shipped. Tuning progress, total tuning time and how many units were searched or stored are
reported before readiness; no tuning or preparation occurs after readiness. Two development
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
fast math functions are admitted on any backend when they pass. A row's result never depends on peer rows' values; it may depend on its launch's shape class
and prepared configuration, and different shape classes agree within the gate's tolerance, not
bit for bit. Speculative verification is therefore statistically, not exactly, equivalent to
plain decoding; acceptance over the logits a verification produced remains exact.

The resource plan authorizes persistent weights and state, including the permanently pristine
recurrent zero seed, concurrent typed workspaces, outputs
that outlive workspaces, structural retention slots bounded by service request capacity and
the device domain, optional component residency, and the
qualification/startup peak. Persistent allocation follows planning. Qualification scratch is
released before readiness. Execution receives plan-issued leases and cannot allocate general
scratch outside the plan.
Variable retained feature tensors require a fitting heap claim. Retained checkpoints occupy
finite slots and release their physical charge when their final owner drops them.

Seismic composes native checked entries into prepared workflows for decoder blocks and other
numerical units. Its checked entry contracts derive graph-local mutable scratch, host-uploaded
inputs, intermediate and result tensors' representations, extents, alias conditions, and lifetimes.
It reports exact storage charges and owns bounded concurrent execution slots. Compatible launch
classes share one physical scratch arena per concurrent slot, charged at their maximum footprint.
Each slot also holds a fixed set of host-upload regions, allocated and charged with the slot: one
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

The target and head workflows cover the admitted row and history-segment ladders. Row classes are
powers of two up to 32 rows (decode, verification, concurrency) and multiples of 64 above, up to
512; history segments are powers of two. Blocks whose sealed workflow would be identical apart from
their layer (same geometry, weight representations and shapes, and state layout) share one sealed
plan per class and bind their own weights to it; the composition root reports the class count,
sealed graph count and sealing time. Decoder numerical
pipelines share projection results through checked Seismic result edges. Dense feed-forward owns
its activation product; attention owns the normed Q/K/V projection and one fused entry that prepares
queries and keys (norms, rotary) in place, accumulates the stable softmax and gates the values
(decode row classes use the partitioned decode entry, larger classes the streaming prefill entry);
routed feed-forward owns normalized input, routes and scores (ranked once by probability), and the
shared coefficient, then either the per-choice expert and shared products (row classes within the
GEMV bound) or, for larger classes, grouped tables and grouped expert outputs: choices grouped by
expert into tile-aligned blocks whose capacity derives from the class, the selected-expert count and
the tile rows, so no table is uploaded per step and no host readback sizes a launch. Seismic prepares an
exact workflow for the selected physical batch class, so a small decode batch does not execute the
maximum class width. Draft head blocks use their declared dense or routed feed-forward geometry
and the same routed numerical composition as target blocks. Header assessment includes the head's
routed weights, program entries, and class-dependent workspace before residency begins. Linear
projection stages use cooperative subgroup reductions for the smallest row classes and subgroup
matrix operations above them. A decode projection, normalization prologue
included, is one launch: each workgroup reduces its few rows' norms while staging them. Larger classes
normalize once per row into entry scratch, never per output tile. A monolithic entry that recomputes normalization,
projection, routing, or softmax for each output coordinate is not an admissible production program.
Target readout preserves every demanded feature row, and projects only rows that demand logits:
the feature and head entries each gather their hidden rows through a row table in their own
normalization prologue, so no copy or gather node precedes them. Projected rows are ordered with
the selected rows first, so shaping and sampling read the leading logits rows. Selection graphs
exist with and without the shaping stage; a step whose selected rows all leave the logits
unchanged under shaping (greedy or unit temperature without cuts, and no penalties) samples the
projected logits directly. Each selected row carries a constraint flag; only a step with a
constrained row uploads vocabulary masks, and an unconstrained row's mask is never read. The projected-row
capacity follows the service's finite decode and request-batch bounds; prefill row capacity does
not imply the same number of logits rows. Features, logits and selection stay inside one checked
Seismic workflow with one owned output lifetime.
Conditioning overlays are Seismic workflows with only external ports, sealed once per overlaid row
count and never per request. A step with conditioning queues, after its embedding entry, one
overlay run per contiguous range, binding the source span and the matching row view of the
embedding output; the device queue orders them before the first block. Overlays add no scratch or
result storage; the embedding result remains under its original output lease.
The fused attention entry appends each row's key and value at its destination while other rows read
history. This is ordered by construction: destinations are freshly reserved rows, so no row of the
batch sees one through its visible spans, and fresh rows are read from the batch's own projections.

Vision patch capacity is the admitted merged output row limit times the merge area; input validation
rejects a larger aggregate before reserving a vision slot. Vision attention sees every physical
patch row, so its prepared workflow uses the exact admitted patch-row count; padding with additional
patches would alter real outputs. Seismic's recurrent workflow derives
the exact window and delta arena contracts from its checked entries. Each recurrent block binds
the layer's arenas and, per run, bank tables for its exact active request slots: each slot's
state entry reads the slot's accepted bank and writes only its successor bank, in place, within
the block's ordered submission. The engine does not dispatch state transfers around the block,
copy state between banks, or bind padded request state. Workflow slots cover
submitted concurrency, and retained outputs cover submitted and live request owners. Source-weight
upload uses the largest admitted encoded tensor as a one-shot startup resource. Qualification and
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
- Every selectable kernel configuration has passed the precision gate for each shape class it
  serves; at a fixed shape class, changing a peer row's inputs leaves a row's results
  bit-identical.
- No independent kernel requirement set, semantic class set, or runtime handle query remains.
- No independent numerical tensor recipe or request-time role lookup remains; Seismic owns each
  prepared workflow's complete tensor contracts and reports its exact storage charge.
- Recurrent state is read and published in place by the block's checked state entry for the
  exact active request slots; no bank is copied, and no entry writes an accepted bank or the zero
  seed.
- Every persistent, workspace, output, retention, and startup-peak byte traces to the plan.
- Ready and pending submissions drive one executor lifecycle, and no round is submitted before
  its predecessor's output is reconciled.
- The same service owner accepts either family without lane-specific path branches or erased submissions.
- Finished work retains all state needed for exactly one reconciliation.
