# MLX performance ceilings

**A ceiling is an optimistic theoretical upper bound on performance, derived from
unavoidable resource demands under MLX. Reference performance never defines it.** Prefer a
bound that leaves too much headroom over one that falsely declares saturation.

[Components](components.md) identify implementations and contracts.
[Derivations](performance/derivations.md) define reusable formulas; the
[catalog](performance/catalog.md) binds model contracts to them. Architecture trees
select implementations of those contracts.

## Dimensions

Each component contract defines the smallest set of performance dimensions that
captures meaningful, independently moving outcomes. Use one percentage when one
objective suffices. Split only when a single number would conceal a meaningful
trade-off or an independently important improvement/regression. Never average
dimensions into an overall score.

Every dimension has a concise uppercase code and a full identifier, including
when a contract has only one dimension:

```text
FAMILY:COMPONENT/DIMENSION
MODEL:ATTENTION/EXEC
STATE:QWEN35/MEM
STATE:QWEN35/RESTORE
```

The catalog defines every such ID: meaning, metric/unit, observation boundary,
theoretical bound and why it is separate. Codes are scoped to their component
contract; reuse a code consistently but never infer its metric from spelling alone.
The dimension ID names the assessment target, independently of implementation source
and variant. Evidence separately identifies `STATE:QWEN35:MAG:HYBRID` and its revision.
A single-dimension contract still uses `/DIMENSION` in its full assessment ID; its
tree node displays one percentage without an unnecessary label. Multiple dimensions
display their codes beside their percentages.

Context length, batch size, prefill/decode mode and cache residency normally select
operating points within a dimension. Bandwidth, arithmetic and dispatch explain
execution efficiency; they are not automatically separate scores. Additional
diagnostics and constraints remain visible without becoming headline dimensions.

## Meaning of the percentage

For a fixed workload producing `u` units of useful work, let `L` be a derived lower
bound on elapsed time and `T` the measured elapsed time:

```text
throughput ceiling U = u / L
efficiency           = (u / T) / U = L / T
displayed percentage = 100 × efficiency
```

The primary displayed number is efficiency against this upper bound. A loose bound
understates efficiency; it does not establish how much faster an implementation
can actually become. Record looseness and assumptions alongside the percentage.
Do not require a kernel approaching the ceiling before publishing a valid bound.

A measurement above the ceiling challenges the derivation, its applicability or
the measurement. Never clamp it to 100%. If no positive time bound is established,
report an unbounded/unresolved ceiling and no percentage. A zero-overhead wrapper
may have meaningful excess time without a meaningful efficiency ratio.

For a footprint dimension, derive a minimum necessary byte count `M_min` and measure
the corresponding footprint `M`: efficiency is `M_min / M`. This is storage
efficiency, not a throughput rate. In both cases an optimistic minimum resource
cost defines the ideal; the percentage measures actual performance against it.
An unresolved or zero minimum gives no meaningful percentage. A measured cost below
the minimum challenges the model just as throughput above its ceiling does.

Each dimension optimizes its own objective under the declared constraints. The
separate optima need not be jointly attainable. In particular, state restoration
may exploit the allowed memory budget while minimum footprint may require replay.
Do not fix the incumbent snapshot/copy policy in the theoretical denominator.

## Contract and operating point

The contract defines a derivation for each `FAMILY:COMPONENT/DIMENSION`, independent
of source and variant.
Equivalent upstream and owned implementations use the same ceiling. Where input
layouts or observability differ, state those conditions and include adaptation
when comparing equivalent work. Do not change the denominator to favor a variant.

An operating point fixes:

- Mathematical work, supported input domain, shapes and positions; no foreknowledge
  of the particular answer or future requests.
- Weight encoding, numerical/output requirements, conditioning and state semantics.
- Initial residency, memory budget, retained history and required observable outputs.
- MLX runtime/version, device capacities and relevant memory hierarchy.
- Useful-work unit and timing boundary: startup, standalone call, fused parent,
  steady decode interval or request workload.

Footprint dimensions specify the ownership boundary and observation instant or
interval instead of a timing boundary. Shared allocations are counted once.

Custom Metal, graph compilation, packing, fusion and legal reuse are permitted.
Changing the model's precision, semantics or useful output is not the same work.
The formula is portable; a numerical ceiling still depends on a platform profile.

## Resource rule

For each resource `r`, derive a minimum necessary demand `D_r` and use an upper
capacity `C_r`. Include only justified terms:

```text
L_resource = max_r(D_r / C_r)
L          = max(L_resource, independently justified dependency/time bounds)
```

Competing work on one resource shares its capacity. Different resource constraints
combine with `max`, not a sum that assumes no overlap. Count conversion, instruction,
reduction or runtime costs only where necessity and capacity are established.
Conventional arithmetic counts are conditional on their algorithm class, not proofs
of minimum work across all algebraic algorithms.

When uncertain, omit an unproved cost or allow optimistic overlap/reuse and label
that relaxation. Missing capacity values remain symbolic; do not replace them with
a slow observed kernel rate. A profile distinguishes authoritative upper capacities
from estimates and observations. Measured sustained bandwidth alone is not a proved
maximum. Estimated profiles yield estimated, not certified, numerical ceilings.

## Recursive composition

Components expose resource demands and boundary conditions, not just scalar times.
At the parent boundary:

1. Select the actual mathematical children and invocation multiplicities.
2. Identify shared data and required externally visible state/output.
3. Remove intermediate reads/writes and dispatch boundaries that legal fusion can
   eliminate. Count shared inputs once unless extra transfers are proved necessary.
4. Aggregate unavoidable demand on each shared resource, then apply its capacity.
5. Add a serial-stage bound only when those stages cannot overlap even under the
   allowed fusion, tiling and runtime execution model.

A dependency between values does not prove that entire standalone calls must run
serially; their work may pipeline or fuse. Summing isolated child minima can therefore
produce a false ceiling. Unknown extra transfers, dispatches and barriers contribute
zero to the initial optimistic bound, with the missing constraint made explicit.

The parent is a theoretical evaluation of the contract graph. A separate diagnostic
model may use measured child timings, actual copies and dispatch counts to explain
current behavior. Never feed those observations into the theoretical denominator
merely because the current implementation incurs them.

The display is a tree; accounting follows the dependency/resource graph. Percentages
are not averaged. Keep exposed time and parent sensitivity available with each node;
a low-efficiency tiny component need not control full-model performance.

Compose each parent's declared objective from relevant child demands, even when the
children expose different dimensions. Child memory affects feasibility; child
restoration time can affect parent execution. Neither implies extra parent score
dimensions. Recompute the parent's bound rather than combining child percentages.

## Evidence and current assessments

A percentage belongs to a dimension, implementation revision and operating point.
Samples establish performance at those points; they do not prove a score everywhere.
Retain sample identities, coverage, variability and any interpolation assumptions.
Distinguish directly measured assessments from estimates between samples. A tree
view selects an explicit workload/profile and shows one percentage per declared
dimension, or `unmeasured` / `unresolved`; it never silently extrapolates a global score.

An assessment records:

- Full dimension ID, full implementation ID and implementation fingerprint.
- Artifact/configuration, actual child selections and their fingerprints, relevant
  dependencies/runtime, operating point and platform-profile revision.
- Derivation revision, parameter binding, evaluated bound and raw measurement IDs.
- Result, evidence coverage and uncertainty; whether measured or estimated.

**Any implementation change resets all its current dimension assessments to
`unmeasured`.** Preserve the stable ID and historical results. Apply this transitively
to every actual parent composition using the changed implementation; unaffected
components keep their assessments. Fingerprints must cover implementation content,
selected children and performance-relevant configuration/dependencies, so an unchanged
parent source file cannot conceal a changed child. Do not carry old scores forward
on an assumption that the change is performance-neutral. New samples qualify only
their covered points on the new fingerprint.

## Derivation records and history

Each catalog entry identifies its contract, reusable formulas, parameter binding,
assumptions, relaxations, declared dimensions/metrics and outstanding inputs. A useful bound
can be symbolic and loose. Improving its tightness is separate from improving code.

Reference a derivation by document section plus content revision/hash. Evaluated
records preserve the derivation revision, component/source revision, selected child
IDs, operating point, platform-profile revision, symbolic inputs, evaluated bound
and measurement identity. No hardware rates or measurements means no invented
throughput number or efficiency percentage.

Keep observations immutable. A revised derivation may re-evaluate an old observation
as a new assessment; it must not overwrite the old score or appear as a code speedup.
Until re-evaluated, its old percentage is historical. Changing only a derivation does
not invalidate a matching raw measurement or require another benchmark. Changing an
implementation does not by itself invalidate the contract's theoretical derivation.
Architecture docs link to their catalog bindings rather than duplicating formulas.
The existing benchmark system can evaluate these formulas when inputs are available;
this design does not require another benchmark framework.

## Current deliverable

The derivations and catalog are symbolic analysis. No benchmark campaign or runtime
calibration accompanies them. The Qwen static worksheet described in the catalog reads
configuration and tensor headers only. It supplies geometry and storage quantities,
not measured traffic, timing or a numerical ceiling.
