# Seismic accounting

**Accounting is the derivation of work, movement, launch, and storage quantities of
a candidate under its prescribed mapping, and of an execution-cost estimate from
them.** It exists to give joint selection hard limits and an objective. It claims
no more than its derivation supports.

Physical tiles, windows, staging phases, and publications named below are compiler-owned execution
IR concepts retained during migration. They are not source types or source statements.

## Authority levels

| Level | What it is | Examples | May be used as |
| --- | --- | --- | --- |
| Exact static count | Arithmetic over workload constants and selected site values, by the mapping's fixed rules | Pieces of a launch, visits of an ordered region, launches, declared tile bytes per placement, pieces sharing a threadgroup, surviving outputs of an interval | Hard constraint; input to an estimate |
| Static bound | The static upper bound of a runtime-valued extent, by interval arithmetic | Trip count of a loop over visible history; work under a branch, taken as the maximum of its arms | Input to an estimate, labelled as a bound; a capacity constraint evaluated at the bound, which then holds for every admitted runtime value |
| Estimate | A model's prediction from counts and coefficients | Nanoseconds of a launch scope; write-plus-read of a surviving tile | Solver objective only |
| Unmodeled native effect | Behavior the derivation does not represent | Register allocation, spills, instruction selection and scheduling by the native compiler, cache and TLB behavior, driver submission variance, thermal state, contention | Nothing. Stated as unmodeled. |

An estimate is never named an upper bound, a measurement, or a guarantee. A missing
derivation is an explicit failure (*analysis unavailable*), never zero. A hard limit
is never folded into cost, and a cost preference is never expressed as a limit.

## One authority

Quantities, constraints, cost factors, and realization all follow the same selected
structure and the same mapping rules:

- The family fixes which candidates, sites, and sequences exist.
- The backend's mapping fixes, per candidate body, what each structure becomes
  (launches, serial loops, storage placement, local element types).
- Quantities are symbolic expressions over site values, derived once per candidate
  from its body under those rules. Constraints and cost factors evaluate them; they
  do not re-derive structure.
- Realization applies the same rules to the instantiated execution IR. Where a rule
  appears in both the quantity derivation and realization — tile placement, local
  element widening, pieces per threadgroup — it is one rule. The realized execution's
  own launch grid and declared private bytes are rechecked against the limits, so a
  disagreement surfaces as a diagnosed failure rather than a wrong kernel.

Emission consumes the realized execution. No quantity is derived from emitted
source, and nothing is measured during selection.

## Derived quantities (Metal)

Per candidate, scoped by where a statement executes: its block's multiplicity
(enclosing visits and loop trips) and the pieces of its launch.

| Quantity | Derivation |
| --- | --- |
| Pieces per binder | `ceil(extent / width)` of the site |
| Launch pieces | Product over a root `parallel` region's binders |
| Launches | One per root region, one per run of invocation-scope serial statements; a selected interval of root regions pays one |
| Lane operations | Element operations on the critical path of one piece: a tile of 32 elements or more with fixed extents is lane-distributed (`ceil(n / 32)`), any other replicated (`n`). Integer arithmetic over loop binders, constants, shape parameters, selected geometry and the participant index is coordinate computation and is not counted |
| Reduction operations | One quantity that mirrors the reduction rule: replicated input `n`; lane-local `ceil(n / 32)`; collective (contract permits reassociation, 32-bit elements) `ceil(n / 32)` plus one collective per output; ordered over a distributed input, one shuffle per element |
| Visits | Window visits of `ordered` and `pipeline` regions |
| Device bits | Distinct bits read from or written to device buffers, at the stored element width or a packed representation's exact fractional rate. A snapshot counts once per distinct value of the binders its view mentions; a view that mentions any other non-parameter variable, or a repetition inherited from a caller, counts on every visit |
| Borrowed snapshot traffic | Device bits of a `load` of external storage passed straight to a call: charged in the selected callee's body scope instead of the caller's, with no copy operations |
| Local bits | Bits moved through tile storage |
| Tile bytes, threadgroup | Declared bits of matrix-operand tiles at their native element type |
| Tile bytes, private | Declared bits per lane of other tiles at the widened local element type (`bf16`/`f16` held as `f32`; packed coefficient planes scaled likewise) |
| Private bytes of a scope's kernel | Privately placed tiles of the candidate and its ancestors in the same launch: a tile handed to a call at its whole size, a snapshot of external storage as zero (borrowed), others by the distribution rule. Sibling occurrences inlined into the launch are not seen |
| Interval materialization | Bits of every elementwise output that survives a selected interval: the last output and any output referenced after the interval |

Child calls are not folded into their parent: each candidate accounts for its own
body and is active under its own guards, so producer work is charged once per
occurrence and consumer reads once per consumer. Dynamic repetition multiplies
quantities; it never creates decisions.

A quantity with no supported derivation evaluates to an error carrying its reason.

## Estimate

The estimate model and its factor structure are part of the backend contract
([Backends](backends.md#estimate-model)). Its identity,
`metal-estimate-probe-calibrated-m4max-20260919-v3`, travels with every selection and
compiled kernel.

What the model represents: fixed launch overhead; per-visit bookkeeping; compute
throughput per lane scaled by concurrently running pieces; the reduction algorithm
the reduction rule will pick; distinct device-bus traffic and tile-storage traffic at
fixed bandwidths; tile traffic overlapped with compute by maximum and bus traffic by a
Euclidean norm; a borrowed snapshot's bus traffic overlapped with the compute of the
callee that consumes it; the slowdown of a kernel by the thread-private bytes it
declares; the materialization a fusion interval removes or keeps.

What it does not represent: everything in the *unmodeled native effect* row, overlap
between launches, the access pattern of device and tile reads (the 32-element stride of
the interleaved lane cover, window-major traversal of several rows per lane), the cost
of a bounds check the emitter could not prove (about 100 ns per iteration of a dependent
chain), the placement of a tile that is read at foreign coordinates when its size
depends on selected geometry (counted as threadgroup-placed), the limit threadgroup
memory puts on concurrency, instruction-level overlap of
independent operations inside a lane, and any interaction between scopes beyond
addition and the borrowed-snapshot rule. Launch scopes execute in sequence on Metal
under the current mappings, so addition is the prescribed protocol, not an
independence assumption about concurrent work. The measured size of the unmodeled
effects is recorded in [Backends](backends.md#estimate-model).

**Status: probe-calibrated on one device, unqualified.** Coefficients are fitted to
standalone probes and emitted kernels on one Apple M4 Max; every other device
inherits them. Consequences:

- A lower estimate is a modeled comparison. It is not evidence that a witness is
  faster.
- *Model-optimal* means optimal under this model over the exported family.
- No parity or performance claim may cite an estimate. Such claims require measured
  execution of the emitted kernels on the named device and workload.
- Replacing a coefficient by a calibrated value changes results but not the
  unqualified status. Qualification is a separate, explicit act that binds a model
  identity to a device, toolchain, and workload domain, and must change the
  identity.

## Observation

Runtime observation reports host time and device command-buffer time per invocation
([Runtime](runtime.md)). Observations are evidence for qualification and reports.
They never feed selection and are never presented in estimate units as if they were
the same quantity.

## Superseded accounting

The earlier accounting design is **not on the production path**:

- The ideal-schedule execution model and its makespan objective.
- Export of resource-scheduling problems (independent, structured, periodic,
  symbolic) to the solver.
- Hardware contracts and timing profiles as a prerequisite for selecting or
  executing.
- Workload-domain generalization of a selection across integer input domains.
- Necessary-demand and physical lower-bound derivations, and the objective check
  that gated executable construction on a proved optimum.

None of these is consulted when selecting, realizing, emitting, compiling, or
running a kernel, and nothing in this pipeline may call them from a scoring or
legality path. What remains in use from that package is mechanical: exact integer
value algebra and typed choice identities inside the Metal realized execution's
storage plan. Exact scheduling may return as a diagnostic or research tool under its
own authority labels; it may not become a selection prerequisite again.

## Acceptance

- Every constraint and factor evaluates the same symbolic quantities the mapping
  derived for that candidate; none re-walks structure with different rules.
- Every figure in inspection output is labelled by its level: count, bound,
  estimate. The estimate model identity accompanies every estimate.
- A witness whose realized execution violates a limit that a constraint covered is
  a defect in the quantity derivation, fixed there, never by clamping at emission.
