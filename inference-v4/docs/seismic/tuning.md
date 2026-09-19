# Seismic tuning

Automatic selection has one path:

**Checked source and declared conditions → retained execution family → immutable
shared solver model → search → checked selected execution → native artifact.**

Seismic constructs and interprets the execution family. `magnitude-solver` owns
optimization strategy, search state, bounds, pruning and completion. The compiler
uses the common `Model` and `Search` contract for both exact and neighborhood
algorithms. Changing algorithms does not change the family, model vocabulary or
reconstruction procedure.

## Inputs and ownership

| Input | Meaning |
| --- | --- |
| Portable program | Checked definitions, specialization and numerical/effect permissions. |
| Execution form | Complete admitted transformation and implementation domains. |
| Workload | Bound allocations, views, scalars, aliasing, known data and declared runtime domains. |
| Hardware contract | Primitive services, resource capacities, units and applicability conditions. |
| Objective | Timed execution boundary, scheduling interpretation and aggregation over the workload domain. |

The frontend owns transformation legality, operation occurrence identities,
producer identity and snapshot provenance. Backends own typed terminal operations,
dispatch, storage, participation and emission correspondence. Accounting derives
joint resource and scheduling constraints from those operations. Runtime owns
invocation checks, native compilation and artifact reuse.

A supplied Lowered IR artifact represents an explicitly selected source domain;
it is useful for compiler diagnostics and does not establish coverage of the
original portable program's alternatives.

## Execution-family construction

Construction retains local alternatives, their original finite domains and their
activation guards. Numeric domains remain arithmetic intervals or progressions
where possible. Choice identity includes the source definition, call occurrence
and defining topology; it is not an ordinal path through repeated lowering.
Independent choices do not require a Cartesian product of completed kernels.

The same original variables feed shapes, dispatch counts, coordinates, storage
capacities, operation guards and resource demands. Derived numeric operands retain
their defining equations as well as their bounds. Aliases refer to the original
variable identity. Conditional equations apply only under their defining guards.
A construction envelope bounds possible occurrences; it is never substituted for
the selected runtime quantity.

Serial setup, parallel regions, dependencies and values crossing launches belong
to the same family. Guarded alternatives can introduce different phase counts,
allocations and dependencies. Inactive phases retain their correspondence slots
but create no native pipelines or dispatches. Scratch binding identities remain
stable even when a selected scratch allocation has zero size.

Reduction topology retains the source's ordering and identity permissions.
Explicit ordered trees use bounded frontier decisions over the original ordered
leaves, including the seed. Their constraints and reconstruction preserve every
admitted tree without enumerating completed tree combinations. Segmented folds
retain segment preparation, partial identities, tails and final publication.

Applicability must be established before an alternative can become executable.
A genuinely illegal assignment is constrained out. Missing construction or
analysis remains a typed coverage obligation in the shared model. A compiler
failure is an error. None of these may silently drop an unfinished alternative,
choose a diagnostic default or report an optimum over a smaller family.

## Joint accounting and search

One model owns all implementation decisions, optional activities, dependencies,
storage lifetimes, resource reservations and the declared objective. Every shared
resource sees all applicable uses, including uses from different launches and
alternative regions. Independently optimal child schedules do not establish a
joint optimum.

Concrete and symbolic interpretations share dispatch, storage and primitive
service equations. Typed helper semantics and parameter conversions are shared
with emission. Numeric participation facts remain attached to the control paths
where they hold, including the live continuation of a padding return. Returning
lanes cannot reappear when alternative states join.

Equivalent constant quantities and predicate definitions may share bindings.
Distinct operation occurrences, reservations, memory effects and dependencies
remain distinct. Reuse requires equivalent semantics and compatible activation;
equal counts or isolated costs do not establish equivalence.

Structured repetition must describe its serial/parallel meaning, recurrence,
resource scope and finite tail through the common model contract. Bounded
expansion can implement those semantics within a declared construction limit.
Exceeding that limit remains unresolved; it does not authorize a backend-owned
search, a separate exact child solve or an unaccounted executable path.

`tune` exports once and creates one `Search`. `resume` advances that retained search
over the same immutable model and reconstruction owner. Search limits are
incremental; algorithm options remain fixed for the session. Changed source,
workload, backend conditions, objective or construction limits require a new
export. Resumption does not replay lowering or rediscover implementation choices.

## Outcomes and executable boundary

| Outcome | Meaning |
| --- | --- |
| `Optimal` | Feasible original-model assignment and completed global optimization under the declared objective. |
| `Incomplete` | Retained search, stop reason, bounds and optional feasible incumbent; no executable selection. |
| `Infeasible` | The admitted modeled domain is proven to have no feasible member. |
| Error | Invalid request, malformed model or failed compiler/reconstruction invariant. |

Only global `Optimal` can construct executable Tuned IR. An incomplete incumbent
may be reconstructed for diagnostics; it cannot be converted into a runtime
kernel. Time, work and memory limits do not weaken this gate. Unresolved coverage
cannot become proof of infeasibility or completed optimization.

Reconstruction validates the original assignment, resolves the retained source
and backend choices, and checks geometry, storage, operations, objective and
conditions against the same model. It consumes the retained implementation; it
does not prepare another kernel from settings or select a nearby legal value.
The selected execution retains its reconstructed source, not the unresolved
template. Native compilation consumes that selected execution directly.

There is no old selection coordinator, backend-local optimization fallback,
whole-kernel candidate score table, automatic algorithm switch or native
compile-and-try selection path.

## Workload reuse and objective meaning

Runtime controls are invocation inputs, not optimizer decisions. Compilation over
a declared integer domain must establish validity and the stated objective over
all admitted values. Canonical input bytes are not a representative sample. An
unsupported control or transaction relationship remains unresolved rather than
specializing to a convenient decode position.

Reuse binds semantic source/form/workload identities, hardware and implementation
contracts, objective, scheduling interpretation and analysis versions. Floating
literal identity preserves representation bits, including signed zero and NaN
payloads. Reusing a result under changed conditions requires a checked
applicability relationship. Runtime checks the device and invocation conditions
again before executing an artifact.

A model optimum is conditional on the supplied model. A hypothetical timing model,
a physical lower bound and an observed GPU duration are different claims. A
feasible schedule in an optimistic relaxation is not an upper bound in the
original model. Unenforceable hardware issue order cannot become a compiler action
merely by retaining an unchanged kernel. Physical performance claims additionally
require qualified mappings and hardware assumptions.

All compared lower bounds and feasible upper bounds must share the objective,
conditions and time units. A zero lower bound has no finite gap ratio. Native
feedback from a selected winner can inform external qualification; it does not
silently resume implementation selection.

## Verification

Validation concentrates on the boundaries:

1. Both solver algorithms receive the same complete model and reconstruction.
   Resume retains that model; changed requests are rejected.
2. Small generated cases compare family coverage and reconstructed semantics with
   independent legal realizations, including guards, dependencies, storage,
   numerical permissions and snapshot provenance.
3. Completed automatic selection passes through native compilation and execution,
   checking values, bounds, phases and repeated invocation behavior.

Typed stage verification checks scope, shape/type consistency, effects, snapshot
provenance, resolved choices and collective participation. Missing mappings or
unsupported analysis remain visible before native execution. Generated fixtures
are produced by generators rather than checked-in data files.

An installed shared interface does not imply complete language/backend coverage,
solver throughput, native correctness or Qwen readiness. Those are established by
the corresponding completed boundary and workload checks.
