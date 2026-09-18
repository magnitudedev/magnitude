# Seismic tuning

The tuner resolves Lowered IR into Tuned IR using the legal execution space and derived
resource/dependency model. Its objective is exact optimization over the declared form,
enforced through sound search operations and validation of the same execution
constraints. Native candidate compilation, resource queries, benchmarks, heuristic
scores, and compile-and-try fallback are excluded from this path.

## Input and output boundary

| Input | Required meaning |
| --- | --- |
| Lowered IR | Backend computation and the complete supported legal alternatives. |
| Execution form | Admitted transformations, implementations, and choice domains. |
| Workload domain | Shapes, layouts, representations, values/predicates where relevant, aliasing, and initial-state conditions. |
| Implementation contracts | Numerical/effect legality, resource derivation, and emission mappings. |
| Hardware contract | Bound machine model, relevant axioms/parameters, and operating conditions. |
| Objective | Quantity to minimize, execution boundary, units, and treatment of dynamic scenarios. |

A completed result contains the actual selected Tuned IR, its derived account, validated
feasible schedule/objective, completed search status, and retained conditions. No
separate proof artifact or certificate is produced. Physical lower bounds, invocation
applicability, and mapping qualification accompany it as distinct claims; a model
optimum does not manufacture those claims.

The emitter consumes the selected execution directly. It neither replays a bag of
settings against a changed program nor chooses an unmodeled implementation.

## Objective semantics

The objective identifies exactly which work is timed: for example, completion of a
resident invocation's full launch sequence, including its required dependencies and
specified submission work. Compilation, model loading, transfers, and readback are
included only when the declared boundary includes them.

Known bindings are specialized. Remaining runtime variation stays in the workload domain
or in explicit checked specializations. Per-binding, worst-case, and expected-value
objectives are different propositions. A distributional assumption cannot justify a
universal claim about every invocation.

Specialization cannot rely on future route IDs, sequence lengths or tensor values
unavailable at compilation. Unknown control/addressing facts remain in the workload
domain or checked variants. Representation preparation and retained storage are
charged at the declared boundary; amortized conversion needs an explicit reuse
horizon. Lossy representation changes require semantic permission, not just a
better objective.

All compared bounds and feasible schedules share the same objective, conditions, model,
and time units. A physical floor cannot be paired with a predicted or measured upper
bound to claim guaranteed physical optimality.

## Search-space construction

[Execution](execution.md) defines legal choices independently of the optimizer. The
tuner consumes symbolic domains, dependent alternatives, and constraints. Different
branches may introduce different operations and further choices.

All decision families in the execution contract participate in one space, including
composition, layouts, completion-aware lifetimes and asynchronous pipelines. Domains
are dependent: decisions can introduce operations and further domains or constrain
earlier choices. A fixed decomposition or preferred placement list is not coverage
of the full form.

Decomposition choices implement a fixed logical computation. Candidate-dependent
piece shapes or counts must not feed back into portable arithmetic or effects.
Compare candidates under the same numerical permissions, including allowed merge
orders; reject partition-dependent source meaning before constructing this space.

A region is a mechanically described subset of assignments. Every subdivision preserves
coverage of the parent by construction, including dependent domains. Regions are removed
as infeasible only when their own constraints establish it. Compiler failures,
unsupported analysis, and timeouts cannot be reclassified as illegal executions.

Selection completeness is relative to the preserved execution family promised by
the [compiler](compiler.md#optimization-guarantees), not just the candidates its
current implementation happened to expose. Missing promised construction support
is a compiler failure or unresolved coverage, not an optimum over a silently smaller
space. Diagnostic restrictions remain explicitly scoped.

## Optimization procedure

```text
Construct legal symbolic domains
    → propagate constraints
    → derive region lower bounds from relaxations of those constraints
    → construct and validate feasible executions
    → exclude regions unable to improve the incumbent
    → cover or resolve the remaining frontier
    → establish frontier completion and validate the selected execution
    → materialize the selected execution as Tuned IR
```

Search operates on shared structures and partial assignments rather than requiring a
separate complete tree for every candidate. Bounds and constraints specialize as choices
resolve. Reuse of subproblems or derived analyses is valid only under compatible
identities and conditions.

Typed source identity compares floating literal representations, including the
sign of zero and NaN payload bits. Numeric floating equality cannot identify an
unchanged program or validate a retained selection against new input.

A retained choice owner may construct its selected member directly when it owns
the necessary execution context. This refinement must produce the same execution
or dependent domain as full-path construction under the bound request identity.
Unrecognized owners use full-path construction; construction failures remain
errors. Immutable execution plans may be shared across siblings, while each
assignment retains its own resolved decisions. Resumption retains the owner and
ordinal needed to recover an execution whose improved schedule becomes the
incumbent. This reuse changes construction work, not legal domains, objective
bounds, or the accounting required for a selected execution.

Metal launch grouping retains the prepared execution while resolving each launch's
own legal domain, including padding exclusions. Selecting the complete assignment
replans storage quantities for those dispatches without changing allocation
identities. Prepared implementation identity includes source conditions, ownership,
storage and synchronization plans; a lazily populated emission cache is derived
state and does not change that identity.

Retain structured iteration, predicates and symbolic resource relationships where
possible; full scalar expansion of each candidate is not the production scaling
strategy. Search decomposition requires independence or a frontier retaining the
resource/interface alternatives needed by enclosing decisions. Independently
choosing each child's best implementation can discard the best composition.

Deterministic traversal and ordering by sound bounds are allowed. Symmetry elimination
and decomposition require evidence that they preserve coverage and the optimum. Source
order, device names, magic thresholds, and observed kernel rankings do not supply
performance preferences.

The exhaustive implementation remains an independent small-instance oracle. The
production search uses constraint propagation, symbolic regions, checked relaxations,
and sound pruning to avoid enumerating every full realization. Practical time and memory
are qualification obligations; exactness is not weakened to meet a budget.

## Lower bounds and feasible schedules

A region lower bound is derived from a relaxation of the same [resource
constraints](accounting.md) used to evaluate its executions, as specified in the [lower-
bound specification](../../../specs/26-09-17/seismic-sound-lower-bounds.md). The tuner does
not own a separate cost interpretation, decision registry, legality mechanism, or bound
provider accepting arbitrary numeric assertions. Necessary demand is not the same as the
consumption of a chosen implementation. A roofline need not be attainable, and
minimizing candidate lower bounds does not select the fastest candidate.

A feasible upper bound for the model comes from a validated execution schedule,
including applicable allocations, resource reservations, dependencies, and timing. A
feasible schedule in an optimistic relaxation is not automatically feasible in the
original model. Resource models and objective evaluations must be derived from the
selected execution, not supplied as arbitrary scores.

Compiler-controlled schedules include executable local order, transfer issue/wait
sites, launch dependencies and admitted work-assignment policies. Hardware block
placement, warp issue, cache replacement and CPU out-of-order issue are machine
behavior. The best resource-feasible interleaving of that behavior is only an
optimistic bound unless the model establishes its achievability. Do not materialize
an unenforceable ideal schedule by merely retaining an unchanged program.

Model feasibility and physical achievability remain distinct. A model upper bound
requires a schedule achievable under that model's execution semantics; physical
claims additionally require qualified mappings and hardware assumptions.

A region can be excluded when its derived lower bound cannot improve a compatible
validated incumbent. Search state records the region, applicable bound, and exclusion so
coverage remains complete within the search itself. It does not construct a second graph
describing a proof of the search.

## Invariants and validation

Construction, transformation, resource analysis, and search own the following
invariants; existing stage boundaries validate the applicable global constraints:

1. Applicability of transformations and selected implementations to the computation.
2. Satisfaction of workload, numerical, effect, safety, and hardware constraints.
3. Derivation and feasibility of the incumbent's model objective.
4. Correctness and scope of region bounds and infeasibility conclusions.
5. Coverage of all admitted choices, including excluded regions.
6. Exclusion of every strictly better legal execution.
7. Agreement between the evaluated execution and the actual Tuned IR.

Small exhaustive cases check both coupled-domain coverage and sound pruning.
Documented source-refactoring pairs must retain their execution families and model
optima. Search limits preserve the frontier; they neither narrow the declared form
nor weaken the source-stability contract.

Costs are computed from the execution, not accepted from a caller or justified by
repeating a callback. Sound domain partitioning, relaxations, and pruning are compiler
algorithms over shared typed structures; their mathematical prerequisites are enforced
where applied. No independent proof language, certificate checker, or parallel semantic
interpreter is introduced. Time and memory limits leave an incomplete frontier; they
cannot turn unfinished search into an optimum.

## Result states and lifecycle

| Result | Meaning |
| --- | --- |
| Model optimum | A feasible execution and complete checked exclusion of better executions under the stated model. |
| Incomplete search | Incumbent if available, checked frontier bounds, coverage, and unresolved regions; no optimality claim. |
| Proven infeasible form | Checked evidence that the admitted execution domain has no feasible member. |
| Invalid request or compiler failure | Invalid inputs or failure to construct the promised supported model/space; not a performance result. |

For compatible sound lower and feasible upper bounds, a positive lower bound permits a
model-gap ratio. A zero lower bound gives no finite ratio. A nonzero gap does not
satisfy exact optimality. The qualified production path does not silently execute an
incumbent from incomplete search as if tuning had completed.

Instruction/operation derivation limits produce typed budget exhaustion, distinct
from unsupported analysis or invalid model construction. The frontier retains the
unfinished execution and its inherited bound alongside any incumbent. Increasing
those limits under the same semantic inputs retries that derivation; completed
models and schedule searches remain cached. Partial model construction can be
restarted from its retained IR without changing the legal family. Progress reports
unresolved choice regions, derivations, schedules and missing model mappings
separately, so an empty choice frontier does not imply completed optimization.

Cache reuse binds to semantic program/form/workload identities, hardware and mapping
contracts, objective, and analysis versions. Reusing a result under changed conditions
requires a checked implication. Native feedback from a compiled winner can inform
external qualification, never hidden continuation of candidate selection.
