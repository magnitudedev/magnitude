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

All compared bounds and feasible schedules share the same objective, conditions, model,
and time units. A physical floor cannot be paired with a predicted or measured upper
bound to claim guaranteed physical optimality.

## Search-space construction

[Execution](execution.md) defines legal choices independently of the optimizer. The
tuner consumes symbolic domains, dependent alternatives, and constraints. Different
branches may introduce different operations and further choices.

Construct/lowering alternatives, instruction covers, tiling, mapping, storage,
recomputation, fusion, synchronization, and launches participate in one space. A fixed
decomposition or preferred placement list is not coverage of the full form.

A region is a mechanically described subset of assignments. Every subdivision preserves
coverage of the parent by construction, including dependent domains. Regions are removed
as infeasible only when their own constraints establish it. Compiler failures,
unsupported analysis, and timeouts cannot be reclassified as illegal executions.

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

Cache reuse binds to semantic program/form/workload identities, hardware and mapping
contracts, objective, and analysis versions. Reusing a result under changed conditions
requires a checked implication. Native feedback from a compiled winner can inform
external qualification, never hidden continuation of candidate selection.
