# Seismic tuning

The tuner resolves Lowered IR into Tuned IR using the legal execution space and
derived resource/dependency model. Its objective is exact optimization over the
declared form, with independently checked evidence. Native candidate compilation,
resource queries, benchmarks, heuristic scores, and compile-and-try fallback are
excluded from this path.

## Input and output boundary

| Input | Required meaning |
| --- | --- |
| Lowered IR | Backend computation and the complete supported legal alternatives. |
| Execution form | Admitted transformations, implementations, and choice domains. |
| Workload domain | Shapes, layouts, representations, values/predicates where relevant, aliasing, and initial-state conditions. |
| Implementation contracts | Numerical/effect legality, resource derivation, and emission mappings. |
| Hardware contract | Bound machine model, relevant axioms/parameters, and operating conditions. |
| Objective | Quantity to minimize, execution boundary, units, and treatment of dynamic scenarios. |

A completed result contains the actual selected Tuned IR, its derived account,
checked feasible objective witness, optimality certificate, and retained conditions.
Physical lower bounds, invocation applicability, and mapping qualification accompany
it as distinct claims; a model optimum does not manufacture those claims.

The emitter consumes the selected execution directly. It neither replays a bag of
settings against a changed program nor chooses an unmodeled implementation.

## Objective semantics

The objective identifies exactly which work is timed: for example, completion of
a resident invocation's full launch sequence, including its required dependencies
and specified submission work. Compilation, model loading, transfers, and readback
are included only when the declared boundary includes them.

Known bindings are specialized. Remaining runtime variation stays in the workload
domain or in explicit checked specializations. Per-binding, worst-case, and
expected-value objectives are different propositions. A distributional assumption
cannot justify a universal claim about every invocation.

All compared bounds and feasible witnesses share the same objective, conditions,
model, and time units. A physical floor cannot be paired with a predicted or measured
upper bound to manufacture a physical optimality certificate.

## Search-space construction

[Execution](execution.md) defines legal choices independently of the optimizer.
The tuner consumes symbolic domains, dependent alternatives, and constraints.
Different branches may introduce different operations and further choices.

Construct/lowering alternatives, instruction covers, tiling, mapping, storage,
recomputation, fusion, synchronization, and launches participate in one space.
A fixed decomposition or preferred placement list is not coverage of the full form.

A region is a mechanically described subset of assignments. Every subdivision
retains evidence that its children cover the parent. Illegal regions need checked
infeasibility evidence. Compiler failures, unsupported analysis, and timeouts cannot
be reclassified as illegal executions.

## Optimization procedure

```text
Construct legal symbolic domains
    → propagate constraints
    → derive and check region lower bounds
    → construct and check feasible executions
    → exclude regions unable to improve the incumbent
    → cover or resolve the remaining frontier
    → check the complete optimization certificate
    → materialize the selected execution as Tuned IR
```

Search operates on shared structures and partial assignments rather than requiring
a separate complete tree for every candidate. Bounds and constraints specialize as
choices resolve. Reuse of subproblems or proof results is valid only under compatible
identities and conditions.

Deterministic traversal and ordering by sound bounds are allowed. Symmetry elimination
and decomposition require evidence that they preserve coverage and the optimum.
Source order, device names, magic thresholds, and observed kernel rankings do not
supply performance preferences.

The exhaustive implementation remains an independent small-instance oracle. The
production search uses constraint propagation, symbolic regions, checked relaxations,
and certified pruning to avoid enumerating every full realization. Practical time
and memory are qualification obligations; exactness is not weakened to meet a budget.

## Lower bounds and feasible witnesses

A region lower bound comes from the [accounting proof system](accounting.md), with
the detailed rules in the [lower-bound specification](../../specs/26-09-17/seismic-sound-lower-bounds.md).
Necessary demand is not the same as the consumption of a chosen implementation.
A roofline need not be attainable, and minimizing candidate lower bounds does not
select the fastest candidate.

A feasible upper bound for the model comes from a checked execution witness,
including applicable allocations, resource reservations, dependencies, and timing.
A feasible schedule in an optimistic relaxation is not automatically feasible in
the original model. Resource models and objective evaluations must be derived from
the selected execution, not supplied as arbitrary scores.

A region can be excluded when its verified lower bound cannot improve a compatible
checked incumbent. Every such exclusion remains represented in the coverage proof.

## Independent checking

The checker establishes:

1. Applicability of transformations and selected implementations to the computation.
2. Satisfaction of workload, numerical, effect, safety, and hardware constraints.
3. Derivation and feasibility of the incumbent's model objective.
4. Correctness and scope of region bounds and infeasibility conclusions.
5. Coverage of all admitted choices, including excluded regions.
6. Exclusion of every strictly better legal execution.
7. Identity agreement between the witness and the actual Tuned IR.

The checker does not trust optimizer traversal history, reported best costs, or a
callback that recomputes the same unsupported assertion. Its trusted rule vocabulary
is explicit and versioned. Proof checking has its own size/work limits and cannot
accept an unchecked remainder.

## Result states and lifecycle

| Result | Meaning |
| --- | --- |
| Verified optimum | A feasible execution and complete checked exclusion of better executions under the stated model. |
| Incomplete search | Incumbent if available, checked frontier bounds, coverage, and unresolved regions; no optimality claim. |
| Proven infeasible form | Checked evidence that the admitted execution domain has no feasible member. |
| Invalid request or compiler failure | Invalid inputs or failure to construct the promised supported model/space; not a performance result. |

For compatible certified lower and upper bounds, a positive lower bound permits a
model-gap ratio. A zero lower bound gives no finite ratio. A nonzero gap does not
satisfy exact optimality. The qualified production path does not silently execute
an uncertified incumbent as if tuning had completed.

Cache reuse binds to semantic program/form/workload identities, hardware and mapping
contracts, objective, and rule versions. Reusing a result under changed conditions
requires a checked implication. Native feedback from a compiled winner can inform
external qualification, never hidden continuation of candidate selection.

## Conformance

Independent finite oracles must agree on optima and reject omitted branches,
incorrect pruning, fabricated costs, and infeasible witnesses. Qualification also
measures search/proof cost as domains grow, exercises dependent choices and dynamic
conditions, and demonstrates useful performance on representative and held-out
CPU, CUDA, and Metal workloads. These checks complement, rather than replace,
arguments for the trusted rules and complete backend coverage.
