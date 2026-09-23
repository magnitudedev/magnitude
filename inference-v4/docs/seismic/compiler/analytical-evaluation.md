# Analytical evaluation

Analytical evaluation predicts candidate performance from its actual physical demand and the target's service model. It chooses exploration and retention under an explicit objective; it does not establish semantic validity.

| Input | Output |
| --- | --- |
| Closed physical structure, applicability, target characterization and objective/workload assumptions | Symbolic or concrete cost predictions, uncertainty/model limitations and comparison decisions |

## Physical demand

Demand comes from the physical IR: instructions and numerical modes, participant assignment, accesses, representation decoding, scratch/register/storage use, transfers, synchronization, dependencies, launches and host/native control. Supporting operations count even when absent from source syntax.

The model consumes the same normalized layouts, liveness and resource structure as emission and runtime. It does not derive a second local layout or invent an alternative scratch plan. Source-level work supplies context; it is insufficient to predict actual traffic or time by itself.

Selected emission recipes contribute demand through the same operation meanings used by lowering. An unmodeled recipe produces an explicit model limitation; the evaluator cannot keep scoring the recipe it replaced.

Logical tensor axes remain the actual SSA operands used by execution. Any exact host expression is derived on the same value owner and retained through closure and import. The model reads that value fact; it cannot substitute allocation capacity for a device-produced logical extent. A missing sound extent bound returns a typed model limitation from the operation and whole-executable assessment. A valid operation cost always contains demands or an explicit semantic elision. Analytical unavailability leaves the candidate available to native feedback evaluation.

## Target supply

Characterization describes throughput, latency, occupancy limits, memory hierarchy and contention for the effective target. Acquisition belongs to analytical evaluation; other execution routes need no analytical profile.

Hard target limits filter legal choices. Service rates predict cost among those choices. Confusing the two can admit illegal candidates or unnecessarily exclude valid ones.

The model combines demand with dependencies and possible overlap. Taking the maximum of a few FLOP/bandwidth ratios can be a useful lower bound, but is not automatically an exact duration model: startup, critical paths, limited parallelism, synchronization and contention matter.

## Symbolic costs and search

Symbolic invocation geometry and conditional choices can produce symbolic costs. Region composition respects loop multiplicity, empty ranges, branch conditions and lifetime overlap. Expected branch or workload costs require a declared distribution; otherwise the model must state the objective it actually predicts.

Solver-driven search uses domain-owned choices and applicability constraints. Solver output re-enters through checked coordinate binding and source-directed construction; an arbitrary solver assignment is external data, not a complete candidate. A timeout preserves pending search, and a heuristic restriction must be reported as a search restriction rather than domain membership.

| Result | Meaning |
| --- | --- |
| Exact structural quantity | Count or resource fact derived from the represented execution. |
| Safe bound | An envelope valid over the stated domain; may be too loose for ranking. |
| Prediction | Estimated performance under a stated service model. |
| Unknown/model limitation | The model cannot yet predict this operation or composition adequately. |

Unknown cost must not become zero cost or an execution correctness failure. The evaluator may defer comparison, improve characterization or use feedback services when configured. Predictions remain separable from applicability.

## Quality and reporting

The [optimality equation](../overview.md#the-optimality-equation) requires exact cost evaluation for an ideal global choice. Practical model accuracy is an approximation, and reported optima are relative to the modeled and searched scope. Measured agreement on a finite suite supports model development but does not establish exhaustive compiler correctness or universal cost accuracy.

Before publication, selected candidates pass through [native formation](native-emission.md); reflected artifact facts may require the evaluator to revise its prediction or retention. This reconciliation happens before `SelectionPolicy` finalization.

