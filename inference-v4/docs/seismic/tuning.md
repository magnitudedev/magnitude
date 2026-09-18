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

Selection retains the scheduling representation with its witness. Flat schedules
and structured repeated schedules share the same feasibility, objective-bound,
frontier-completion and applicability requirements. A structured witness need not
be expanded merely to retain it in a compiled artifact. Consumers that require
explicit instruction start times must request a flat witness and handle its
absence; they cannot interpret an empty graph as a structured schedule. Bounded
expansion may invoke the existing exact scheduling oracle for small structured
models. Exceeding that bound preserves an unresolved result, even when a
feasible compact schedule is available. Repeated parallel bodies can retain
bounded-concurrency wave schedules without flattening; feasible waves alone do
not establish that every legal interleaving has been covered by the search.
Mandatory serial composition and repetition can refine their independent
subproblems through retained exact scheduling searches. A repeated body shares
one frontier across its visits. Subproblem durations add only across required
serial boundaries; unfinished parallel regions remain unresolved.
Parallel repeated bodies also share internal refinement, with feasible waves
derived from the refined body's peak resource use and bounds from mandatory
residency. This improves bounds without restricting the parallel region's
interleavings; it is completed only when the bounds meet.
Oversized heterogeneous parallel compositions retain child searches as well.
Their lower bound includes the maximum child bound. Checked child schedules may
start together when the sum of their peak reservation envelopes fits every
residual resource capacity, including resident scopes. Otherwise a serial
composition remains only a feasible upper bound; it does not remove staggered
or interleaved schedules from the legal family. Refinement can close an optimum
when these bounds meet without expanding repeated children. General contention
between overlapping children still requires further scheduling refinement.

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

The implementation frontier is indexed by derived lower bound and exact decision
path. Each globally selected region is refined along one dependent path to a leaf,
choosing the smaller bound at each interval split and retaining every sibling.
The next leaf begins at the globally smallest frontier bound. These bounded dives
prevent weak shallow bounds from starving either the first feasible construction
or improvements to an existing incumbent. Budget exhaustion returns an unfinished
dive to the same frontier. This traversal never authorizes executing an incomplete
incumbent or excludes unvisited choices.
A new interval is relaxed once before entering that frontier; resumption
refreshes retained intervals so larger derivation budgets can strengthen them.
Complete derived models contribute their dependency and residency bounds before
selection allocates a scheduling frontier. Identical scheduling models share one
retained search across implementation paths. A structural hash only indexes possible
matches: full model equality, including resources, units, constraints, mapping gaps,
model relationship and expansion limit, establishes reuse. Each source path retains
its own inherited bound, reconstruction owner and coverage. A shared schedule is
advanced once per resume call and is released when its final path retires. This
sharing does not identify different implementations or discard any legal choice.

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

The frontend retains the checked, partitioned computation and the inliner's
captured shape/view facts before backend-body selection. It retains the resolved
bodies again for value composition, then retains completed fusion and producer
sharing before representation selection. Every stage refines the same IR, keeping
preceding decision records and global ordinals. Body alternatives therefore reuse
normalization, output grouping and contraction partitioning; packet-width choices
also reuse resolved bodies and value composition. Each stage may replay from its
own boundary. This does not retain every intermediate analysis or eliminate model
derivation per realization.

Metal execution preparation retains its existing mapping, fold ownership, load,
storage, reduction and allocation boundaries. Refinement replays only the current
stage's prefix; later storage and allocation siblings share completed folded IR
and resolved plans. Retained owners compare the prepared computation, selected
configuration and stage prefix under the same request identity. Diagnostics and
full-path reconstruction consume these same stage functions.
Decomposition and coordinate-mapping choices retain their lowered computation,
legal numeric domain and device conditions as well. Their direct refinement no
longer reconstructs the portable frontend while visiting backend siblings.

Forced Metal domains are resolved by their owning stage without adding a search
node. Remaining execution and launch-grouping regions carry a conservative
necessary dispatch-service bound: each actual launch remains, and threadgroup
count is bounded below using every remaining legal grouping. The bound uses the
same primitive service expansion as concrete accounting. Body work, residency
lifetimes and ordering are dropped, so it is intentionally weaker than a full
execution cost and cannot establish native performance.
Earlier decomposition regions retain selected coordinate counts as well. The
maximum remaining coordinate width bounds work items below, while the minimum
remaining split count includes required main and merge launches. Static external
dense publications contribute their mandatory scalar writes before decomposition;
unknown branches take the minimum guaranteed publication count, and private
intermediates or unsupported loops contribute nothing. The bound maximally packs
scalar writes into subgroups and uses the same relaxed transaction expansion as
terminal accounting. These facts also apply to retained frontend choices whose
external publications survive composition.

After an incumbent exists, a completed execution can first supply aggregate demand
from its own terminal walk, retaining admitted loop multiplicities symbolically.
Known mandatory work can contribute even when later control or a service mapping
is unresolved; it is a lower bound, never a complete execution estimate.
If the resulting bound excludes that execution,
selection retains the bound and releases its construction state without allocating
a scheduling graph. This also applies when resuming a previously budget-limited
derivation. Otherwise ordinary schedule construction continues and checks any
feasible result against the retained bound. Incomplete counting or missing
service mappings can only omit the unresolved contribution; neither establishes
infeasibility or authorizes a feasible upper bound.
The pruning walk retains full dispatch groups as symbolic coordinate intervals,
scaling only operations established over the entire interval and visiting the
partial group separately. It must not enumerate the dispatch before reaching
structured scheduling. Immutable retained accounts are shared across interval
relaxations instead of copying their operation and address terms.

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
models and schedule searches remain cached while they can improve the incumbent.
A dominated leaf retains its derived lower bound and releases its scheduling
model, frontier and deferred execution. The incumbent keeps its feasible witness,
and unresolved leaves keep the state needed to resume. These summaries remain
inside the original selection progress and immutable search identity. A feasible
candidate below an inherited region bound is an analysis error, not an optimum.
Partial model construction can be
restarted from its retained IR without changing the legal family. Progress reports
unresolved choice regions, derivations, schedules and missing model mappings
separately, so an empty choice frontier does not imply completed optimization.

Cache reuse binds to semantic program/form/workload identities, hardware and mapping
contracts, objective, and analysis versions. Reusing a result under changed conditions
requires a checked implication. Native feedback from a compiled winner can inform
external qualification, never hidden continuation of candidate selection.

## Executable selection boundary

Implementation alternatives remain internal compiler search structures. The CLI
exposes source checking, printing, logical planning and ABI binding generation;
it does not accept choice ordinals, load/storage preferences, launch sizes or
partition settings. The former unselected `run`, `lower`, `native`, `inspect`,
`explore`, `choices`, `account` and `calibrate` commands have been removed.
A future executable CLI must consume completed automatic selection, with no
explicit-candidate or incomplete-incumbent fallback.

Compiler tests may inspect alternative IR and derived accounts. Such construction
does not authorize a runtime executable or count as automatic optimization or
performance parity. The fixed-candidate model runners and direct Metal plan
executor are removed; model and weight-import APIs accept automatic selection
settings instead of physical implementation assignments.

Completed choice records hold weak diagnostic references to their construction
owners. Unresolved regions and recoverable executions keep the strong references
needed for refinement and resumption. Once a region is excluded by its derived
bound, only its exact indexed path coverage and bound remain; its prepared IR is
released. Historical decision inspection can therefore return no owner for a
completed region. This changes memory retention, not domain coverage or bounds.

A retained Metal traversal stage carries a budgeted body account keyed by the
complete workload. Every remaining loop width preserves memory accesses, floating
arithmetic and floating conversions, matrix operations, and barriers occurrence-for-occurrence;
index arithmetic, checks, integer conversions, and loop control are relaxed away. These retained
operations give one sound demand bound for the entire remaining traversal region,
including later legal regrouping of its work items. The account uses the same
terminal walker and hardware service mappings as completed execution analysis.
An exhausted walk contributes only its observed mandatory prefix. Missing mappings
contribute no bound. Refinements reuse completed accounting; increasing a request's
derivation budget can extend a previously exhausted analysis. Refinements that leave
terminal operations and launch geometry unchanged share their immutable emission
and workload-bound account across preparation stages and completed executions.
Changing either replaces both retained results before analysis can reuse them.

Retained terminal transfer domains also derive a necessary-work bound from their existing baseline execution. The bound keeps only operations preserved by every remaining transfer and traversal choice: floating arithmetic, existing private/shared accesses, writes, and collectives. It excludes device reads because vector covers can combine them, and excludes control, checks, and integer address work that later refinements can remove. The stage caches this same-walker account by workload and derivation budget; it does not compile candidates or use native measurements. Partial accounts contribute only observed mandatory work. This can exclude an entire transfer region before constructing its descendant traversal schedules.

Launch-grouping regions retain the selected body's invocation account as well as
its dispatch geometry. Matrix, floating-point, collective and memory work stays
fixed per logical subgroup across the grouping domain; the bound combines that
mandatory work with the minimum remaining launch/group demand. Address arithmetic,
control and constant-parameter reads are excluded. This uses the same terminal
walker and hardware mappings as completed executions, and caches only under the
exact prepared execution and workload. Incomplete derivations contribute only
the observed mandatory prefix.
