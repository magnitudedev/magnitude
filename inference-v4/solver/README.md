# Structured combinatorial solving

`magnitude-solver` minimizes a nonnegative integral objective over a finite,
typed constraint model. It lives in the `inference-v4` workspace and has no
language, compiler, kernel or hardware dependencies. Exact and neighborhood
search share `Model`, `Search`, `Limits`, and `Outcome`. The evaluation program is
in [lab](lab/README.md).

```rust
use magnitude_solver::{Domain, ModelBuilder, Search, Options, Limits, Outcome};
use magnitude_solver::model::Cost;
use std::sync::Arc;

let mut builder = ModelBuilder::new();
let representation = builder.variable("representation", Domain::boolean());
// These costs define a hypothetical mathematical example.
for costs in [[2, 8], [9, 3], [9, 4]] {
    builder.cost(Cost::Table {
        variables: vec![representation],
        entries: vec![(vec![0], costs[0]), (vec![1], costs[1])],
    });
}
let mut search = Search::new(Arc::new(builder.build()?), Options::default())?;
match search.advance(Limits::default())? {
    Outcome::Optimal(solution) => assert_eq!(solution.cost(), 15),
    Outcome::Infeasible => panic!("this instance is feasible"),
    Outcome::Incomplete(progress) => {
        // Retain `search` and call advance with a further incremental budget.
        // A diagnostic incumbent has a different type from an optimal solution.
        println!("unresolved lower bound: {}", progress.lower_bound);
    }
}
# Ok::<(), magnitude_solver::Error>(())
```

Set `Options.algorithm` to `Algorithm::Neighborhood(NeighborhoodOptions::default())`
to improve feasible candidates with large-neighborhood search. Exact remains the
default. An incomplete result can contain a complete legal implementation;
`Incomplete` describes missing proof/coverage, not a partial executable witness.
Only globally proved results are `Optimal` or `Infeasible` with either algorithm.
The production compiler's requirement for an optimum is unchanged.

Neighborhood search first checks the domain-minimum assignment, then uses
unrestricted constraint search when a legal seed is still needed. It releases
connected choices and dependent values inferred from typed factors, holds the
remaining values fixed, and runs bounded repair. All crossing factors remain
active. Repairs use isolated proof caches and an internal result type; their
conditional bounds and completed local proofs are never global certificates.
Every candidate is checked against the original model.

The default settings use seed 0, up to 8 requested primary choices per move,
4096 repair work units, 4 population members, and a restart after 32 unsuccessful
repairs. Dependency closure may release more than 8 variables. Fixed operator
selection, forced numeric proposals and seeded uphill acceptance explore beyond
greedy improvements; the best-ever witness is retained separately. Set
`exploration=false` for greedy acceptance and `restart_after=u64::MAX` to disable
restarts. Width one is the coordinate-search ablation, still releasing dependent
outputs. These are experiment settings, not a quality or latency guarantee.

An interrupted repair resumes without changing its assumptions or spending
its remaining work on external time/memory stops. Statistics separate repair
work, attempts, released-variable counts, improvements, failures, and restarts.
Construction of model adjacency is included in `Search::new`; the lab charges
per-instance construction and initialization in its end-to-end times. A repair
setup/validation operation, like an existing factor evaluation, is a cooperative
work unit that can cost more as the model grows. The engine does not expand
numeric domains to sample them or invent compact semantics for repeated work.

Run targeted checks:

```sh
cargo test --manifest-path inference-v4/solver/Cargo.toml
cargo test --manifest-path inference-v4/solver/Cargo.toml --features serde
```

## Supported model and exactness boundary

Variables have explicit finite i64 sets, intervals or arithmetic progressions.
Constraint/objective variants are closed, inspectable types. Scopes are derived
from their operands, including guards and every possible resource effect. Table
objectives must cover the complete active domain. Arbitrary user callbacks
cannot introduce unverified lower bounds or infeasibility claims.

A `Search` owns one immutable validated `Model`. Its cache cannot cross model,
objective or unit boundaries. `Solution` is privately constructed after complete
exclusion of improvement. `FeasibleSolution` is a diagnostic assignment, checked
against the original model, with no optimality guarantee. Arithmetic overflow,
malformed models and broken internal invariants are errors. They are never
infeasibility, saturated costs or an optimum. After a search error, subsequent
advances return that error; start a new search after fixing the input.

All complete claims are relative to the supplied mathematical model. Source
family correspondence, controllability of decisions, machine policy and native
qualification belong to a domain adapter. Solving a hypothetical ideal schedule
does not establish that a compiler can emit that schedule or that hardware
will follow it.

## Search and interface sufficiency

The exact engine performs deterministic AND/OR branch-and-bound with propagation,
exact context caching and disjoint domain partitions. It uses explicit frames,
not recursive search calls. A pending component retains its scoped domains;
one reusable dense workspace serves built-in factor evaluation.

Every unresolved factor belongs to one residual component. Components may split
only after all their shared variables are fixed. Exact factors are discharged
only when their constraint and cost hold for **every** assignment in the current
domains. Their costs and surviving assignments stay in the parent. This rule is
why narrowing a variable and then dropping its final constraint must retain a
value from its narrowed domain.

The initial interface is an **unevaluated residual constraint relation with
fixed boundary contexts**. A cached scalar optimum is sufficient precisely
because all external factors can observe only the fixed shared variables and
the additive scalar contribution. Parent feasibility cannot inspect a private
variable eliminated by this interface. If an unresolved time, representation,
lifetime or resource relation crosses a proposed boundary, that relation keeps
the components in joint search. A shared cumulative factor therefore prevents
an invalid additive split.

This is a conservative sufficient interface, not a claim that timing boundaries
are small. The current cache uses model-local variable/factor identities and
exact domain equality. It does not infer isomorphism between differently named
occurrences. Reusing a mathematical result does not merge production identities
or remove another occurrence's cost. No profile dominance, generic Pareto
frontier, or physical sharing is inferred.

Bounds are derived from the original typed factors. OR nodes retain the minimum
of all alternative bounds; independent additive AND nodes retain the sum.
Previously established regional bounds survive refinement. A child excluded by
a parent's incumbent is not marked infeasible or globally optimal, so another
parent can continue its shared work. Exact hash equality never substitutes for
structural key equality.

## Conditional construction

`Fragment` definitions are finite immutable models with explicit interfaces.
`instantiate` eagerly maps their factors; `instantiate_lazy` retains a complete
typed bundle and materializes one child per work unit after its guard becomes
active. Inactive private variables have one canonical value. Complete possible
scopes remain visible before activation, and full original-model evaluation
still checks every active descendant.

Definition validation, interface mapping and variable allocation are eager.
Lazy factor activation does not claim constant-size construction for arbitrary
parameterized templates or millions of distinct occurrences. No opaque generator
may introduce an undeclared dependency. Nested definitions are bounded to 128
levels by model validation.

## Scheduling and repetition classes

Explicit finite schedules use activity start/duration/end equations, optional
presence, precedence, half-open capacity reservations and event lifetimes.
Variable domains declare the horizon. A caller may justify that horizon with an
independently validated feasible schedule; the library never invents a cutoff
and then declares an unbounded scheduling problem infeasible.

Each explicit occurrence has its own variables. Coupled parallel occurrences
are one joint finite model, including every shared capacity and cross-occurrence
edge. This gives exhaustive finite refinement even when a convenient initial
wave schedule or resource lower bound does not close the optimum. It makes no
compact-completion promise for large repeated coupled workloads.

`IndependentRepetition` is a separate exact compact class: each occurrence is a
complete independent copy, with no shared resource, state or boundary relation,
and the objective is additive. Solving one body and checked multiplication by
the occurrence count is exact for that class. Its caller must establish that
its real process has this recurrence/reset semantics. Parallel makespan is not
covered by additive multiplication.

`Composition` aggregates truly independent component models by sum or maximum.
For maximum, an aggregate proof may close when every component has a feasible
witness and its aggregate lower bound meets the upper, even while an individual
component's optimum remains unresolved. Sharing a model identity never merges
execution occurrences.

`CoupledRepetition` supports proved compact identical-activity batching and
exclusive-stage pipelines. Other supported finite templates use a joint finite
expansion with all cross-copy precedence/resource constraints. Compact witnesses
are checked by their construction rules; a separate exhaustive tiny time-grid
oracle tests both reductions against finite expansion. General repeated state
and resource interactions do not acquire a compactness guarantee from this API.

## Completion argument and resource limits

For exact search on a valid fully constructed finite model with supported
constraints, sufficient resources and repeated positive work budgets:

1. Propagation only narrows finite domains. A fixed-point pass with no change
   finishes; a changed pass removes values or reaches the same finite normal form.
2. Fragment activation processes a finite number of declared children, one at a
   time. No recursive generation or novel alternative is introduced.
3. Exact factor discharge reduces the residual factor set. Decomposition produces
   smaller disjoint residual factor sets. Normalization retains a strictly
   narrowed context; it does not create new assignments.
4. A branch partitions a nonsingleton finite domain into two nonempty, disjoint
   regions with strictly smaller cardinality. Their union is the original domain.
5. At singleton assignments all supported factors evaluate exactly. The finite
   graph therefore has only finitely many distinct refinements and leaves. Both
   traversal policies select pending work until all improvements are excluded.

No pending region is silently truncated. Under memory pressure, completed
payloads can be discarded while their summaries preserve parent proofs. Live
pending coverage is retained and node identities/parent links are rebound after
compaction. A later cache miss recomputes discarded work. A memory limit too
small for pending coverage returns a memory stop; repeating that same limit
need not make progress. Eviction/resumption tests cover guarded fragments,
unresolved coverage, and crossing cumulative-resource constraints.

Work/time limits apply per `advance`; memory is a retained-search-storage target,
excluding the immutable input model, allocator overhead and temporary factor
work. Validation/construction happens before search timing. A factor evaluation
is one cooperative work unit and may itself be expensive; the lab additionally
uses process time/RSS watchdogs. No strict wall-clock preemption is promised.

Budget exhaustion retains resumable work. Declared missing construction,
analysis or refinement requires resolving that obligation or proving its region
cannot improve the incumbent; more identical work budget alone need not help.
The finite fully specified class has no restricted schedule repertoire that can
be repeatedly exhausted without an exhaustive next partition.

This is the exact implementation's eventual-completion argument, not a useful deadline. Worst-case work
and memory remain exponential. Practical completion must be measured separately
on representative unresolved compiler families, including construction and
proof, without hiding alternatives or treating censored runs as successes.

## References and evidence

The algorithm follows context-cached AND/OR constraint optimization; see the
[theoretical specification](../../specs/26-09-18/seismic-independent-optimizer.md)
and its [architectural corrections](../../specs/26-09-18/seismic-optimizer-corrections.md)
for Dechter, Mateescu and Marinescu references and the required evidence gates.
The lab contains a separately implemented direct evaluator, chain dynamic
programming and a pinned CP-SAT comparison. Their scope and measured results
must be read separately from the library's semantic guarantees.

The [initial performance report](lab/results/2026-09-18/README.md) publishes 270
measured runs, ablations, censored cases and independent checks of every recorded
bound and completed cost. The [CP-SAT fixtures](lab/evidence/cp-sat-corrections.json)
cover the packing and repeated coupled scheduling corrections. These are finite
solver evidence, not proof of practical completion for unresolved kernel families.
The [compiler-family inventory](../../specs/26-09-18/solver-gate0-inventory.md)
records actual unresolved grouping domains and compact repeated accounting
structure, including the coverage that fixed-schedule translation does not supply.

The [neighborhood-search report](lab/results/2026-09-18-neighborhood/README.md)
compares complete-candidate search, exact incumbents, and ablations on saved
synthetic models. Its [verification record](lab/results/2026-09-18-neighborhood/verification.md)
documents the combined correctness checks and the exact-solver optimization
work still outstanding. The approximate engine is available for experiments;
this evidence does not authorize production compiler acceptance of unproved
solutions.
