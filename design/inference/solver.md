---
applies_to:
  - inference-v4/solver/**
---
# Structured solving

The solver optimizes finite, typed constraint models. It has no language, kernel,
hardware or native compiler dependencies. Domain adapters own correspondence
between their real systems and these models. The solver establishes model
optimality only; it cannot establish physical or numerical correspondence.

A model owns finite domains, guarded constraints, nonnegative objective terms,
and explicit resource interactions. Production bounds derive from those same
terms. Unsupported coverage remains unresolved. Invalid inputs and arithmetic
failures are errors, never infeasibility. Objective arithmetic is checked.

One search interface accepts an immutable model and incremental limits, with
selectable exact or neighborhood implementations. Both return the same outcomes:
`Optimal` and `Infeasible` require proofs about the original model; `Incomplete`
can carry a complete validated feasible witness without an optimality proof.
Exact remains the default. A caller's policy for accepting an unproved witness
does not change the model or the meaning of an optimal solution.

Exact search uses AND/OR branch-and-bound. Independent residual components can compose
additively only after all constraints, guards and objective scopes are accounted
for. A cumulative resource couples every activity in its scope. Shared analysis
never implies shared execution: a producer is charged once only when the model
explicitly represents one producer.

Memoization compares exact residual context within an immutable model. A model-local
shared store reuses completed residual proofs across order-preserving variable
renamings and neighborhood repairs. Keys retain every scoped domain, guard,
constraint, objective and fragment definition; names are excluded. This is
structural equality under a checked rebinding, not general graph isomorphism.
Occurrences still own separate assignments and execution costs. Hashes
accelerate equality and do not establish it. Parent cost cutoffs cannot turn a
partially searched context into an unconditional optimum or infeasibility claim.
Completed payload eviction preserves parent proofs and live pending coverage;
evicted results can be recomputed. Insufficient pending storage remains a memory
stop rather than a reason to discard coverage.
Bounds, feasible witnesses and unresolved coverage survive budget interruptions.
An incomplete incumbent cannot be converted into an optimal solution.

Every upper witness is validated against the original active constraints. A
completed optimum excludes every strictly better admissible assignment. Missing
analysis can be skipped only when an independently sound bound excludes its
entire region. Budget exhaustion never narrows the legal model.

For exact search, the supported complete class is a fully specified finite model with the built-in
constraints, finite guarded definitions and sufficient retained search storage.
Exact domain partitions provide exhaustive schedule refinement for explicit
finite occurrences. Compact additive repetition requires independent copies
and recurring/reset boundary state. General coupled parallel repetition is not
covered by that reduction. Missing construction, analysis and unavailable
refinement are separate typed obligations; more identical budget alone need not
resolve them.

Independent components can aggregate by sum or maximum. A maximum proof may
close from aggregate bounds without finishing every component proof, provided
every component has a checked feasible witness. Coupled finite repetition uses
proved compact batching/exclusive-stage pipeline reductions only for their
declared subclasses; other finite templates retain all occurrences and shared
constraints in joint search. These reductions do not cover general repeated
state or imply that finite expansion scales to large counts.

Neighborhood search conditions repairs on temporary variable restrictions,
retains complete candidates, and explores overlapping groups derived from typed
relations. Crossing constraints remain active. Restricted proofs and bounds
cannot become global claims. Repairs share immutable factor definitions and
completed proofs only under their full residual context. Established conflict
proofs may exclude stronger conjunctions with narrower premise domains, never
wider domains, missing factors, or cutoff-only failures. Cache eviction forgets
reusable work without deleting pending search coverage. Global
incumbents are checked against the original model and never worsen. Heuristic
search promises no eventual proof; failure to find a witness is not infeasibility.
Seeded work-limited execution resumes without restarting pending repairs.
Resource pressure and critical completion influence neighborhood priorities;
constraint-derived numeric boundaries are preferred proposals, never domain
restrictions. Propagation derives mandatory resource ordering from positive
minimum durations, release-suffix energy, and threshold-demand incompatibility
cliques. These deductions are shared by exact search and neighborhood repairs.

The initial subproblem interface is the residual constraint relation under fixed
shared boundary values. Scalar costs compose only when every external relation
has been accounted for by that interface. Resource profiles and lifetime
relationships cannot be collapsed to isolated duration or peak storage without
a separate sufficiency argument. Typed symbolic arithmetic preserves declared
integer family relationships; constraints remain the authority for bounds.

Soundness, eventual finite completion and practical compiler completion require
separate evidence. Compiler-derived unresolved-family coverage, objective
controllability and native reconstruction belong to adapters. Synthetic results
and fixed-execution comparisons do not establish those integration gates.

Tests compare tiny instances with exhaustive evaluation, validate conditional
composition, distinguish producer identity from memo identity, preserve
resource coupling, and compare resumed with uninterrupted solves. The separate
lab reports observed work, time-to-quality and time-to-proof with censored
timeouts; no performance claim follows merely from compact domains or repeated
structure.
