# Seismic tuning

**Tuning is one joint selection: authored implementations, contiguous fusion groups,
and numerical sites, chosen together under one estimate, within a budget.** It
happens in IR, before any native source exists. Its result is a checked witness
with an honest proof status.

## Decisions

| Decision | Domain | Identity |
| --- | --- | --- |
| Implementation | The applicable candidates of one occurrence | The static call in its parent candidate; the entry is occurrence zero |
| Grouping | The legal intervals of one sequence; selected intervals cover every unit exactly once | The authored block |
| Number | The finite value domain of one site | The static binder (width) or `merge` axis (parts) in its owning candidate |

Nothing else is a decision. Piece counts, visits, dispatch geometry, byte counts,
addresses, and masks are arithmetic over site values. Load mode, tile storage,
allocation, transfer width, reduction algorithm, and unrolling are fixed backend
rules ([Backends](backends.md)).

### Guards

A candidate is active when it and all its ancestors are selected. Its child
occurrences, sites, sequences, requirements, constraints, and cost factors exist only
while it is active. Inactive decisions carry no cost and no constraint, are absent
from the witness, and cannot force assignments elsewhere. Two occurrences may select
different bodies of the same function.

### Candidates and requirements

A candidate with a `where` predicate over a structural extent carries numerical
requirements on the bound site: multiple-of (atom and packet alignment), at-least,
at-most, equal, and divides (`full`). The candidate is selectable only with site
values that satisfy them. A requirement can name a site owned by an ancestor, which
couples the child's choice to the parent's width; that coupling is exported, not
resolved greedily.

### Sites and domains

The family gives each site its static extent. The backend narrows it to a finite
value domain by a documented rule that does not consider cost. A single-value
domain is not a search variable but keeps its identity in the witness. A site with
no admissible value makes its owner unselectable.

### Intervals and exact cover

For each active sequence the solver selects intervals `[start, end)` such that every
unit is covered exactly once. Only backend-listed intervals exist. An interval may
require particular child candidates and may tie pairs of sites to equal values
while it is selected; separate execution leaves them independent. Pairwise legality
does not imply a longer interval.

## Estimate and decomposition

The objective is the sum of local cost factors supplied by the backend. Each factor
names exactly the decisions it depends on:

- the candidates whose selection activates it,
- the intervals whose selection activates it,
- the sites its value reads.

**Decomposition is a compiler-to-solver requirement.** No factor or constraint hides
the program behind an all-variable callback. Independent occurrences export
independent factors, so the solver's residual decomposition solves them separately:
twenty independent three-way choices cost sixty local evaluations, not `3^20`
assignments. Real coupling is kept and only real coupling: a shared site, a
requirement on an ancestor's site, an interval's equalities, a capacity limit over
several tiles of one kernel. A parent's total may select a locally slower child.

Factors and constraints are tabulated over the product of their scope's domains.
A scope whose product exceeds the tabulation bound is *analysis unavailable*, not a
truncated model.

Hard limits are constraints, never costs. Preferences are costs, never constraints.
No factor may return zero for a quantity it cannot derive; it fails, and selection
reports *analysis unavailable*.

Numerical precision is also a hard constraint. Exact selection admits the reference path. A
bounded policy additionally admits only alternatives with sufficient static proof or a matching
whole-witness qualification. Numerical error is never an objective term: the solver cannot trade a
policy violation for lower estimated time.

## Seed

The backend supplies one constructive complete witness. It is a starting result,
not a default implementation and not a rule that excludes other bodies. The seed is
audited against the joint family like any witness. An audited seed is the baseline
result: a searched witness replaces it only when its estimate is strictly lower.
The seed and its estimate are retained in the result for comparison.

Seed construction obeys the same numerical admissibility relation as solver export. A strict
request cannot seed an unqualified lowering or approximate primitive; unconstrained exploration
may prefer target lowerings. A disagreement is a defective seed policy, never a reason to weaken
precision.

The seed is not injected into the solver. Search starts from the model alone, so
the seed bounds the result from above but does not steer the search.

A seed that fails the audit while the family is not proved empty is a defective seed
policy and is reported as a reconstruction defect. No unchecked partial path ever
executes.

## Search

One immutable solver model is built once. The budget bounds solver work and wall
time.

1. **Exact search** with half the budget. It may prove optimality or infeasibility.
2. **Neighborhood improvement** with the remainder, when a validated seed exists.
   Moves change implementations, covers, and numbers together.

The better incumbent of the two phases is compared with the audited seed; the
selected witness is the cheaper under the estimate, the seed on a tie. The proved
lower bound is the larger of the two phases' bounds, capped by the selected
estimate. A solver optimum costlier
than the validated seed is a contradiction and is reported as a defect.

The search consumes the generic solver unchanged: guards, residual AND/OR
decomposition, completed-proof reuse, budgets, and feasible incumbents keep their
solver meaning. There is no Seismic-specific solver and no second cache.

### Strategy

The budget carries a strategy (`Budget.strategy`, runtime `Settings.strategy`, CLI and
harness `--strategy exact|greedy`). `Exact`, the default, is the search above.

`Greedy` is a **diagnostic alternative**, kept to measure what solver search buys. It
runs no solver. Starting from the audited seed it sweeps the decisions in a fixed
order — occurrence choices in pre-order, then each active sequence's cover (all
singletons, or one maximal offered fused interval with singletons around it), then
active sites in id order — and for each decision tries every other value with all
others fixed, keeping the cheapest complete witness. Every trial is a complete witness
priced by the same exported model (`Model::validate_assignment`); nothing else ranks
alternatives. Sweeps repeat until one improves nothing, at most 16 times; the work and
time limits of the budget do not apply.

Changing an occurrence's choice deactivates the old candidate's subtree and activates
the new one. `Backend::seed` cannot be constrained to a partial choice, so newly active
decisions take the seed policy's base values: the seed's own choice and value wherever
the seed has one, otherwise candidate 0, the largest admissible width, parts 1, and
singleton covers; if that completion is infeasible the all-smallest-values completion
is tried once, and if both are the move is skipped. The seed's piece-target and
limit-repair steps are not replayed.

A greedy result is always `Feasible` with lower bound 0. Coordinate moves cannot make a
change that needs several decisions to move together (a fused interval whose
`equal_sites` differ, a slower child that enables a cheaper parent), so greedy
witnesses are in general costlier than exact ones.

### Timings

`Selected` records the wall time of each phase — family construction, backend hooks
(`bind_structure`, `constraints`, `intervals`, `factors`), model export (tabulation),
seed (construction and audit), search, instantiate, realize — and search statistics:
model variables and factors, solver work and nodes of the exact and neighborhood
phases, or greedy sweeps and trials. `seismic select` prints them. These are
measurements of the compiler, not estimates of the kernel.

## Proof status

| Status | Meaning |
| --- | --- |
| Feasible | A complete, audited, instantiated execution. Search ended by budget, or the family has open obligations. Performance is an estimate. |
| Model-optimal | Additionally, the solver proved optimality over the exported family — the listed candidates, intervals, and site domains — under the stated estimate model, and the family has no obligations. |

Model-optimal is a statement about a model. It proves nothing about physical
optimality, about values outside the backend's site domains, or about candidates
nobody authored. A feasible witness is executable; execution never waits for a proof.

Performance proof status and numerical evidence are independent. Every selected result separately
retains an `Exact`, `Proven`, `Qualified`, or `Unknown` numerical assessment. `Unknown` is valid only
for explicitly unconstrained exploration. Model-optimal therefore means optimal within the
numerically admissible family, not permission to weaken the requested policy.

## Outcomes

| Outcome | Meaning | Not to be read as |
| --- | --- | --- |
| Invalid source or contract | Type, numerical, effect, ownership, or declaration error | — |
| Missing target coverage | No applicable portable body, matching target lowering, or backend-specific helper has complete coverage for a reached call on this target and workload | A reason to interpret or fall back |
| Unsupported structural mapping | The backend has no mapping for this structure, or no domain value satisfies a site's requirements | Infeasibility of the family |
| Incompatible composition | Interfaces disagree, or a hard capacity cannot be met on a path selection cannot avoid | A cost |
| No feasible configuration, proved | The exported family has no solution | A statement about one body or grouping |
| Construction incomplete | Candidate construction left obligations | Missing support or infeasibility |
| Selection incomplete | The budget ended with no checked configuration | Infeasibility |
| Analysis unavailable | A quantity or estimate has no supported derivation | Zero cost |
| Reconstruction defect | Seed, witness, or instantiation disagreed with the family | Candidate infeasibility; it is a compiler defect |
| Selected feasible | Complete checked execution, estimated performance | A tuned or qualified result |
| Selected with model proof | Optimal within the stated family and estimate | Physical optimality |

Open obligations do not block a feasible selection whose own path is fully
analyzed. They are listed with the result and they block the model proof.

Because a backend must supply a seed and a rejected seed is a defect, a selection
currently either returns a checked witness or fails with one of the other outcomes;
*construction incomplete* and *selection incomplete* are reserved classifications
that the present pipeline does not produce.

## Replay

A complete witness can be replayed: the family and model are rebuilt, the witness
is audited exactly as a seed is, then instantiated and realized. No search runs. The
result is *feasible* with no lower bound, because nothing was compared. A witness
that no longer fits the program, workload, or backend is rejected; it is never
repaired. Replay is how a qualified witness is deployed without depending on
search.

## Reuse

Selection identity is `(entry, shapes, elements, precision policy, qualification catalog)` on one device. Buffer contents,
scalar arguments, and bounded index values such as the decode position are not
part of it, so decode steps do not retune. Adding or removing a portable body, target
lowering, or backend-specific helper changes the family and invalidates earlier witnesses.
Qualification reuse additionally requires equal program identity, complete witness,
specialization, target numerical environment, input domain/corpus, and evidence method. A replayed
qualification is audited against the current family; it is never repaired or transferred.

## Search analysis

`analyze-search` constructs the family for an entry and workload without selecting,
emitting, or compiling, and reports structure: templates, occurrences, occurrences
with a choice, the largest alternative count, sites, sequences, a `log10` upper bound
on raw assignments, independent components of the occurrence interaction graph,
and open obligations. The raw product is a description of the family, not a
prediction of solve work. The command predicts no times.

## Acceptance

- Independent choices export factors whose scopes never span two occurrences.
- A coupled parent and child select the jointly cheaper assignment even when the
  child alone is slower.
- An unsatisfiable hard constraint yields *infeasible*; an invalid seed yields a
  reconstruction defect.
- The selected witness reproduces the solver's assignment and cost exactly.
- Performance optimization ranges only over witnesses admitted by the precision policy.
- Qualification of one complete witness cannot authorize another witness or device environment.
