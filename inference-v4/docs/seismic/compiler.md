# Seismic compiler

**The compiler turns a checked closed program into one selected, verified execution
for an entry, a target, and a workload.** It selects among what authors supplied. It
does not synthesize execution structure.

## Stages

| # | Stage | Input → output | Guarantees |
| --- | --- | --- | --- |
| 1 | Check | Sources → logical checked IR | Names, function families, portable and target capabilities, ownership and borrows, shapes and bounded indices/ranges, loop ordering and independence, initialization, and numerical obligations. Every declared body is checked, selected or not. |
| 2 | Construct the family | SIR + entry + target + workload → family | Applicable candidates at each static call occurrence, guarded by their parent candidate; portable-reference identity and numerical effects; interned templates; numerical sites; normalized execution-unit sequences; obligations for anything construction could not analyze. |
| 3 | Bind the backend | Family → site domains, hard constraints, legal intervals, local cost factors, seed | Deterministic. No ranking, no profitability filter. |
| 4 | Select | Solver model + precision evidence → witness | One complete joint assignment satisfying hard numerical constraints, audited against the family and the exported model. |
| 5 | Instantiate and verify | Witness → concrete execution IR | Exactly the selected bodies, covers, and numbers. The result passes execution-IR verification. |
| 6 | Realize and emit | Execution IR → realized execution → target source | Deterministic mapping rules; hard limits rechecked on the realized execution. Emission makes no decision. |
| 7 | Compile and bind | Target source → native kernel | See [Runtime](runtime.md). |

Stages 2–5 run per `(entry, target, workload)`. A workload binds every shape
parameter and element parameter of the entry. Buffer contents, scalar values, and
bounded index parameters are not part of it.

## Authoritative representations

**Logical checked IR** is the checked, typed program. Owned values, borrows, ordered and
independent loops, state, results, and calls retain their source meaning. A definition is a
template; a call names a contract, not an implementation. Nothing in logical IR chooses.

**The family** is a finite description of the supplied program family for one entry,
target, and workload:

| Element | Meaning |
| --- | --- |
| Template | One definition specialized to concrete semantic shapes and elements, plus which shape parameters are bound to caller slices (structural) or to runtime-valued caller extents (dynamic). Shared by equal specializations. |
| Occurrence | One static call in one parent candidate, or the entry. Holds every applicable candidate and every rejected definition with its reason. |
| Candidate | One applicable portable body, backend-specific body, or target lowering at an occurrence, with reference identity, numerical effects, structural requirements, child occurrences, sites, and sequences. |
| Site | One finite physical mapping decision owned by a candidate and derived after logical checking. |
| Sequence | The execution units of one block with at least two units, in authored order. |
| Obligation | A supported-looking candidate that construction could not analyze. Reported; never silently dropped. |
| Witness | One choice per active occurrence, one value per active site, one exact contiguous cover per active sequence. Inactive elements are absent. |

Dynamic repetition (visits, elements, tokens, heads) never creates occurrences,
sites, or sequences. Two calls to one function are two occurrences with independent
choices even when they share a template. Template sharing never merges producers
or undercounts repeated cost.

The **execution IR** is the instantiated physical form: launches, mapped loops, local
allocations, transfers, and synchronization. It contains no alternatives. Physical block or tile
objects in this IR are compiler-owned and are not source types.

## Applicability

- A call resolves by name and argument binding to a contract family. Its candidates are the
  applicable portable bodies together with the selected target's lowerings, or the applicable
  bodies of a backend-specific helper for that target.
- A `where` predicate over static semantic shapes is decided at construction.
  A predicate over a structural extent becomes a numerical requirement on the
  bound site (multiple-of, at-least, at-most, equal, divides). A predicate over a
  dynamic shape parameter is undecidable at selection, so that candidate is
  inapplicable.
- Zero candidates at an occurrence makes its parent candidate unselectable; at the
  entry it is missing coverage. One candidate is no categorical decision. Several
  are a solver decision. No declaration order, specificity, or priority applies.
- Expansion is finite: a candidate cannot recursively re-enter an unresolved family without a
  decreasing static measure.

## What the compiler may do

When the ordinary preconditions hold, and identically for every candidate:

- Name resolution, inlining of the selected body at every call during
  instantiation, scalar simplification, constant propagation, dead pure scalar
  code elimination.
- Checked borrow, slice, range, and index normalization.
- Folding a single-consumer pure logical value into an adjacent consumer's unit when ownership and
  observable copy behavior are unchanged.
- Mechanical expansion of the selected mapping and native primitive sequences.
- Instruction selection by the mapping's fixed rules within the numerical contract.
- Deterministic scoped allocation.

These happen before family construction or are part of every candidate's
realization and estimate alike. None runs after selection to change a grouping or
remove a transfer the solver evaluated.

## What the compiler may not do

- Fuse noncontiguous units, permute units, or fuse without a prescribed realization.
- Change ordered `for` semantics, or introduce dependence between `parallel for` visits.
- Move a producer across scopes, rematerialize it at consumers, or merge separate
  producer occurrences.
- Infer streaming from a materialized intermediate.
- Interchange loops where reuse order changes, decompose a reduction, or introduce
  a split reduction.
- Search representation conversions, memory placement, allocation reuse that adds a
  barrier, or a schedule.
- Insert implicit dtype conversions to make an overload match.
- Treat a numerical effect as equivalent to the reference without exact, proven, or matching
  whole-witness-qualified evidence.

## Source stability

Preserve the family: renaming, alpha-renaming of shape parameters, naming a
single-use expression with `let` next to its consumer, and helper extraction that
keeps the same definitions and contracts and does not separate units that could
share an interval.

Units are formed per authored block. A statement that calls a helper is one unit of
its block, and the helper's body has its own sequences; an interval never spans a
call boundary. Extracting part of a fusible run into a helper therefore removes the
intervals that crossed the new call. This is a current limitation: the governing
specification requires unit granularity to be invariant under helper extraction.

Change the family: adding or removing a portable body, target lowering, or backend-specific
helper; moving a
producer; reordering dependent statements; changing ownership or borrow boundaries; creating
another producer occurrence; or giving a `let` a second consumer.

Statement order of independent work is author-significant for fusion. The closed
linked program is the compilation identity: a library cannot add a lowering after
compilation.

## Audit

A witness becomes executable only after all of the following hold. Failure of any
is a reconstruction defect, reported as such.

1. The witness is structurally valid for the family: one applicable choice per
   active occurrence, values for exactly the active sites, requirements satisfied,
   exact contiguous covers for exactly the active sequences.
2. It maps to a complete assignment of the exported model, satisfies every
   constraint, has an exact estimate, and survives the round trip back to the same
   witness.
3. A solver-produced witness reproduces the solver's assignment and cost exactly.
4. The backend seed passes 1–2 before search may use it.
5. Instantiation consumes the witness without choosing, and the execution IR
   verifies.
6. Realization applies its rules and rechecks hard limits on the realized execution.

## Diagnostics and inspection

Errors name the definition, call path, region or site, the violated requirement,
and the candidate or mapping involved. They never recommend a fallback.

`select` reports, for one entry and workload: every occurrence with its candidates,
the definition origin and dependency path of each, requirements, and rejections with reasons; every site
with its kind, extent, owner, domain, selected and seed value; every sequence with
its units, completion boundaries, selected and seed cover; both estimates, the
proved lower bound, the estimate model identity, the proof status, and unresolved
obligations. An unselected candidate is never reported as inferior.

## Ownership

| Concern | Owner |
| --- | --- |
| Source meaning, checking, SIR, interpreter, family, instantiation | `seismic-lang` |
| Solver export, search, audit, replay, search analysis, `Backend` contract | `seismic-compiler` |
| Search algorithms and proof semantics | `magnitude-solver` |
| Mapping, realization, emission, native limits, estimate model | the backend crate |

The compiler contains no target names beyond passing the requested target to
family construction, and no kernel or model names at all.
