# Seismic compiler

**The compiler turns a checked library composition into a selected execution for a
workload and device.** It owns semantics-preserving transformations and joint
implementation selection; the runtime binds and executes its result.

## Compilation flow

| Stage | Responsibility |
| --- | --- |
| Load and check | Resolve libraries, types, symbolic shapes, effects, and lowering coverage |
| Canonicalize | Apply deterministic, idempotent normalization and recognize valid operation compositions |
| Compose | Inline and transform enclosing computations while preserving precision and effects |
| Lower | Expose applicable backend implementations and unresolved typed choices |
| Derive and optimize | Propagate legality/resource constraints, resolve choices and schedules, validate the objective |
| Emit | Translate the resolved execution through its implementation definitions |
| Native compile | Compile selected target code under its backend/toolchain contract |

Canonicalization covers supported rewrites, not arbitrary program equivalence.
Fusion and decomposition are choices about whole executions: eliminated transfers
can trade against longer lifetimes, lower concurrency, or added synchronization.

The compilation boundary is the enclosing checked computation. Ordinary calls do
not fix GPU launches or independent compilation units. Internal, nonescaping
intermediates are distinguished from externally observable bindings before
eliminating publications. Inlining alone is insufficient: iteration relationships,
access maps, value versions, and effects remain available for joint construction.
Construct physical decomposition from logical iteration, access maps, reductions,
and value dependencies; do not require a source streaming loop to make an efficient
execution reachable. Whole-domain loads and logical intermediates must admit bounded
streaming, shared producers, and state retention within the supported execution form.
Merely joining already-authored streams does not fulfill this requirement.

Geometry dependencies are distinct from element accesses. A direct parameter
extent query on an unchanged axis can survive contraction partitioning; it does
not require an element at the reduction coordinate. Queries observing a changed
axis or carrying unproved view/index dependencies cannot authorize that rewrite.

Retain operation meaning until applicable instruction/implementation covers can
be selected; premature scalar expansion must not remove supported alternatives.

Independent output grouping operates on retained construct calls before
contraction partitioning and backend body selection. Every admitted static output
axis exposes widths 1..N with exact full and remainder rectangles. Grouped calls
retain each output's seed, scalar calculations and conversions while sharing
invariant operands; subsequent contraction choices remain dependent alternatives.
For an ordered sequence of calls, each call's writes become visible to the next
local computation before its operands are grouped. Operand sharing requires the
same reaching snapshot and axis mapping; checked call effects participate in
both dependency analysis and snapshot invalidation.
The current constructor admits straight-line regions with retained calls,
provable operand maps and a common rectangular publication. Dynamic operand shapes,
unknown effects and unsupported maps remain outside this constructor's coverage.
Its applicability cannot be inferred from a model or construct name.

Later visits to retained calls reconstruct the enclosing logical domain facts,
including proven capacities and captured view identities. Those facts follow
lexical scope across branches and loops; moving partition selection later must
not lose dynamic contraction alternatives.

Grouping retains source alias requirements before changing work-item geometry.
Combining all outputs into one item must not relax a source independence condition.
Identical in-place maps may admit exact overlap where proven; shifted overlapping
maps retain their disjointness requirement through accounting and native admission.

## Optimization guarantees

| Guarantee | Required property |
| --- | --- |
| Intrinsic expressiveness | Structured Seismic lowerings can express the required legal implementations of admitted backend mechanisms. |
| Execution-family preservation | Normalization, composition, and lowering retain every implementation promised by the declared source/execution form, subject to established legality constraints. |
| Selection completeness | Completed optimization minimizes the stated model objective over that family. |
| Native correspondence | Emission and native compilation implement the selection within the qualified numerical and resource mapping. |

The [language](language.md#lowering-authoring-contract) owns the authoring domain;
[execution](execution.md) defines its legal realizations; [tuning](tuning.md) owns
selection; [backends](backends.md) own native correspondence. These guarantees
compose but are not interchangeable. A family containing a valid competitive
implementation cannot produce a worse model objective after complete selection.
Physical optimality additionally requires adequate coverage and qualified hardware
and native mappings. Type correctness alone establishes none of those performance
claims.

## Source stability

Under identical bindings, numerical permissions, effects, and compilation conditions,
the documented normalization domain preserves the reachable execution family, up
to identity renaming, across these variations:

| Source variation | Condition |
| --- | --- |
| Rename values, parameters, or helpers | Bindings and semantics unchanged |
| Extract or inline a helper | Body visible; no new opaque/ABI boundary or intrinsic-scope change |
| Introduce or remove pure scalar temporaries | Evaluation and precision unchanged |
| Compose views or normalize equivalent indices | Equality established by supported index algebra, including bounds and overflow |
| Reorder independent pure statements | No observable ordering, alias dependency, or numerical change |
| Refactor a local producer through a logical tile | Same logical values and conversions; no new observable publication |

Analyses consume normalized dataflow, iteration, access and effect relationships;
incidental AST shapes cannot restrict these guarantees. Canonicalization is
deterministic and idempotent within the supported domain, not a claim to decide
arbitrary program equivalence or print one universal form for every semantics.
Document and check extensions to that domain.

Eliminating redundant execution-oriented language syntax must preserve the efficient
realizations available through it for computations with equivalent semantics. Build
the derivation from logical composition and migrate its uses together; deleting the
syntax while forcing full intermediates or losing efficient intrinsic covers is not
a completed simplification. This does not promise equivalence for programs whose
results depend on a compiler-chosen partition.

New casts, reassociation, FMA changes, reduction ordering, opaque calls, and external
writes can change the legal family. Explain the actual semantic restriction or
missing analysis rather than recommending a hidden matcher pattern. Bounded
refactoring-pair checks compare execution-family coverage and model optima as well
as numerical behavior; identical native code is not required.

## Representations

| Representation | Contents |
| --- | --- |
| Portable IR | Checked computation, values, symbolic shapes, representations, control, numerical permissions, and effects |
| Lowered IR | Shared computation extended with backend implementations, dependent choices, and legality constraints |
| Tuned IR | Actual selected operations, allocations, layouts, mappings, checks, synchronization, and launches |
| Target code | Backend code-generation input implementing the selection |
| Executable | Native code, binding interface, retained conditions, and relevant identities |

Stages share semantic definitions and explicit transformation relationships.
Resource and dependency models are derived views. Optimization refines this execution;
accounting derives its consequences; emission consumes it. There is no separate
computation graph for bounds, authored companion cost model, or proof-artifact pipeline.

Tuned IR resolves compilation decisions into actual execution structure. Runtime
values and declared dynamic extents may remain; emitter-selected tuning may not.

## Ownership

| Owner | Authority |
| --- | --- |
| [Language](language.md) | Source semantics, primitive definitions, libraries, and checking |
| [Execution](execution.md) | Legal forms, operation contracts, choices, dependencies, allocations |
| [Accounting](accounting.md) | Derived resource constraints, exact quantities, timing relationships, sound relaxations |
| [Tuning](tuning.md) | Search, propagation, scheduling, and exact selection within the declared form |
| [Backends](backends.md) | Concrete implementations, emission, native mappings, hardware inputs |
| [Runtime](runtime.md) | Device/artifact ownership, binding, submission, completion, and caching |
| Tooling | Read-only explanations of these structures |

Shared contracts do not depend on the engine or runtime. Concrete backends are
composed through compilation interfaces; the runtime composition root does not
acquire optimization policy.

## Build-time and device-time work

| Boundary | Work |
| --- | --- |
| Application build | Check standard/model libraries, embed parsed/canonical programs, generate typed Rust bindings |
| Device compilation | Specialize shapes and conditions, select implementations, derive storage and ABI, emit native input |
| Warm invocation | Validate dynamic bindings and reuse prepared execution |

Signatures define host shapes, representations, parameters, and effects. Generated
bindings eliminate independently maintained ABI layouts. Missing declarations,
incompatible calls, and coverage errors fail library checking. Accelerator presence
is not required for device-independent checking.

Development source overrides pass the same checks as embedded programs. Relevant
program, workload, hardware, and implementation identities govern reuse.

## Optimization principles

- Preserve operation meaning, value versions, numerical permissions, and effects.
- Expose performance-relevant choices in the legal execution space.
- Derive resources from the same implementation definitions used by emission.
- Select from IR and hardware contracts; candidate benchmarks, native resource
  queries, heuristic scores, and compile-and-try fallback are not selection inputs.
- Distinguish model optimality, sound physical bounds, hardware fidelity, and
  invocation applicability. A private constructor cannot establish physical truth.
- Keep construction and validation on the same path for explicit diagnostic choices.
- Preserve complete search status; unfinished work cannot establish an optimum.

## Authoring and inspection

| Capability | Required information |
| --- | --- |
| Check / interpret | Source locations, typed semantics, violated conditions, reference outputs |
| Inspect lowering | Applicability domain, source commitments, free choice domains, shapes, ownership, dependencies, and unsupported analysis |
| Inspect performance | Resource terms, limiting constraints, choice explanations, scope and assumptions |
| Inspect emission | Target code, native diagnostics, applicable mapping information |
| Reproduce | Bounded source/library identities, inputs or references, options, device/model conditions |

The CLI operates independently of the engine. Editing a kernel or lowering affects
only dependent artifacts. When authors must contort a natural program to work around
compiler behavior, repair the responsible language, analysis, lowering, or tooling.
