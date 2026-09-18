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
Copying a computation gives its defined values fresh identities, while writes
through externally bound tensor references retain those references. Backing-memory
effects remain part of dependency analysis; they are not new value definitions.
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
Complete pure tile producers can also retain one shared prepared value. Equality
uses checked coordinates and scalar precision, including helper extraction and
pure temporary bindings. Sharing preserves the stored conversion, requires stable
inputs and immutable retained storage, and cannot remove an escaping scalar write
or a possible failure. Its longer lifetime remains an implementation choice.
Independent serial regions over the same constant interval expose a fusion
choice after backend-body resolution. The constructor preserves each region's
iteration order, checks typed intrinsic writes and scalar dependencies, and moves
only independent prerequisites across the intervening statements. Tensor effects
and unknown accesses prevent this motion. Fusing row-block loops can therefore
expose shared activation preparation across separate matrix calls without changing
contraction order or introducing construct-specific recognition.
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

Shared IR verification runs after decomposition, after each completed body/value/
representation stage, and before executable realization. It checks lexical bindings,
parameter and reference types, index ranks, iteration bindings, load/store shapes,
snapshot borrowing lifetimes, retained choice membership, and tensor-effect operands.
Executable verification additionally rejects unresolved calls, loads, and reduction
trees. Backend participation and terminal verification refine these checks for the
selected execution. A failed invariant is a compiler error at that boundary; it is
never evidence that an otherwise legal optimization choice is infeasible.

Exact decoded storage admits element-wise and packet-oriented producers. A packet
producer retains each coefficient group's code words and coefficients while decoding
its elements, without factoring a contraction or changing decode rounding. Each
group admits every consecutive owner width from one code to the whole group,
including nondivisible subpackets. Splitting repeats coefficients and boundary
words while exposing more independent owners; these consequences remain visible
to joint selection instead of fixing a preferred width. Each owner also retains
specialized and indexed decoder covers. Specialized decoding selects fixed code
ranges and reuses their words. Indexed decoding visits a bounded element loop,
computes word/bit coordinates, and reads a following word only when a code crosses
that boundary. Both use the representation's exact code interpretation and FMA;
indexed decoding avoids a separate source branch for every possible subpacket
origin. Its
ownership, scalar temporaries, writes and publication use the existing execution
operations. Admission derives complete-group geometry and alignment from captured
view provenance; a compact snapshot does not erase an unaligned prefix. Partial
final groups use ordinary element reads. Runtime packet counts and tails derive
their capacities from checked bounds on the captured logical extent, including
quotients and remainders, rather than evaluating the expression at capacity.
Other views retain element-wise decoding and encoded storage alternatives.

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

Independent retained folds with identical axis, extent, segment size and selected
merge tree admit a product execution. Their state fields and callback bodies are
concatenated without algebraic rewriting. Read-only named input snapshots may
share one prepared leaf; destructive helper inputs keep separate copies. The
product has an explicit resolved callback origin, not an invented source helper
or primitive reduction contract. Dependency, external-effect and numerical
failure checks constrain interleaving, including movement of intervening setup.

Encoded fold inputs also admit packet preparation inside each selected segment.
The captured input remains the physical addressing authority; decoded coordinates
are local to the segment. Complete coefficient groups and aligned subdivisions
reuse words and coefficients without decoding unused neighboring codes. A
participant execution therefore retains only that participant's prepared segment,
while whole-value cache preparation remains a separate representation choice.
Word-aligned segment origins adjust the physical word address directly, so the
decoder does not clone identical extraction arithmetic for each group position.
Subsegments that straddle word boundaries retain their exact guarded patterns.

The decode preparation window is independent of the numerical reduction segment.
A segment can prepare and consume several smaller windows while retaining its
partial state and visiting the same FMA sequence. The common window across its
decoded inputs respects each representation's packet alignment; the domain includes
all supported aligned subdivisions and multiples, including a shorter final
window. Large multiple domains remain arithmetic progressions rather than lists.
Full windows and the final window lower to ordinary bounded caches and loops, so
existing placement, lifetimes, accounting and emission see their actual work.
Direct inputs and encoded segment snapshots keep their original captured origins.

An encoded segment snapshot is another input preparation choice. It captures the
selected segment through an ordinary packed load, keeping its words, coefficients
and physical prefix geometry. The existing load and storage choices may borrow or
materialize it; participant-owned materialization stays private. Consumers decode
on use with the representation's original arithmetic, so this alternative does
not allocate a full F32 decoded segment or factor coefficients out of a fold.

Fold traversal admits every expansion width from one to the selected preparation
window (or segment when no smaller preparation is selected). Expansion substitutes
both symbolic and scalar indices in the ordinary step body, preserving ordered
visits and the selected partial/merge arithmetic.
Nondividing widths guard the final partial traversal. This is an explicit IR
choice consumed by accounting and emission, not a backend unroll hint.

Within a segmented fold, the step's private state parameters retain the partial
accumulator across visits. Initialization writes the captured identity into those
parameters; the step output remains separate and is copied back after the entire
callback finishes. This removes a redundant state copy without imposing in-place
update semantics on callbacks that read other elements or mutate their arguments.
Participant ownership distributes this same retained state through private slots.

Packet coefficient lifetime is selectable independently of the decoded window.
Packet scope reads the representation's scale and optional bias at each decoder
owner. Segment scope prepares exact F32 coefficient tiles once for all groups
intersecting the aligned segment and indexes them from each decoded window.
The original representation accessors decode hierarchical coefficients; retaining
their values does not factor coefficients out of contraction arithmetic. Tail
windows share the same captured coefficient tiles. Allocation, ownership and
requested traffic remain visible as ordinary IR and backend storage choices.
Encoded word lifetime uses the same packet/segment distinction independently of
coefficient lifetime. A complete word-aligned segment can acquire its exact U32
word span once and serve smaller decode windows from that retained tile. The
producer owns one row at a time and exposes contiguous word copies to backend
transfer choices. It acquires no neighboring word outside the segment, changes
neither bit extraction nor FMA order, and retains the original encoded source
snapshot. Segments that split a word keep packet-scope acquisition.

Independent output grouping retains a choice for its full and remainder rectangles:
separate parallel regions, or one concatenated work domain. Concatenation gives each
rectangle a disjoint interval of work items and reconstructs its original coordinates
before executing its unchanged, statically shaped body. Rectangles share the original
source parallel independence and invocation alias requirements; no new cross-operation
independence is inferred. A positive combined extent must fit the I32 coordinate
contract. This permits a single backend launch for grouped outputs with remainders
without reading or publishing padded logical outputs.

Pointwise fold callbacks also expose separate versus retained step-result storage.
Retention is admitted from the resolved ordinary callback: every output field is
written once over its complete owned domain, reads only its corresponding left
state coordinate, and has no cross-field, cross-coordinate, or external effects.
Scalar temporaries within that element remain valid. The selected realization
aliases private output to private left state and removes the resulting self-copy;
the source seed, identities, FMA/cast boundaries, and merge tree are unchanged.
Callbacks with coupled state retain distinct step output storage.

Each read-only fold operand separately admits private leaf storage or a view of
its retained input snapshot. The existing parameter mutation/alias analysis
admits views; a callback that writes the operand or a derived alias keeps its
private copy. View substitution preserves checked indexing and exact decoding,
and runs after snapshot selection. It introduces no pointer into changing source
state. This choice composes with packet windows, state retention, and ownership.

Matrix output coverage may include a complete physical fragment around a partial
logical rectangle. The standard Metal covers guard all seed reads and publications,
zero only unpublished rows/columns, and never pad the contraction axis. Complete
K tiles use the existing F32 matrix intrinsic; remaining leaves keep scalar FMA
order. The covers expose per-fragment staging and bounded-piece left/right row
reuse through ordinary backend bodies. Affine branch comparisons retain their
unit-coefficient atom bounds in the checker's existing lexical facts, so guarded
tails are legal without requiring padded source tensors or divisible output sizes.

Complete conditional producers can also share an intermediate after serial range fusion. Both branches must write the same owned point exactly once, and the branch conditions, values, scalar versions, and precision conversions must agree. Normalization retains the existing conditional IR, so a guarded input read remains lazy. Writes to an input, different fallback values (including signed zero), incomplete branches, and escaping scalar writes prevent sharing. The shared intermediate uses the same allocation and memory accounting as a straight-line producer.

Regroupable reductions expose `SeedThenPairwise` as a compact tree choice: the initial state is the left root child and the right child is the pairwise tree of contiguous input leaves (or fold segments). This is an existing legal explicit tree, not a new numerical permission. It leaves input order, step arithmetic, and the seed's single merge intact. Participant realizations can retain these input leaves or complete subgroup-sized waves before the root merge. Their private storage excludes the seed leaf, while the seed and merge callback remain ordinary typed operations visible to accounting and emission. Ordered and empty reductions do not acquire this choice.

Fold input preparation includes decoded snapshots for dense and encoded operands, with either segment or window lifetime. A fused product fold can therefore acquire a common activation window once and let all its step consumers read the same private value. Snapshot types preserve the input's decoded scalar precision. Ordinary guarded producers acquire only live input elements; the final partial segment never reads beyond the captured source or contributes padded leaves. Window widths cover every size up to the segment when no packed decoder imposes alignment. Encoded plane snapshots and exact packet decoders remain independent alternatives. Snapshot storage and reads are derived from the same expanded IR as execution.

A rectangle cover can retain four 8×8 matrix accumulators across K while each
bounded 8×8 operand preparation serves two products. Its ordinary serial loops
and conditional staging producers remain visible to fusion and sharing across
calls. Output padding is unpublished; complete K fragments retain matrix
semantics and the K remainder keeps scalar FMA order. This is an additional
backend body, with no preferred-size policy or claim of automatic selection.

Independent grouped calls also expose an unrolled or compact serial epilogue.
The compact form is admitted when every live value from the original per-item
prefix is a distinct call output, with no intervening use of an earlier output.
It consumes disjoint slices of the retained grouped results in source slot order,
using one ordinary range body for local intermediates, casts and publications.
Prefix evaluation and call seeds remain intact. Live prefix scalars, dependent
calls and repeated output state retain the unrolled form. The new range remains
available to normal producer, storage and terminal traversal choices.

Matrix operand panels buffer pure per-iteration preparation across any selected
contiguous number of iterations in a static serial fragment loop. Admission uses
the existing complete point-producer analysis and requires the remaining body to
contain only fragment declarations, read-only matrix loads and matrix updates.
Prepared operands cannot read other prepared operands or state written by those
updates. Source tile mutations and external publications exclude this cover.
The selected panel is an ordinary leading tile axis, prepared cooperatively and
sliced by the original ordered fragment-update loop. Final partial panels guard
both preparation and consumption; their padded ordinal is tested in U32 before
evaluating a live I32 source index. Matrix accumulator lifetime, operand precision
and K-tail arithmetic remain unchanged. All widths from one through the loop's
iteration count are retained as a compact typed domain; physical capacities are
resolved by the existing storage/accounting path.
