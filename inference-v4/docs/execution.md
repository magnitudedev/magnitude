# Seismic execution representation

An execution describes how a computation obtains values, performs operations,
uses storage, coordinates participants, and publishes observable results. The
same structure supports legal-choice construction, resource derivation, emission,
and proof references under the [compiler principles](compiler.md).

## Shared semantic structure

| Entity | Meaning |
| --- | --- |
| Operation | A typed computation or effect with numerical and participation semantics. |
| Value version | A particular definition/publication, including its representation and precision. |
| Allocation | Canonical storage identity with size, alignment, address space, and lifetime. |
| View | A mapping from logical coordinates to an allocation/version, with offsets, strides, and extents. |
| Control region | A loop, branch, ownership region, or parallel domain and its execution conditions. |
| Dependency | A required value, effect, synchronization, or completion relation. |
| Obligation | A proposition that must be established statically or enforced at an appropriate runtime boundary. |
| Decision | An unresolved choice among implementations or execution structures with a derived legal domain. |

Identities are scoped by semantic program identity. Source spans support diagnostics
but do not distinguish cloned operations or establish equivalence. Transformations
retain origin relationships while assigning new identities where meaning changes.

Allocation identity is not a parameter name. Different views can alias, and the
same bytes can hold different value versions over time. Copies, borrowed views,
and recomputed values have distinct semantics even when their current contents match.

## Operation and implementation contracts

An admitted operation has an exhaustive definition of:

- Operand/result types, shapes, and representations.
- Numerical meaning, rounding/publication points, overflow, exceptional values,
  and permitted reassociation or approximation.
- Reads, writes, alias relationships, and value dependencies.
- Required participants, convergence, ordering, and synchronization.
- Shape, alignment, layout, and backend capability constraints.

Each backend implementation additionally supplies its execution structure,
resource semantics, and emission mapping. Implementations can expand into shared
operations or terminate at backend primitives. Primitive contracts form a versioned
trusted vocabulary; arbitrary callbacks cannot assert missing semantics or proofs.

The admitted implementation definition is the common authority for accounting and
emission. It is not sufficient to maintain an opcode emitter and an unrelated cost
table with matching names. Hardware-dependent parameters are supplied by a bound
hardware contract; the operation definition specifies which parameters it requires
and how they participate in the model.

Structural completeness is an admission invariant. An operation cannot enter the
qualified compilation path with an optional resource implementation or an opaque
unmodeled effect. Physical fidelity of the completed contract remains a separate
qualification requirement described in [Backends](backends.md).

## Stage invariants

Portable IR contains construct calls and common typed operations. Applying
Seismic `lower` definitions replaces calls with backend implementations while
retaining alternative bodies and dependent choices where optimization is required.

Lowered IR contains the computation and its constrained execution family. It must
not prematurely choose an implementation merely because it is the first lowering,
a convenient tile size, or an emitter's preferred path.

Tuned IR resolves all compilation decisions into the actual execution. Its
allocations, addresses, checks, communication, synchronization, and launch graph
must agree with its checked model and selection witness. It contains no unresolved
performance fallback. Runtime control flow remains explicit where allowed by the
workload domain.

## Legal execution forms

An execution form defines the implementations and transformations admitted for a
computation. It is defined independently of search order, search budget, and the
set of candidates an optimizer happened to visit.

| Decision family | Representative domains |
| --- | --- |
| Implementation | Lowering alternatives, primitive/instruction covers, scalar/vector/matrix implementations. |
| Decomposition | Tile dimensions, streamed pieces, work per invocation, reduction splitting and merge structure. |
| Mapping | Vector widths, lanes, subgroups, workgroups, CPU workers, iteration assignment. |
| Storage | Address spaces, layouts, materialization, recomputation, allocation granularity, lifetime reuse. |
| Composition | Fusion, intermediate elimination, publication placement, launch boundaries. |
| Coordination | Communication, barriers, dependencies, ordering, scheduling. |

Domains may be symbolic integer ranges or dependent alternatives. Semantic,
representation, ownership, and hardware constraints determine legality. Workload
extents and backend capabilities bound choices; handwritten preferred subsets do
not establish coverage.

For example, placing a tile in subgroup-shared storage affects capacity,
communication, publication barriers, and legal reduction implementations. Those
consequences are derived from one decision, rather than selected independently and
reconciled after emission.

Form restrictions must be explicit and meaningful. A proof for graph-preserving
direct computation does not automatically cover algebraic replacement algorithms.
Expanding the admitted form requires rechecking dependent proofs and cached results.
A diagnostic restriction cannot silently replace required compiler coverage.

## Safety and effect obligations

Bounds, shape compatibility, aliasing, initialization, representation validity,
parallel independence, and collective convergence are IR obligations.

| Discharge | Required behavior |
| --- | --- |
| Static proof | Preserve the supporting facts and checked derivation; emit no redundant check for that obligation. |
| Invocation check | Check properties of bindings or execution conditions before relying on them. |
| Dynamic check | Represent the necessary check at the execution point, its dependencies, failure behavior, and resource cost. |

Transformations may hoist or combine checks only when doing so preserves observable
behavior and validity. A masked lane does not justify an out-of-bounds access before
the mask is applied. A lane-local predicate does not establish collective convergence.
An allocation bound does not by itself prove a logical view access valid.

Remaining checks are part of the selected execution. Emission must not rediscover
bounds or introduce a second checking policy. Failure behavior distinguishes a
statically invalid program, a rejected invocation, and a dynamic execution failure;
runtime completion and partial-effect handling follow [Runtime](runtime.md).

## Transformation contracts

Every transformation establishes semantic equivalence under retained conditions,
reconstructs dependencies and value/storage relationships, and preserves or
rederives obligations and decision domains.

Fusion preserves required publications and dependencies. Recomputation preserves
numerical/effect semantics. Storage reuse requires non-overlapping live intervals
and appropriate completion ordering. A reduction rewrite requires a legal merge
operation and the relevant reassociation permissions. Floating-point identities
cannot be justified solely by real-number algebra.

Examples of required counterexamples include overlapping aliased views, snapshots
separated by writes, signed zero and NaNs, partial tiles, zero-length domains,
nonuniform collective participation, and mutation of borrowed storage.

## Conformance

For every supported construct and composition, checking, transformation, accounting,
and emission agree on one typed execution description. All introduced operations
have contracts; all retained runtime obligations have execution semantics; all
selected decisions are reflected in Tuned IR. Numerical and effect preservation
must be checked independently of optimizer cost improvements.
