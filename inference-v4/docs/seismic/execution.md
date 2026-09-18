# Seismic execution representation

An execution describes how a computation obtains values, performs operations, uses
storage, coordinates participants, and publishes observable results. The same structure
supports legal-choice construction, resource derivation, emission, and performance
constraints under the [compiler principles](compiler.md).

## Shared semantic structure

| Entity | Meaning |
| --- | --- |
| Operation | A typed computation or effect with numerical and participation semantics. |
| Value version | A particular definition/publication, including its representation and precision. |
| Allocation | Canonical storage identity with size, alignment, address space, and lifetime. |
| View | A mapping into a logical value version, with offsets, strides, extents, and backing storage when its elements are needed. |
| Control region | A loop, branch, ownership region, or parallel domain and its execution conditions. |
| Completion event | Completion and visibility of an operation, distinct from its issue where asynchronous. |
| Dependency | A required value, effect, synchronization, or completion relation. |
| Obligation | A proposition that must be established statically or enforced at an appropriate runtime boundary. |
| Decision | An unresolved choice among implementations or execution structures with a derived legal domain. |

Identities are scoped by semantic program identity. Source spans support diagnostics but
do not distinguish cloned operations or establish equivalence. Transformations retain
origin relationships while assigning new identities where meaning changes.

Allocation identity is not a parameter name. Different views can alias, and the same
bytes can hold different value versions over time. Copies, borrowed views, and
recomputed values have distinct semantics even when their current contents match.

Geometry can remain live after a value's element data becomes dead. Its definition
still captures endpoints, clamps dynamic windows, checks points, and validates
reshape layout at the original evaluation site. An owning tile snapshot has its
own contiguous layout even when no element allocation is necessary. Endpoint and
axis expressions can themselves read data and retain those dependencies.

Geometry-only realization is explicit and cannot service an element access. The
compiler may omit element production only after proving that no data consumer,
escaping state update, publication, numerical failure, or layout failure is lost.
Captured runtime coordinates are ordinary typed scalar values in this same IR;
shape-equivalence identities alone do not identify a runtime snapshot. Projection
retains conversions and guards, and allocates only its demanded result. A dynamic
window that may cover the entire input still has that full capacity unless a
separate bounded consumer or streaming realization establishes a smaller demand.

Ordered reductions over computed tile windows can use the same bounded iteration
as tensor-backed domains. Capacity follows structural view provenance; the loop
reads the already captured value's geometry rather than reevaluating its original
endpoints. A later temporary with the same shape does not replace that snapshot.
Inputs that overlap mutable reduction state are captured once before decomposition,
so each piece consumes the original input while carrying the preceding state.
An empty logical domain executes no pieces and leaves the initial state intact.

## Operation and implementation contracts

An admitted operation has an exhaustive definition of:

- Operand/result types, shapes, and representations.
- Numerical meaning, rounding/publication points, overflow, exceptional values,
  and permitted reassociation or approximation.
- Reads, writes, alias relationships, and value dependencies.
- Required participants, convergence, ordering, and synchronization.
- Shape, alignment, layout, and backend capability constraints.

Each backend implementation additionally supplies its execution structure, resource
semantics, and emission mapping. Implementations can expand into shared operations or
terminate at backend primitives. Primitive contracts define the supported semantics;
arbitrary callbacks cannot assert missing behavior, legality, or costs. The same
implementation structure accounts for helpers, temporary operations, and synchronization
before emission.

The admitted implementation definition is the common authority for accounting and
emission. It is not sufficient to maintain an opcode emitter and an unrelated cost table
with matching names. Hardware-dependent parameters are supplied by a bound hardware
contract; the operation definition specifies which parameters it requires and how they
participate in the model.

Structural completeness is an admission invariant. An operation cannot enter the
qualified compilation path with an optional resource implementation or an opaque
unmodeled effect. Physical fidelity of the completed contract remains a separate
qualification requirement described in [Backends](backends.md).

## Stage invariants

Portable IR describes logical domains and computation independently of physical
partitioning. Execution choices cannot determine source-visible shapes, iteration
counts, or index meaning. Algorithmic windows retain their declared boundaries;
physical pieces refine their execution without redefining them. Numerical variation
is limited to the computation's explicit permissions.

Portable IR contains construct calls and common typed operations. Applying Seismic
`lower` definitions replaces calls with backend implementations while retaining
alternative bodies and dependent choices where optimization is required.

Lowered IR contains the computation and its constrained execution family. It must not
prematurely choose an implementation merely because it is the first lowering, a
convenient tile size, or an emitter's preferred path.

Tuned IR resolves all compilation decisions into the actual execution. Its allocations,
addresses, checks, communication, synchronization, and launch graph must determine its
derived model and objective. It contains no unresolved performance fallback. Runtime
control flow remains explicit where allowed by the workload domain.

## Legal execution forms

An execution form defines the implementations and transformations admitted for a
computation. It is defined independently of search order, search budget, and the set of
candidates an optimizer happened to visit.

| Decision family | Admitted dimensions |
| --- | --- |
| Algorithm and implementation | Declared library/lowering alternatives, permitted factorizations, scalar/vector/packed/matrix instruction covers and their staging/conversions. |
| Decomposition | Multi-axis tiling, traversal, interchange, grouping, streamed pieces, unrolling, tails, reduction trees and splits, segmented work, supported scans. |
| Composition | Fusion/fission, producer placement and sharing, local intermediates, state retention, partial/merge launches, execution boundaries. |
| Ownership | Coordinate-to-participant mappings, vector widths, lanes/subgroups/workgroups/workers, producer/consumer roles, supported persistent work policies. |
| Layout and representation | Supported strided/blocked/permuted/packed/swizzled maps, padding, alignment, fragment layout, internal repacking with exact conversion semantics. |
| Residence | Borrow/materialize, distributed/replicated values, address spaces, rematerialization, allocation granularity and lifetime reuse. |
| Movement and pipeline | Direct/indirect/vector/bulk transfers, staging, prefetch, synchronous/asynchronous movement, buffering depth, pipeline stages and overlap. |
| Communication | Broadcast/collectives, shared exchange, barriers/events, permitted atomics, scratch handoff and launch dependencies. |
| Executable order and control | Local ordering, issue/wait placement, launch overlap/submission grouping, predication, guarded specialization, supported runtime assignment policies. |

These families describe one constrained execution, not independent tuning knobs.
Fusion changes liveness and ownership; layout changes instruction applicability
and communication; splitting adds merge work and scratch; pipelining changes
storage and residency. Derive all consequences together. A chosen traversal order
for search must not fix earlier choices irreversibly or remove legal combinations.

Libraries supply algorithms and their permitted alternatives. The form optimizes
their realizations; it does not search arbitrary equivalent algorithms. Backend
mechanisms may extend intrinsic contracts without introducing model-specific
decision families.

An execution assignment determines:

- Iteration domains, partition hierarchy, traversal, tails and multiplicity.
- Selected operation covers, execution regions, participant roles and ownership.
- Value producers/consumers, access maps, precision and retained/recomputed instances.
- Instance layouts, allocations, residence, alignment and live intervals.
- Actual movement, issue/completion events, visibility and storage release.
- Executable control, launch boundaries, policies and ordering dependencies.

Every result has a valid producer/access path; every effect retains its required
multiplicity and order; every intrinsic's requirements hold. Necessary communication
is explicit. Search specializes this structure rather than treating arbitrary
rewrite-pass histories as distinct candidates.

Decision identity and alternatives remain typed through lowering, accounting, and
selection. Display labels do not replace those identities. Resolving a decision
specializes the same execution structure and its derived constraints; neither the tuner
nor a diagnostic candidate path reconstructs its meaning separately.

Domains may be symbolic integer ranges or dependent alternatives. Semantic,
representation, ownership, and hardware constraints determine legality. Workload extents
and backend capabilities bound choices; handwritten preferred subsets do not establish
coverage.

For example, placing a tile in subgroup-shared storage affects capacity, communication,
publication barriers, and legal reduction implementations. Those consequences are
derived from one decision, rather than selected independently and reconciled after
emission.

Form restrictions must be explicit and meaningful. A bound for graph-preserving direct
computation does not automatically cover algebraic replacement algorithms. Expanding the
admitted form requires rederiving affected bounds and invalidating incompatible cached
analyses and selections. A diagnostic restriction cannot silently replace required
compiler coverage.

## Finite domains and coverage boundaries

Each admitted form defines its parameter and structural domains independently of
search budgets. Finite tensor sizes alone do not bound arbitrary duplication,
algorithm expansion or scheduler programs.

| Domain | Required boundary |
| --- | --- |
| Algorithm expansion | Finite alternatives and terminating, acyclic or well-founded expansion |
| Decomposition | Bounded work domains and specified partition hierarchy; include tails and empty work |
| Ownership/layout | Explicit finite families and parameter bounds, including backend fragment maps; no implicit search over arbitrary integer functions |
| Padding | Bounds from admitted layouts/instructions or an explicit form restriction |
| Reduction/order | Finite trees/orders over admitted occurrences with numerical and dependence restrictions |
| Replication/recomputation | Defined occurrence construction and bounds; no unrestricted rewrite duplication |
| Pipelines | Useful outstanding work/capacity bounds where established, otherwise an explicit supported depth domain |
| Dynamic policies/variants | Finite supported policies and guards with bounded work and progress requirements |

For a regular tiled axis of positive extent N, expose 1..N unless a justified
constraint restricts that domain; account separately for admitted padded
implementations and N=0. Powers-of-two or divisor-only lists do not establish
general coverage. Arbitrary layouts and arbitrary runtime scheduler programs are
not silently included in a bounded family.

Distinguish semantic/hardware constraints, optimum-preserving dominance arguments,
and deliberate form restrictions. Less arithmetic or storage does not inherently
dominate: extra recomputation, padding or buffering can improve communication and
concurrency. Finiteness does not establish tractable search. Unsupported analysis
is not evidence of infeasibility.

## Safety and effect obligations

Bounds, shape compatibility, aliasing, initialization, representation validity, parallel
independence, and collective convergence are typed IR predicates. “Obligation” names a
required condition, not a string assertion or a separate proof artifact. Types and
construction enforce local invariants; the owning analyses and stage validators
establish global properties. Runtime-dependent conditions remain explicit checks with
execution semantics.

| Discharge | Required behavior |
| --- | --- |
| Static validation | Construction or analysis establishes the predicate under retained conditions; emit no redundant runtime check. |
| Invocation check | Check properties of bindings or execution conditions before relying on them. |
| Dynamic check | Represent the necessary check at the execution point, its dependencies, failure behavior, and resource cost. |

Transformations may hoist or combine checks only when doing so preserves observable
behavior and validity. A masked lane does not justify an out-of-bounds access before the
mask is applied. A lane-local predicate does not establish collective convergence. An
allocation bound does not by itself prove a logical view access valid.

Remaining checks are part of the selected execution. Emission must not rediscover bounds
or introduce a second checking policy. Failure behavior distinguishes a statically
invalid program, a rejected invocation, and a dynamic execution failure; runtime
completion and partial-effect handling follow [Runtime](runtime.md).

## Transformation contracts

Every transformation establishes semantic equivalence under retained conditions,
reconstructs dependencies and value/storage relationships, and preserves or rederives
obligations and decision domains.

Fusion preserves required publications and dependencies. Recomputation preserves
numerical/effect semantics. Storage reuse requires non-overlapping live intervals and
appropriate completion ordering. A reduction rewrite requires a legal merge operation
and the relevant reassociation permissions. Floating-point identities cannot be
justified solely by real-number algebra.

An internal memory round trip may disappear while its numerical conversion remains.
Externally visible writes, KV publication and successor-state updates cannot be
removed as if they were temporary values. Asynchronous storage remains live until
all relevant users complete, not merely until their operations are issued.

Communication requires the actual visibility and participation scope. Persistent
or cooperative execution requires a valid progress contract; an ordinary GPU
launch does not imply a cross-workgroup barrier. Coupled reduction state requires
its admitted merge semantics, including masked, empty and exceptional-value cases.

Numerical and effect preservation is independent of optimizer cost improvement.
Transforms must handle aliases and snapshots, exceptional floating-point values,
partial or empty domains, and collective participation according to their semantics.
