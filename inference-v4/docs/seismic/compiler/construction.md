# Physical construction

Physical construction turns the actual checked computation into executable structure. It owns source correspondence, complete values, effects, scoped control and the resource projections of those operations.

| Input | Output |
| --- | --- |
| Checked operation/region, already-bound operands, target and active choices | Complete typed results and effects, constructed physical regions and composed numerical meaning |

The compiler owns traversal. Optional construction choices select realizations of the current operation or region; they cannot emit arbitrary computation and then label it with a source identity. The backend registry supplies its closed intrinsic vocabulary and target contracts. Raw kernel, schedule, publication and constraint APIs remain trusted internal mechanisms; there is no external arbitrary-IR factory registration.

A candidate family names an immutable authored root body and its implemented construction choices.
The domain owns the materialized physical data separately. Definitions do not disappear when
optional construction receives no effort or when a cached member is evicted. Evaluators own which
partial coordinates to advance; the domain owns the corresponding construction state.

Coordinates name the checked root body, construction-time call/selected-arm path and typed
physical axes in deterministic construction order. Expression `DecisionId`s and physical cache
digests do not define coordinate identity. A resumed or rebuilt member maps those axes to its
own private expression parameters. Runtime loop visits execute the same closed choices.

The domain advances only the path requested by the evaluator. It reports a next choice, a closed
physical member, an established exclusion, or pending work. Work exhaustion retains the actual
constructor; unresolved optional initialization retains the same selected continuation. Evaluator
queues revisit pending work fairly. Closed physical data is not numerical admission or authority
to execute.

The required general member uses this same constructor. Its deterministic driver selects the
reference child at each call and distinct static allocation sites; dynamic instances still follow
the ordinary bounded storage-bank protocol. The resulting general coordinate is an ordinary
coordinate, and only equality with that coordinate selects required resource accounting. No
construction label grants numerical exactness. Numerical admission still follows the actual
closed outcome relation.

The domain retains one append-only expression arena. A constructed handle pins its immutable
physical data and that same expression storage. A read view borrows the domain as well as the
handle, so construction cannot advance while the view is alive. The private shared lock implements
storage lifetime and borrowing; it grants no separate construction or execution authority. Native
services consume bounded views and do not reenter construction through them. Published kernels
retain their compiled guards and requirements independently of later domain cache changes.

A suspended constructor retains its actual physical builder state, bound values and lexical source
position. Resuming temporarily borrows the same immutable checked program and expression owner.
Calls and branch arms preserve their current initialization and alias state across these pauses;
completion consumes the child product once. A suspended kernel similarly retains operations and
values in its existing construction owner; its noncopyable cursor names the current lexical block.
Opening another kernel cannot overwrite that state, and execution closure requires it to be closed.

The region owner derives a legal frontier from checked dependencies and actual bound versions. Optional choices may realize compound contributions, fuse work or share staging across that frontier. Reference construction follows source order; that strategy does not freeze every optional physical ordering. Optional providers cannot assemble arbitrary source labels to claim a region.

## Structural contract

These rules apply to every normal path, including calls, imports, specialization and normalization:

| Rule | Required construction |
| --- | --- |
| C1: Meaning and value together | An operation derives result expressions, types, shapes, bounds and effects from its actual operands. There is no independently supplied symbolic result. |
| C2: Complete producer results | Every definition returns its entire checked result product. Tensors are stored or computed; an absent producer is not a third state. Effects are constructed even when there is no returned value. |
| C3: Transport preserves identity | Copying, materializing, packing or publishing preserves semantic origin, versions and rounding. A scalar slot is a transport, not a new meaning. |
| C4: Expressions retain dependencies | Constructors derive required bindings. APIs accept expressions only in environments where their operands are available. |
| C5: Regions close local state | Branch joins, calls and loops construct complete parent results and project resource demand without escaping local handles. |
| C6: Resources follow execution | Closure derives layout, scratch, ABI and lifetimes from the normalized execution it consumes. Independently supplied resource tables are forbidden. |
| C7: Closure completes construction | A closed executable is the result of these operations. A late scan, rejection, guard restriction or consumer reconstruction cannot repair missing coverage. |

For a supplied `range[N]`, C1 preserves actual endpoints; `N` may bound capacity but never replaces the range. For an invocation scalar carried through a device slot, C3 preserves its original invocation expression. For a truly device-computed scalar, C4 prevents its use in an invocation-time guard. These distinctions prevent whole classes of errors at their owning boundaries.

## Complete value construction

A computed tensor owns its operation, captured operand versions, indexing and representation/rounding. A stored tensor owns its logical storage/view, initialized region, version and availability. Both are realizations of a defined value.

An element consumer evaluates the bound realization. An addressable consumer materializes that same realization. Neither revisits a source ID to recreate a skipped producer. Views transform indexing while retaining the captured value. Before a conflicting write, old-version uses must complete or the value must be preserved; conservative materialization is the general choice.

The concrete computed producer retains its complete tensor result type and the actual scalar operands captured by its shape. Physical axes are derived from those captures and transported with the value. Preservation follows source write events through helper and region captures, then compares actual backing storage under the entry alias contract. A pre-existing value is preserved before a control region that may overwrite its captured storage; values defined inside that region are preserved before their own conflicting write. Selected products preserve only the chosen arm. Logical axes remain view geometry; any reserved capacity determines backing strides and never replaces the logical shape.

Calls, branches and loops use this same complete value product. Calls accept computed or stored operands and may return computed or region-owned values. A common caller destination is required only when the chosen realization uses it. External ABI publication is a separate consumption of the result, not the definition of every internal call boundary.

Import is one construction operation: it remaps physical owners and inserts the child's ordered
lexical schedule scope together. The closed schedule retains that scope, including for Unit and
effect-only calls. Repeated and nested call occurrences retain distinct scopes and their actual
formal-to-argument binding correspondence. Native execution traverses the scope as an ordered
sequence. Source associations on physical bindings route numerical analysis; they never assert
that two computations are equal. Historical bindings remain physically traceable after local
storage leaves scope, while initialized contents transfer only through live escaping products and
the checked call's effects on existing roots.

Allocation initially yields a writable place. Complete maps, fills and writes establish initialized regions. A branch selecting different allocations returns the selected allocation, version and initialization together. An unconditional read of the same shared allocation after either arm requires the corresponding region to be initialized on both paths. Loop carries preserve their checked invariant; empty loops return their initial values. A partial write cannot produce a supposedly complete readable tensor.

The language checker owns the initialization calculus and derives each body's callable transfer. Construction applies it to actual argument views and initialized state. Optional bodies may require stronger initialized input when the caller supplies it, but must still provide the reference call's promised initialized post-state. Private analysis predicates do not escape as compiler assertions; callable requirements conservatively cover private outcomes, and callable guarantees hold across them.

Initialization is checked in source order over actual logical views and shared storage roots. View
formation does not read contents; a point or slice write initializes precisely that region. Complete
joint loop images, ordered earlier iterations and actual helper substitutions use the same region
calculus. An optional body must both accept the actual incoming initialized state and preserve the
reference call's promised post-state. All aliased argument leaves update one root state.

An owned tensor carry has an initialized-region invariant in its own logical coordinates. Bounded
inference intersects initial and yielded regions, closes private predicates and binders, and then
checks the body once under that candidate invariant. Initial and every next state must preserve it.
Zero and one iteration retain their exact states; later fresh partial storage cannot inherit an old
allocation's complete initialization. A failure to establish this invariant is a static construction
limitation, distinct from a demonstrated uninitialized read.

An operation check remains at its source position even when its predicate is known from invocation parameters. Explicit types and declared source preconditions define entry admission. Moving a later failure before earlier writes changes source behavior. Construction preserves the failed operation's prefix effects and stops its dependent continuation; collective protocols must also let failed participants reach required completion gates without executing later source work.

Memory accesses and possible source failure use the same semantic event owner and predecessor
chain. A scalar or elementwise operation derives its failure effect from its actual typed language
recipe; explicit bounds checks retain their checked meaning. Calls preserve their callee's failure
ordering even when they have no tensor effect. Tensor evaluation that may fail occurs at its
definition, including when its value is unused. Construction consumes the recipe's typed value and
all failure predicates together, publishes actual causes to planned status storage, and binds the
result only through its successful continuation. Internal dead-path transport bits are never
successful source results.

## Expression environments

One expression DAG retains mathematical Nat/Int meaning. Node constructors derive immutable required-binding sets. Operations combine requirements; substitution uses actual substituted operands; a fold removes precisely its own binder. Simplification preserves value and definedness, including lazy branches.

Construction owns an entry environment and nested region environments. Private scoped handles refer to DAG nodes admitted in that environment. Operand acquisition enforces owner, binding membership and dominance locally. There is no public conversion from an arbitrary expression plus a caller-supplied dependency list.

Invocation expressions use parameters and fixed target/choice facts. Execution expressions can additionally use dominating produced values and active lexical binders. Environment IDs are implementation ownership, not cache identity. A final free-symbol scan may diagnose a defect or check external data; it is not the normal operation for granting invocation scope.

Semantic origin/version, lexical and participant environment, and execution availability are distinct relationships owned by actual values and operations. Lexical dominance does not imply visibility on another execution context, and uniformity does not imply completion. A varying value crossing a boundary retains transport indexed by its actual participant; one scalar slot cannot represent every participant's value. No attachable scope or readiness certificate supplies these relationships.

A kernel loop carries an inductive uniformity schema as part of its ordinary value construction. Initial values and every next value must satisfy the same scope, and the loop result keeps that scope even on zero iterations. An initially uniform value cannot silently become varying on a backedge. Reads of participant or register storage include that private address scope even at a constant index; shared address uniformity alone does not establish memory visibility or a stable observation. The subgroup ordinal is an actual native geometry value uniform within its subgroup, not a uniformity annotation applied to arbitrary lane arithmetic. Collective control uses these constructed origins and the actual completion protocol.

| Region | Result and resource closure |
| --- | --- |
| Branch | Build arms in child environments; join selected values and state. Exclusive resources may reuse capacity. Preserve lazy conditions where available before execution. |
| Loop | Bind actual endpoints and typed carries; consume the complete next-carry product; return initial carry on zero iterations. Close body-local indices and resource demand at the parent. |
| Call | Substitute actual parameters/versions, construct the selected body, return its complete result product and remap physical owners together. |
| Asynchronous operation | Issue returns pending results; its actual ordered completion exposes those results to subsequent consumers. |

Sequential reusable scratch folds by maximum over iterations, with zero demand for an empty range. Concurrent live demands add; persistent carries remain live. Nested loops close from the inside out. Removing a loop binder does not remove execution-produced endpoint dependencies: reserve a safe source-derived capacity envelope or use an explicitly planned later resource lifecycle. Exact execution geometry remains separate from that capacity.

## Reference completeness

The general recipe traverses the complete checked vocabulary and composes these cases:

| Checked cases | Required general behavior |
| --- | --- |
| Primitive | Registry reference recipe, exact types and rounding/exceptional behavior. |
| Elementwise, reduce | Full logical domains, broadcast/index mapping, correct association and empty identity, complete results. |
| Intrinsic | Typed target participation, result, numerical, resource and completion protocol. |
| Call, tuple pack/get | Complete substitution, effects and result products with ownership preserved. |
| Alloc, fill, copy | Logical identity, initialization, independent copies of the actual viewed value. |
| Representation conversion, view | Every plane/packet, conversion rounding, exact geometry and alias rights. |
| Element read/write, store | Actual indices, initialization, versions, permissions, rounding and publication. |
| Atomic | Actual combined operation, participant scope/order and allowed outcomes. |
| If, loop | Actual condition/endpoints, complete joins/carries, effects, empty behavior and scoped resources. |
| Check, extent | Source failure behavior and logical value shape, independent of physical chunk dimensions. |

Ordered source operations use their reference ordering and rounding. Parallel work may use a serial schedule only where its semantics permits; collectives and participant intrinsics retain their protocol. Internal values use direct or segmented storage as needed. Target-only operations require actual capability support.

The completeness argument covers each constructor and induction through regions/calls. No admitted operation can depend on a consumer-specific fallback, benchmark corpus or arbitrary alternative-equivalence solver. Every legal source composition requires a general recipe; a late guard rejecting that composition violates reference completeness.

See [physical IR](physical-ir.md) for the resulting representation and [numerics](numerics.md) for its composed applicability.
