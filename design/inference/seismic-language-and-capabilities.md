---
applies_to:
  - inference-v4/seismic/**
  - inference-v4/seismic-std/**
  - inference-v4/engine/**
  - inference-v4/validation/**
  - inference-v4/docs/seismic/**
---

# Seismic language and target capabilities

Seismic source describes logical computation. It does not describe physical partitioning,
placement, launch geometry, instruction fragments, or pipeline scheduling.

## One semantic registry

Typing, reference execution, logical construction, backend legalization, and
numerical analysis use one closed registry of primitive signatures. A
signature owns its parameters, result and effect functions, safety function,
reference semantics, and reference numerics; capability signatures add exact
argument and result types, semantics, and numerical transfer. There are no
independent per-phase signature tables: changing a semantic decision changes
the registry, and with it capability, logical, plan, cache, and evidence
identities.

`If`, `Loop`, `Call`, and function boundaries are graph structure, not
primitives. Every checked construct has exhaustive registry handling in the
reference interpreter, logical construction, and every backend legalization;
a construct that cannot be represented is a checking failure, never a later
compiler defect.

Every memory-affecting checked operation carries its complete semantic event:
region, access kind, representation, logical participant domain, atomicity,
memory order, visibility scope, ordering dependencies, and permitted numerical
outcome class. This is derived exhaustively from the sealed checked-node
vocabulary, not authored in a side table. A reads/writes summary may be derived
for convenience but is never semantic authority.

Checked nodes consume opaque, identity-bound capabilities constructed by the
checker. A parallel non-atomic write requires an exclusive-region capability;
an atomic read-modify-write requires its operation, participant domain, relaxed
order, containing scope, publication edge, and association outcome; a barrier
requires a uniform cohort and visibility contract. Unsupported analysis cannot
construct the node. The oracle represents either a deterministic result or an
allowed outcome relation and never uses one traversal order as the definition
of unordered parallel execution. Its consuming execution returns a complete outcome
that owns results and final tensor input backing independently of the interpreter
and semantic arena. Allowed associations follow actual executed nodes, including
called bodies, and are deduplicated rather than retained as an execution trace.

## Computation and implementation choice

`fn` is the sole named-computation abstraction. A portable function body is executable behavior.
A backend `lower` contributes another implementation candidate for the same function contract. A
backend-specific `fn` is a helper available only within definitions for that backend.

A top-level `native <function> for <backend> from <asset>` declaration may attach one explicitly
selected native implementation to an ordinary portable function. It does not create another named
computation, repeat the function signature, participate in static calls, or become an
implementation candidate. The portable function remains the complete type, ownership, effect,
shape, and reference-semantic contract. Generated callers select this distinct path with
`native_for_device`. A native kernel may be called directly or composed into a prepared native
workflow through the same checked entry contract. Native workflow composition does not make the
native implementation a portable compiler candidate.

Direct Metal source receives a generated ABI prefix after element parameters are bound. The
prefix derives representation descriptors exclusively from the semantic registry for every bound
element parameter and tensor parameter/result: canonical identity, dense/packed/external kind,
decoded dtype, and for packed storage the (representation, layout) pair with its packet or row
geometry and plane encoding. These are compile-time macros. Static dimensions render as
constants, as do the extents they fix and the row geometry of a row-layout tensor with a static
packing axis; a tensor whose every extent is static also renders its canonical strides as
constants and must be bound canonically. Other dimensions, extents, strides, and scalars remain
invocation words. Native assets do
not infer representations from byte lengths or reproduce registry layout tables. A Metal or CUDA
asset may include its backend's shared device library, `common/<name>.h` or `common/<name>.cuh`
beside the asset; the build inlines, hashes and ABI-validates included files like the asset and
rejects every other include, vendor and system headers included. A Metal implementation's
buffers, argument words and scalar slots must fit Metal's 31-entry argument table, checked at
build and at preparation.

`vulkan` is a registered, native-only backend name: it has no compiler target, capabilities or
intrinsics, so a `lower … for vulkan` body is rejected at checking. A `native … for vulkan`
declaration is checked, and its `threads_per_threadgroup` and `shared_bytes` may read only static
dimensions and tuning parameters, because a Vulkan pipeline fixes its group size and shared memory
when the kernel is prepared. Its assets may include `common/<name>.glsl`. Until a Vulkan ABI prefix
and runtime exist, the build refuses a Vulkan native implementation (it cannot be ABI-validated),
and discovery reports a `vulkan` diagnostic that this build has no Vulkan runtime, so a request for a
Vulkan device fails with that reason.

At each static call occurrence, compilation considers every applicable portable body and every
applicable lowering for the selected backend. Portable bodies are not fallback implementations and
backend lowerings receive no implicit priority. A candidate is available only when its complete
recursive dependency tree is available.

Compilation roots are selected externally. Source files end in `.seismic`; paths and filenames do
not grant capabilities.

Index arguments and range endpoints use natural-number symbols in the call contract and
invocation bindings. A bounded range carries its actual start and end values. Its declared upper bound is a proof
constraint, never an endpoint substitution. Loop construction preserves those endpoint projections
and derives iteration/index bounds separately. Symbolic runtime integer values retain their
mathematical integer sort independently of scalar storage representation.

Dimensions, range endpoints, loop coordinates, and their arithmetic are mathematical quantities;
Index is the nonnegative bounded refinement of that domain, not an I32 alias. An authored I32/U32
scalar operation instead has the exact fixed-width source meaning, including wrapping and signed
conversion. Its symbolic projection must preserve that typed result. A mathematical rewrite may
replace it only after the applicable source facts prove equality. Converting between a quantity
and a scalar occurs at the expression's actual typed boundary, before an operation when an operand
is typed scalar and after an operation when its consumer requires a scalar.
Thus I32_MAX + I32(1) is negative, U32_MAX + U32(1) is zero, while a loop coordinate
multiplied by a shape extent retains its mathematical product even beyond a native word.

## Logical ownership

A tensor source value is one of:

- an owned `tensor` value;
- a shared `&tensor` borrow; or
- an exclusive `&mut tensor` borrow.

Owned values move. Shared borrows may overlap. Mutable borrows are exclusive. Slicing borrows;
ownership copies are explicit. `let` is immutable and `let mut` authorizes mutation without
manufacturing ownership or write access.

Logical tensor operations accept values independently of their current storage realization.
When an operation requires an addressable input, the compiler materializes a computed value;
authors do not insert `load`, casts, `to_owned`, or scalar loops merely to satisfy an internal
representation. `to_owned` ensures ownership: it copies a borrowed value and is the identity on
an already-owned value; owned allocation constructors return owned storage exactly once, so
`to_owned(zeros_like(...))` is never required. A kernel edited to fit a checker gap is a
compiler defect at the owning transformation, never an accepted source.

Functions return owned outputs or explicitly mutate `&mut` parameters. Parameter modes, alias
declarations, publication statements, source views, and source tiles are not part of the language.
The ABI may use hidden result buffers and legal storage reuse while preserving those semantics.

At an entry boundary, an owned tensor result is represented by a root-ABI
result binding; a tuple result is traversed recursively and each tensor leaf
retains its logical tuple path. The root ABI is created once from the entry
interface and the canonical leaf traversal; it is the only ABI. Nested call
boundaries resolve directly to caller transports and never own public
allocations. The runtime allocates result planes by path, retains them
through native completion, and returns the path-labelled result planes to the caller. A prepared
native workflow derives intermediate storage lifetimes from the checked root ABIs and reuses
compatible storage after its last consumer. Results exported beyond workflow completion have
distinct ownership and capacity from reusable scratch. Backends consume this complete ABI without
exposing destination parameters or storage planes in Seismic source.

A result-bearing capability intrinsic produces one fresh owned logical value. The compiler
preserves that operation atomically through checked and logical IR with a distinct owned
destination; it does not reinterpret it as unrelated elementwise work. Backend realization
receives the typed operands, typed result, and destination identity together. Physical allocation
and storage reuse remain compiler/backend decisions and never enter source syntax.

## Iteration

`for` is ordered ascending iteration over a bounded logical range. `parallel for` asserts that its
logical iterations are independent. Parallel iterations may write only provably disjoint places or
portable atomics with explicit semantics.

Current reduction-style source atomics do not publish the previous value. They
are relaxed read-modify-write operations at the smallest scope containing all
logically contending participants, and their writes become visible through the
enclosing command-completion edge. Message-passing operations require explicit
acquire/release semantic operations; they cannot be inferred from reduction
atomics.

Physical tiling, vectorization, fusion, staging, pipeline depth, participant mapping, storage
placement, synchronization, and launch geometry are generated by the compiler. Physical tiles and
fragments may exist in compiler IR but never in source.

The only physical source exception is the launch tuple on an explicitly selected top-level native
implementation. It is closed integer arithmetic over the attached function's inferred dimensions
and is consumed only by the direct native runtime; it is not visible to portable bodies, lowerings,
static calls, or compiler planning.

A native launch and a native scratch buffer may be conditional (`launch K when C:`, `scratch S bytes
(E) when C`). A condition is comparisons of that same integer arithmetic joined by `and` and `or`;
it reads every entry dimension and tuning parameter and is evaluated with the launch geometry (per
standalone call, once per node when a native graph is sealed). The same condition form restricts
tuning configurations in `where`, which reads only static dimensions and parameters. An inactive
launch is neither encoded nor checked against pipeline or device limits, its geometry is not
evaluated, and it keeps its ordinal (formed functions and trace entries stay in declaration order;
a trace records it as an empty launch). An inactive scratch buffer keeps its ABI slot at the minimum
charge without evaluating its size. A call whose launches are all inactive is legal and does nothing.

## Backend capabilities

Every author-visible backend capability is a namespace containing a coherent family of typed
semantic intrinsics. A backend definition explicitly declares `requires <backend>.<capability>` and
calls operations through that same namespace. Source cannot query or branch on capability support.

The initial capability inventory is:

- `metal.subgroup`;
- `metal.matrix`;
- `cuda.subgroup`; and
- `cuda.matrix`.

Capabilities name semantic operation families, not hardware models, versions, datatypes,
instruction generations, matrix shapes, or physical mechanisms. Exact operand representations,
accumulator types, scale formats, sparsity, and numerical behavior distinguish typed operation
signatures within a family.

A new namespace is justified only when an author must change a backend implementation's semantic
structure, the operations form a coherent family, the facility cannot be an overload of an
existing family, and the compiler cannot select it beneath an existing logical operation.

Calling a capability without declaring it, declaring a capability for the wrong backend, or
declaring an unused capability is an error. Backend-specific helper calls require callers to
declare a superset of the helper's capabilities. Portable calls remain portable: capability
requirements of individual child candidates are handled by recursive candidate construction.

## Effective targets

Each backend gathers device, driver, toolchain, and backend-revision facts and the core assembles
one immutable device legality description: supported intrinsic signatures, device-wide
limits, dtype and atomic support, numerical environment, and a canonical identity.
Candidate-specific native reflection completes admission during preparation.
Unknown is distinct from unsupported.

For an intrinsic signature, availability is the intersection of:

- hardware support;
- driver, OS, and API support;
- shader/PTX compiler and SDK support; and
- implemented Seismic backend support.

Hardware names and raw versions never appear in kernel source. When runtime queries are
insufficient, a narrow compile and native-pipeline probe establishes support before selection.

Metal device opening creates the service/queue and device legality description.
Analytical characterization is acquired only for analytical evaluation; feedback
evaluation and direct native execution do not require an analytical profile.

Capability filtering occurs before solver export. Resource legality and numerical admissibility
remain separate hard constraints; estimated cost is the objective among surviving candidates.

## Numerical behavior

Every typed intrinsic signature has an exact numerical contract or is numerically unknown. Matrix
operations describe operand interpretation, scaling, accumulation, association, rounding,
saturation, and exceptional values. Subgroup reductions and scans describe their association
topology.

The caller's whole-program precision policy decides admissibility. Global fast-math flags remain
disabled, and final emission cannot introduce an unassessed numerical choice.

## Runtime and identity

Preparation covers the entry's full inferred target domain: the semantic domain implied by
types and source constraints, intersected with target representability. Consumers supply
tensors and ordinary parameters at invocation. Optional typed values/ranges at
preparation direct optimization effort while preserving the full inferred call
domain. These ranges imply no application-frequency distribution or runtime tuning.
Capability, static resource, numerical, or native compiler failure cannot first
appear during inference; once a call is accepted, no compiler-structure failure
is possible. A planned reached acquisition may still report a capacity refusal
for an execution-produced size or unavailable live memory, without changing
the candidate or the accepted call domain.

Compiler-prepared variants, compiler-native artifacts, and numerical applicability bind to module semantic hash,
entry identity, backend/compiler version, target-profile identity, precision-policy identity,
implementation/variant identity, and native toolchain identity. Device names are diagnostic,
not semantic cache keys.

## Failure classification

- An unavailable capability or intrinsic signature makes one candidate inapplicable.
- No applicable implementation on a target is `NoApplicableImplementation`.
- No admissible implementation under the precision policy is `NumericalPolicyInfeasible`.
- A domain the target cannot represent is `TargetDomainUnrepresentable`.
- Native compilation reports only toolchain, malformed output, device loss, cache, and toolchain
  resource failures; a contradiction with the target profile is a compiler bug (panic), never a
  result.

These outcomes are never converted into runtime fallback behavior.

Direct top-level native implementations have no numerical-policy selection, modeled duration, or
fallback. They may use reduced-precision arithmetic and explicitly called fast math functions;
their numerical admission is the consuming application's empirical precision gate. Global
fast-math modes remain disabled on this route as well. Their source bytes are captured with the checked module; source bytes and the attached
function contract participate in bundle and generated identity;
Metal compilation or execution errors are reported directly.

## Acceptance criteria

- Portable functions, lowerings, and backend helpers contain no physical tile, storage, launch, or
  pipeline syntax; direct top-level native declarations contain only their explicit launch tuples,
  scratch sizes, tuning domains and conditions (`where`, `when`).
- An inactive native launch does no device work and is exempt from limit checks; its geometry and
  an inactive scratch buffer's size are never evaluated.
- Ownership and bounded iteration determine legal reads, writes, moves, and parallel effects.
- Every accepted write to storage shared across parallel participants carries
  an exclusive or atomic capability for those participants. Iteration-local
  storage remains private; nesting depth alone does not imply sharing. Every
  atomic and barrier owns explicit participant, order,
  scope, visibility, and outcome semantics.
- Event projection, oracle interpretation, and lowering exhaustively match the
  same sealed checked-node vocabulary without default arms.
- Every backend intrinsic use has a matching explicit capability declaration.
- Capability availability is derived from hardware, software/toolchain, and backend implementation
  support before solver search.
- Unsupported specialized candidates do not remove applicable portable candidates.
- Every physical resource requirement is represented before native compilation.
- Capability and numerical identities participate in reconstruction and cache validity.


### Source index values and bounds

A checked loop index conversion retains its exact symbolic source expression.
The index type's bound constrains admissible values; it is not a representation of
the value and cannot be inverted to recover one during lowering. Reference
execution and native lowering consume the same symbolic operation. Launch counts
use the expression language's ceiling division, preserving zero extents without
inventing subtraction preconditions.

The checker records, per indexed axis, whether each point or range bound is proved in
its source scope. Semantic lowering emits a runtime source check exactly for the bounds
the checker did not prove, whether the access is an element read, a slice view or a
store destination; the slice metadata carries the same per-bound flags.


Natural-bound implication uses structural monotonicity: division or ceiling division
by a positive constant cannot increase a natural value, and products preserve
factorwise ordering. Both binary multiplication and n-ary products participate in
this proof. The expression owner preserves partial-operation definedness; source
or target dimensions are never sampled or guessed to establish universal coverage.
This lets a legal row-width bound cover its packet count and a wider storage bound
cover a narrower same-shaped internal allocation.

Preparation-time reference execution can carry an explicit semantic-work limit.
Nodes, loop iterations, element accesses, and allocation/view construction consume
that budget before work or allocation. Exhaustion is a resource outcome, separate
from a source-check failure. Reference tensor views share immutable index maps, so
copying a logical value does not copy its whole tensor index map on every element
operation. The oracle remains a validation tool and is not an execution fallback.

External-call dimension inference matches observed shape expressions by integer
value on their defined domain, including integer/natural conversion wrappers.
This is an observation identity, not an arena rewrite: original axes and their
partial-operation conditions remain intact. The complete triangular solve runs
before original equations are checked, because an observed product can eliminate
one dimension before its constituent dimensions are individually known. Inexact
inversion, zero divisors, undefined axes, and inconsistent extents are rejected.

Loop carries represent changes to values. An element write changes the contents
of its existing place and keeps its storage identity in ordered and parallel
loops alike; leaf events retain the mutation ordering. Copies of views preserve
the view's geometry and acquire independent storage. An index-map prefix is not
an identity view of the entire backing tensor.

Provably nonnegative signed additions and products of natural shape values
canonicalize to the same natural arithmetic DAG. Potentially negative signed
expressions retain checked conversion and their original definedness conditions.
