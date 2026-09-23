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
of unordered parallel execution.

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
decoded dtype, packet geometry, and packed-plane layout and encoding. These are compile-time Metal
macros, while dimensions, extents, strides, and scalars remain invocation words. Native assets do
not infer representations from byte lengths or reproduce registry layout tables.

At each static call occurrence, compilation considers every applicable portable body and every
applicable lowering for the selected backend. Portable bodies are not fallback implementations and
backend lowerings receive no implicit priority. A candidate is available only when its complete
recursive dependency tree is available.

Compilation roots are selected externally. Source files end in `.seismic`; paths and filenames do
not grant capabilities.

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
one complete `TargetProfile` before ordinary planning: exact supported intrinsic signatures, quantitative
limits, dtype and atomic support, numerical environment, and a canonical identity. Every fact
that can change admissibility or selection is in the profile; native compilation never learns
one later. Unknown is distinct from unsupported.

For an intrinsic signature, availability is the intersection of:

- hardware support;
- driver, OS, and API support;
- shader/PTX compiler and SDK support; and
- implemented Seismic backend support.

Hardware names and raw versions never appear in kernel source. When runtime queries are
insufficient, a narrow compile and native-pipeline probe establishes support before selection.

Metal device opening creates the service and queue immediately but acquires the complete target
and execution profile lazily. Ordinary compiler preparation and capability introspection force
that acquisition. A direct top-level native call does not: it needs only the checked call contract,
raw buffer service, authored launch, and concrete Metal pipeline compilation.

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
tensors and ordinary parameters, never workload envelopes, shape buckets, specialization
classes, or expected dimensions. Capability, resource, numerical, or native compiler failure
cannot first appear during inference; once a call is accepted, no compiler-structure failure
is possible.

Compiler-prepared variants, compiler-native artifacts, and numerical evidence bind to module semantic hash,
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
fallback. Their source bytes and attached function contract participate in generated identity;
Metal compilation or execution errors are reported directly.

## Acceptance criteria

- Portable functions, lowerings, and backend helpers contain no physical tile, storage, launch, or
  pipeline syntax; direct top-level native declarations contain only their explicit launch tuple.
- Ownership and bounded iteration determine legal reads, writes, moves, and parallel effects.
- Every accepted parallel write is constructed with an exclusive or atomic
  capability; every atomic and barrier owns explicit participant, order,
  scope, visibility, and outcome semantics.
- Event projection, oracle interpretation, and lowering exhaustively match the
  same sealed checked-node vocabulary without default arms.
- Every backend intrinsic use has a matching explicit capability declaration.
- Capability availability is derived from hardware, software/toolchain, and backend implementation
  support before solver search.
- Unsupported specialized candidates do not remove applicable portable candidates.
- Every physical resource requirement is represented before native compilation.
- Capability and numerical identities participate in reconstruction and cache validity.
