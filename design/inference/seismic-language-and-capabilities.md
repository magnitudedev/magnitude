---
applies_to:
  - inference/seismic/**
  - inference/seismic-std/**
  - inference/engine/**
  - inference/docs/seismic/**
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

Ordinary shape domains retain boolean combinations of integer comparisons. A
`where` disjunction is a whole applicability condition: it does not establish
either alternative by itself. The checker may use established boolean
relationships and sound integer facts to prove a helper call, and rejects a
relationship it cannot establish. A reached branch can eliminate alternatives
proved false and then use the sole remaining condition; an unknown alternative
never grants another arm's facts. Exact integer interval bounds can establish
an exhaustive contiguous set of equality alternatives, but cannot fill gaps in
a declared set. Integer `min` and `max` use the existing exact
mathematical quantity semantics, including in shape expressions.

A static helper call owns a complete dimension plan from its tensor arguments
and explicit named dimension bindings. Helper recipe dimensions need not appear
in public tensor axes, but every call must bind or infer them. An ordinary
external entry still requires its public dimensions to be uniquely inferred
from input tensor axes.

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
The checked bundle exposes whether such a declaration exists for a backend without opening a
device. Declaration presence does not establish that a particular element binding or runtime
kernel can be prepared or executed.
For launch-scoped native implementations, a launch declares entry tuning parameters read by its
kernel when its geometry and activity condition do not already expose those reads. This declaration
determines which launch receives a code variant or a runtime argument and which tuning choices are
coupled. An undeclared entry parameter cannot silently affect a scoped kernel's code.

Direct Metal source receives a generated ABI prefix after element parameters are bound. The
prefix derives representation descriptors exclusively from the semantic registry for every bound
element parameter and tensor parameter/result: canonical identity, dense/packed/external kind,
decoded dtype, and for packed storage the (representation, layout) pair with its packet or row
geometry and plane encoding. These are compile-time macros. Static dimensions render as
constants, as do the extents they fix and the row geometry of a row-layout tensor with a static
packing axis; a tensor whose every extent is static also renders its canonical strides as
constants and must be bound canonically. Other dimensions, extents, strides, and scalars remain
invocation words. The Metal prefix includes the Metal 4 tensor-operation headers and defines
`SEISMIC_HAS_TENSOR_OPS` as 1 only when device discovery establishes that the GPU executes tensor
operations on its matrix hardware (Apple GPU family 10 or later, and a `matmul2d` probe forms a
pipeline); otherwise it is 0. The capability enters the device facts and, through the rendered
source, formation identity. Native assets do
not infer representations from byte lengths or reproduce registry layout tables. A Metal, CUDA or
Vulkan asset may include library files written for its backend (`.h`, `.cuh`, `.glsl`) with
`#include "<relative path>"`, resolved relative to the including file; the canonical target must lie
inside the build's source roots, so directory layout is the library's choice. The build inlines each
included file once, hashes it into the implementation identity and ABI-validates it like the asset;
it rejects vendor and system headers, absolute paths, other extensions and files outside the source
roots. `#include <seismic/<name>>` names Seismic's native library for the backend — dense element
types, packed-weight decoders and layouts, weight slot bindings — which Seismic owns because it owns
representation semantics; library files are embedded in Seismic, inlined and hashed like any include,
and include only each other. An asset binds each weight tensor it decodes to a slot of the library
(`#define KERNEL_W0 SEISMIC_GATE_WEIGHT`), and the library alone maps the bound representation to
its decoder and row geometry; no asset restates that map. Metal and CUDA select a decoder type per
slot. GLSL has no templates, so Vulkan's decoders take a constant kind, and the code that loads or
decodes a kind, in the library and in the libraries over it, is a preprocessor group of the kinds
the slots bind: a pipeline holds its own weights' decode paths alone, since drivers compile code a
constant makes dead at full cost (`seismic/hardware/rdna3-vulkan.md`). A slot bound to a
representation the library has no kind for is an `#error`, so preprocessing alone decides which
representations a slot takes. The ABI names each tensor's representation (`<prefix>_REPRESENTATION`)
for the library to resolve. A Metal implementation's
buffers, argument words and scalar slots must fit Metal's 31-entry argument table, checked at
build and at preparation.

A CPU native asset is Rust compiled into the embedding binary; generated bindings give each entry a
typed context over its ABI and include the authored file by its path. Elements the entry stores are
type parameters of the kernel over the dense types the form is compiled for: `f32`, `bf16` and `f16`
unless the declaration's `elements (A in [..])` clause lists others. A form that covers an integer type
only moves that element's storage, since its type then has no conversion through `f32`. Elements the
entry only reads are weight operands, reached through components of their representation. An
external source the entry only converts, and the packed result of a `repack` into an element
parameter, are raw row views; the registered conversion between their representations is resolved
when the kernel runs, and moves codes and coefficients bit for bit. The CPU `Rows8` resident layout
pads each matrix's row axis to eight and interleaves corresponding code and coefficient storage
groups across each eight-row tile; conversion owns complete tiles, including zero padding. The Metal
`Rows32` resident layout pads the row axis to 32 and interleaves only the code planes, per 32 columns,
across each 32-row tile; its coefficient planes hold the tile's rows one after another. Metal's
packet library addresses a row of either row layout by its tile's base and its index in the tile
(`packets::Rows16`), so a kernel that reads packets through the library runs on both. Because a CPU form is compiled
with the program, its element coverage is declared; other backends compile each binding at
preparation, and `elements` is rejected on them. Declared tuning parameters are runtime values.
CPU weight projections may declare activation INT8 as an arithmetic parameter. Their exact path is
the default; the INT8 path quantizes each staged activation row once for reuse by its dot components.
A multi-row prefill component reduces four activation rows against eight weight rows across the
whole K dimension; each instruction-set tier uses only instructions in its declared feature set.
The kernel is generic over an instruction-set tier: each form is compiled once per tier of the target
architecture and per dense element binding, behind the only target-feature boundary, and the runtime
selects forms at or below the device's detected tier. The representation-specific inner loops are
Seismic-owned components compiled once per tier, representation and row block and resolved when a
kernel is prepared; every weight representation a native weight slot binds has them, and the build
bounds their number. `.rs` files of the build's source roots that no declaration names form a library
module tree mirroring their directories, visible to CPU assets by relative path; each CPU
implementation's identity covers its asset, those files and the Seismic CPU library version. Seismic
adds two tuning parameters of its own to CPU implementations, with domains fixed by the device: the
participants of each launch and the tier.

`vulkan` is a registered, native-only backend name: it has no compiler target, capabilities or
intrinsics, so a `lower … for vulkan` body is rejected at checking. A `native … for vulkan`
declaration is checked, and its `threads_per_threadgroup` and `shared_bytes` may read only static
dimensions and tuning parameters, because a Vulkan pipeline fixes its group size and shared memory
when the kernel is prepared. Its assets include library files under the same rule. The generated
GLSL prefix enables an optional extension only after device discovery verifies and device creation
enables its required features. Workgroup-scope flexible cooperative matrices are an independent
capability from subgroup cooperative matrices; a device may provide the former while the latter
is disabled because of driver behavior. Shared memory reserved by a native launch must fit the
opened device's reported limit. The enabled matrix capability and rendered source enter formation identity.

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
Index is the nonnegative bounded refinement of that domain, not an I32 alias. An authored
I32/U32/U64 scalar operation instead has the exact fixed-width source meaning, including wrapping
and signed conversion; U64 words are full 64-bit words on every backend (a native's `ulong`,
`unsigned long long`, `uint64_t` or `u64`), never narrowed. Its symbolic projection must preserve that typed result. A mathematical rewrite may
replace it only after the applicable source facts prove equality. Converting between a quantity
and a scalar occurs at the expression's actual typed boundary, before an operation when an operand
is typed scalar and after an operation when its consumer requires a scalar.
Thus I32_MAX + I32(1) is negative, U32_MAX + U32(1) is zero, while a loop coordinate
multiplied by a shape extent retains its mathematical product even beyond a native word.
Generated access and nonempty-axis checks compare mathematical observations of the
already evaluated endpoints with the actual descriptor extent. They neither wrap the
extent into a source word nor replay endpoint arithmetic. An authored scalar extent
getter retains its declared word conversion. Participant construction admits a
mathematical extent only when its physical representation preserves that value;
unrepresentable quantities produce an ordinary capacity limitation before lowering.

## Logical ownership

A tensor source value is one of:

- an owned `tensor` value;
- a shared `&tensor` borrow; or
- an exclusive `&mut tensor` borrow.

Owned values move. Shared borrows may overlap. Mutable borrows are exclusive. Slicing borrows;
ownership copies are explicit. `let` is immutable and `let mut` authorizes mutation without
manufacturing ownership or write access.

Initialization requirements and recorded storage accesses preserve the conditions under which
an element is visited. A condition involving a private loop coordinate stays inside that
coordinate's quantified domain when the effect leaves the loop; projecting the external path
must not widen a live prefix to every iteration. Simple integer comparisons may constrain the
half-open visit interval exactly. Other guards remain explicit, and unsupported containment
remains a proof failure. Word-cast identities require established representable bounds; no
initialization is inferred from allocated capacity or from unvisited cells.

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

A native declaration has entry parameters shared by launches and parameters scoped to one launch.
Launches may reuse parameter names; a launch's condition and geometry see only its own parameters,
entry parameters, and dimensions. Scratch sees entry parameters and dimensions. A parameter marked
`code` changes its launch's generated code; other parameters are supplied at launch time. A
parameter marked `form` selects among structurally different algorithms (a vector and a matrix
form of one entry): the best values of the other parameters in one form say nothing about
another's, so tuning searches each form from a start of its own. For a
launch-scoped Metal implementation, non-code entry parameters and then non-code parameters of
each launch occupy trailing ABI argument words in declaration order. A launch source sees its
own local names as `SEISMIC_RUNTIME_<NAME>` at those offsets; it may instead read a device built-in
when the value is already
part of launch geometry. A native
Metal launch with scoped parameters has a forming guard for its kernel, and its template header
lists the code parameters it receives, entry parameters before local parameters, in declaration
order. The build checks both before formation.
The source does not use entry-wide tuning macros for such a declaration. A native
launch and a native scratch buffer may be conditional (`launch K when C:`, `scratch S bytes
(E) when C`). A condition is comparisons of that same integer arithmetic joined by `and` and `or`;
it is evaluated with the launch geometry (per standalone call, once per node when a native graph
is sealed). The same condition form restricts tuning configurations in `where`, which reads only
static dimensions and parameters. `where` may combine conjuncts from different launches, but each
conjunct reads local parameters from at most one launch; a reused local name is ambiguous there.
The default configuration at given static dimensions is the first configuration `where` admits,
in declared parameter and value order, so reordering values or adding a `where` conjunct cannot
leave admissible statics without a default. Statics that no configuration admits lie outside the
kernel's domain: graph construction rejects such a node, naming the call and its statics.
A native declaration states no precision: it has no numerical associations, error classes or
deviation markers, and it never describes how the native kernel computes. Its parameters,
domains, launch blocks and `where` condition define the tuning space; the portable function it
attaches to defines what every configuration computes. Whether a configuration's numerics are
acceptable is decided by measurement against that contract (see `seismic-numerical-precision.md`).
An inactive launch is neither encoded nor checked against pipeline or device limits, its geometry is not
evaluated, and it keeps its ordinal (formed functions and trace entries stay in declaration order;
a trace records it as an empty launch). An inactive scratch buffer keeps its ABI slot at the minimum
charge without evaluating its size. A call whose launches are all inactive is legal and does nothing.
A specialization may hold a call-time dimension to a range (`with_range`): a launch whose
condition is false at every value in the ranges is not formed (no pipeline, function or source of
its own; its ordinal stays), and a call outside a range is refused before any launch is. A backend
forms a pipeline or function per launch, and on some drivers that is most of formation (NVIDIA's
Vulkan compiles inside pipeline creation), so a caller that knows its rows forms only what they
run. A specialization without ranges forms every launch. A range is refused at preparation unless
it holds a call-time dimension of the entry with a nonempty range: on a static dimension, on a name
the entry does not have, or with its low above its high.
A native declaration may name natural-number expressions: `let NAME = E` lines after `params` and
`elements`, each over entry dimensions, entry parameters and the terms before it. A term is
substitution: the `where` condition, scratch sizes and guards, launch conditions and
geometry and the `repeat` count read its name as its expression, and the checked implementation is
that of the declaration with every term written out. A term may not take the name of a dimension,
a parameter (entry or launch) or another term.
A native declaration may dispatch one block of consecutive launches several times: `repeat (E):`
with the launches indented under it, `E` a natural-number expression over entry dimensions and
entry parameters evaluated per call (zero dispatches none of them). Every round is dispatched with
the call's one set of arguments, so a kernel learns its round from scratch the launches themselves
advance (a launch never reads a word it writes); a launch's ordinal, formed function and tuning
parameters are those of its declaration, a trace records each dispatch under that ordinal, and a
parameter the count reads is part of the block's launch geometry for tuning. Only the Metal encoder
dispatches a block more than once: the checker rejects `repeat` on every other backend.
A scratch buffer declared `sync` (`scratch S bytes (E) sync`) holds arrival counters: it is zero
whenever one of the call's launches starts, and the call's kernels restore every counter they use
to zero before the call ends. Kernels use it for "the last threadgroup to arrive finishes the
work", in which no threadgroup waits for another, so it needs no co-residency. The counter only
orders the arrivals: every buffer the last threadgroup reads that other threadgroups of the same
launch wrote must be device-coherent in the kernel's declaration: Metal `coherent(device)`
(`arrive::coherent_device`), CUDA loads that bypass L1 (`__ldcg`). Device-scope fences order the
counter but do not make plain loads see another core's stores; M1 Max-class GPUs serve them from a
stale cache. The engine kernels' build rejects a Metal entry that binds atomic scratch and declares
any other writable scratch without `coherent(device)`.

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
A Metal pipeline whose launch fixes its group size is formed to admit that size, so register
allocation cannot lower its thread limit below the declared size.
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
- Before a release is built, every native implementation of a shipped module forms with each GPU
  backend's pinned toolchain, as preparation forms it, under configurations (element bindings,
  specializations, device facts) that together compile every preprocessor group of its authored
  sources; a group no admitted configuration reaches is rejected with `#error` or removed. Every
  value of every `form` tuning parameter also forms, under each device configuration the host's
  toolchain forms, at statics searched for it where only some admit it (a form may be a template a
  constant selects, with no group of its own); a form value no configuration forms fails the check.
  Every kernel request of every supported catalog model on every GPU backend its assessment
  accepts is admissible, implemented and forms at its default specialization. A weight slot that
  is implemented at one representation of its backend's resident layout is implemented at every
  one, since planning gives any weight any of them. For a backend whose
  toolchain the operating system supplies, this holds for the verifying host's toolchain.
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
store destination; the slice metadata carries the same per-bound flags. For runtime-data
bounds, removing a supported runtime check is an optional optimization and uses only
established facts and successful proof evidence. An unresolved bound retains its check;
only the successful checked continuation owns the ensuing bound assumptions. Initialization
analysis consumes those same per-bound flags on the reached values and actual view axes,
without treating a successful bound as a produced write. Primitive accesses evaluate their
operands before the checks; assignment destinations and atomic selections reach each index
and its check before evaluating the RHS or update value. Deferred reads retain the facts
of their own successful access continuation. Static bounds retain the complete required
arithmetic proof.


Natural-bound implication uses structural monotonicity: division or ceiling division
by a positive constant cannot increase a natural value, and products preserve
factorwise ordering. Both binary multiplication and n-ary products participate in
this proof. The expression owner preserves partial-operation definedness; source
or target dimensions are never sampled or guessed to establish universal coverage.
This lets a legal row-width bound cover its packet count and a wider storage bound
cover a narrower same-shaped internal allocation.

Optional guard classification, common branch-bound strengthening, written-region
simplification, and sign strengthening use established equalities, bounds, and successful
proof evidence. Branch joins retain declared type bounds; a candidate adds a common
bound only when both arms establish it. Unknown candidates add nothing. Optional
written-region simplification retains the exact region when a simpler form is unproved. An unknown condition retains both paths;
required bounds, initialization coverage, and explicitly parallel loop separation keep
their full proof obligation. Inferring independence for an ordered loop is an optional
optimization: it uses established geometric and sign evidence, including positive
constant scaling and products of independently nonnegative factors. An unresolved
relationship leaves the loop ordered without speculative arithmetic search. Exact finite recipe domains may establish a proposition only
when every admitted, defined case proves it; exceeding the proof resource bound
leaves it unknown. Initialization coverage first tries independently sufficient
written regions before exploring unrelated conditional writes. This changes proof
ordering without granting unwritten storage or discarding a possible path.

A produced write reaches its scalar and region meaning once under its source-position
facts. Accumulation retains every prior write and guard as an exact union rather than
renormalizing the entire write history at each store. Optional whole-root promotion
belongs to materialization before branch facts, private loop coordinates, or exit
facts are projected away. Those boundaries retain the source fact world's write
meaning. Reads, carry obligations and implementation applicability still perform
full required coverage; delayed simplification never fabricates initialized cells
or permits a later write to satisfy an earlier read.

A provisional loop read with no abstract incoming initialization may be eliminated
using established coverage or retained with its exact source-position region,
available writes, path, facts, and span. Each loop resolver owns full coverage and
vacuity proof against that saved state before binding private coordinates or
projecting paths for an enclosing loop or input contract. Unknown coverage retains
the read; it never grants initialization. Reads using provisional incoming carry
assumptions keep full input-requirement re-export, so an inferred invariant cannot
replace the caller's initialization obligation.

Typing and source-position initialization share exact source operation semantics.
A value retains its typed recipe, source provenance, and occurrence identity
independently of whether arithmetic derivation establishes a simpler expression.
Word arithmetic preserves wrapping; conversions and payload transport consume
actual typed operands through the language-owned scalar recipes. A mathematical
projection is scoped derived information: no-wrap arithmetic requires bounds on
the mathematical candidate from its operands, never the word result's own type
range. Partial-operation definedness remains distinct from value bounds and from
successful source continuation.

Checked source values retain exact meaning and actual publication as distinct
views of one occurrence. Structured values preserve their scalar leaf identities.
Copies and repeated immutable projections reuse those identities; reads, calls,
assignments, branch joins, and loop versions retain their actual source lifetimes.
Executable geometry refers to the dominating published result, rather than
reconstructing a typed operation from an algebraic expression. Mathematical
header quantities keep their existing host-expression path. Execution binds each
publication in its declared quantity or scalar sort, retaining scalar dtype and
payload bits; one semantic value cannot acquire conflicting interpretations. Static versus runtime
dependence follows source inputs and operand provenance, not derivation success.
Initialization evaluates each operation over its reached actual operands; it never
reuses an earlier expression after those operands change. Private aliases and
continuation evidence cannot escape their scope as global facts. Value identities
confer no tensor initialization by themselves.

The checker has one private arithmetic derivation owner shared by bounds,
initialization, contracts, and scoped simplification. Required checks first
consume the same established relations and interval summaries as optional
queries, then use the existing bounded arithmetic rules if unresolved. Recursive
constraint queries retain their active dependency context through every operand;
a cyclic premise cannot establish its own sign, and other independent premises
remain available. An unresolved request is not a refutation or
an assumption. Optional simplification retains the original operation or effect
when established relations are insufficient. Scoped value definitions and other
premises remain distinct from cached derivation outcomes; neither a preferred
interval summary nor a failed query changes a value's meaning. Type membership
conjoins its bounds with the current source constraints. Republishing a known
value cannot replace stronger loop or path bounds with its broader type range.
Explicit scope-range replacement remains distinct from adding a premise and
cannot erase independent type or path evidence. Repeated identical membership
preserves the constraint state and its derivation caches.

Initialization contracts project private computation at each parameter's actual
logical geometry. If a required image contains an unavailable private coordinate,
only that axis expands to its parameter extent; known coordinates, prefixes, and
supported scoped domains remain correlated. Call views and reshapes supply their
logical axes before those contracts map into backing-root coordinates. The same
projection conservatively expands may-access footprints for independence checks;
it never expands guaranteed writes. Unsupported linear-address geometry can still
require the complete root.

Optional region normalization is not the authority for rejecting a read whose
domain is empty. When no initialization is available, required coverage proves
emptiness using the complete arithmetic rules, with each quantified binder and
guard scoped to its body. A successful empty-domain proof grants no initialized
cells and does not replace bounds or independence checks.

Reached scalar normalization uses one fixed fact world and an arena-bound
DAG memo shared by unchanged copies of that world. A structural fact change
starts a new owner. Reinstalling an identical bound or assumption preserves the
existing owner and evidence; replacing a primary bound starts a new owner even
when an independent extra bound has the same value. Extra assumptions retain
their own lifetime across primary-bound replacements. Newly established successful proof evidence invalidates
reached reductions, so later calls can use stronger established ordering; failed
proofs grant no reductions. Immutable exact-equality metadata remains reusable
within its fact world and arena. Cached shallow proof success can be reused before
constructing a proof engine. Cached recursive-search failure still permits the
independent finite-domain alternative; it is not a complete failure verdict by
itself. Supported exact integer arithmetic is normalized from its operands
upward, so a reached zero inside a minimum and an in-range nested word cast have
the same meaning at a later call. Unresolved minima and maxima retain their
normalized operands; selecting an operand requires established ordering. A
nonnegative reached assumption supplies a weaker constant lower bound only when
existing exact equalities reduce their difference to a nonnegative constant.
These operations do not search additional arithmetic alternatives or increase
proof depth. Unknown divisors, undefined children, opaque scalar recipes, and
coefficient overflow retain their original operation rather than becoming a
numeric placeholder or an initialized region.

Source-position write accumulation reaches each newly formed effect under its
current facts and unions it with the exact previously reached writes. It does
not repeatedly normalize the entire accumulated history or attempt whole-root
promotion after every store. Before an arm, iteration, carry view or exported
parameter guarantee loses its facts or coordinate scope, its accumulated writes
are materialized using the existing normalization and optional full-root check.
Required read coverage, saved deferred reads and alternative applicability retain
their complete authoritative proof. Incoming assumptions remain distinct from
produced writes; materialization cannot turn an incoming assumption into a write.
View projection preserves the constant regions Empty and Full directly. It does
not ask coverage to rediscover these exact inverse images. Partial regions still
use scoped whole-view coverage before projection, so a guarantee established
under private facts can be materialized before those facts disappear at export.
Read admission remains responsible for recognizing an empty access domain.

Region normalization uses this same reached scalar meaning. Binder endpoints
are normalized under outer facts before installing the binder's interval; its
body and image coordinates use only that binder's scoped facts. Guard operands
use outer facts, and its interior uses justified guard facts while retaining the
guard in the resulting region. Different fact worlds in branches and binder scopes have separate
normalization memos. One normalization batch owns the scoped reaching traversal
and geometric reduction of its separate effect results. Identical binders and
ordered guard paths under the same parent scope reuse that scope's constraint
and derivation state within the batch; different parents, paths and clipped
bounds remain separate. Export phases with different constraint states or
mandatory versus established-only policies use separate batches. Each deferred
read first resolves coverage and vacuity in its saved source state; only surviving
outward effects enter an export batch. Local failures do not create input effects.
No scope state escapes the batch or changes effect order, metadata or required
admission. Row-major conversion forms geometry without reaching scalar terms;
the consuming boundary normalizes its complete converted regions once.
Recursive separation compares those normalized children instead of repeating
the reaching traversal. World-exit materialization shares normalization across
separate roots, while each root retains its own geometry and promotion policy.
Region normalization separates reaching
from geometric reduction. After a binder's body is reached once, structural finalization
accumulates enclosing image domains, distributes unions, and lifts independent
guards without rewalking scalar descendants or rebasing each partial domain.
Each union member clips its own bound. A lifted guard remains on the output and
supplies facts only to its interior finalization; it cannot create unguarded
initialization. Geometry reduces each accumulated image domain, retains
unsupported binders and guards, and preserves the existing union joins. Newly
constructed clipped endpoints, widths and addresses use exact reached arithmetic
under their formation scope. A clipped endpoint uses the finalizer's outer facts
before its own binder is installed. Exact
required equality checking also uses this normal form. Numeric normalization never expands guaranteed writes or suppresses bounds,
independence, or missing-cell obligations.

Initialization call effects retain exact typed integer-word cast expressions. A
signed/unsigned same-width roundtrip preserves the original signed value, while
an arbitrary unsigned-to-signed cast retains its modular interpretation. Remainder
normalization may use an established integer interval. Reached numeric assumptions
normalize under the facts that held before they were added, so equivalent modular
values retain the same prefix guards. Signed affine interval bounds follow coefficient
sign; nonlinear products require nonnegative factors. Minima and maxima retain
independently valid bounds from both operands. These rules never assume an unsigned
count fits the positive signed range or grant unwritten tensor coverage. Under an
established zero equation, adding or subtracting that equation preserves a quotient
numerator. A positive constant quotient reduces only when this equivalent numerator
is exactly divisible as checked integer arithmetic and the result has no quotient or
remainder atoms. Missing relations, inexact division and coefficient overflow retain
the original operation. The same equation normal form applies to reached assumptions;
it does not alter the required proof depth or infer consistency of the facts.

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

Provably nonnegative signed additions, products and divisions of natural shape values
canonicalize to the same natural arithmetic DAG, preserving division-by-zero failure. Potentially negative signed
expressions retain checked conversion and their original definedness conditions.

## Ordinary rounding and payload primitives

`rint(x)` returns the nearest integral value in the operand's floating dtype,
with ties to even. It preserves signed zero and infinities and canonicalizes
NaNs through the same language-owned scalar recipe as ordinary arithmetic.
It is scalar or elementwise; it does not change storage type or quantize a row.

`bitcast_u32(x)` accepts F32 and `bitcast_f32(x)` accepts U32, scalar or
elementwise, preserving every payload bit. They perform no numeric conversion,
including no NaN canonicalization. Checked numeric casts remain separate.
The scalar reference constructs both operations from existing terminal word
and payload operations, and portable lowering reuses the existing typed IR
bitcast. Quantizer gain rules, coefficient types, partner-row coupling and
accumulation recipes remain ordinary checked computations around these
primitives; the primitives carry no tolerance or native trust authority.

`exp2(x)` owns a separate base-two scalar recipe: it splits off an integer
power-of-two scale before evaluating the bounded fraction, then publishes once
in the declared dtype. Its bit-built scale handles subnormal output and
exceptional inputs. It is not a source alias for `exp(x * ln(2))`; that
expression rounds its entire transformed input first. Backend approximate
exponentials remain separate physical choices assessed against this recipe.

## Logical quantized components

Checked quantized component reads take a packed tensor and its logical element
indices, with the same bounds checks as an element read. The named `code`,
`scale`, and signed `bias` roles are the integer code and coefficient results
already constructed by that representation's owned decode recipe. Hierarchical
coefficients keep their existing F32 multiplication order. An integer format
without bias returns exact zero; a representation without an integer code
cannot bind the integer code role. The distinct F32 `code_value` role selects
its existing pre-scale converted code temporary, including registered floating
codebooks; it does not decode a complete weight and divide out a coefficient.
Support is checked against the actual representation binding, rather than
inferred from a decoded F32 value or byte size.

The same logical interface exposes the quantization group, interpreted code
bounds, bias presence, and coefficient factorization used by a checked fold.
For a hierarchical coefficient, `scale_factor` and `bias_factor` are the
existing converted super-coefficient temporaries, before coefficient codes and
bias sign. A representation may scale its coefficients by a registered exact
power of two after that product (`iq2xxs` stores its f16 block scale d and
decodes `d * code * 2^-3`); the factor role stays the stored factor, so a fold
over it applies the representation's exponent. The i32 `scale_code` and `bias_code` roles select the owned interpreted coefficient-code temporaries; direct coefficients use one (absent bias uses zero), and neither role requires an integer weight code. `coefficient_code_max` comes from that coefficient's owned integer
interpretation. Direct coefficients expose their own value with multiplier one.
These roles describe the registered representation, not PACK availability or a
numerical permission. Their constant results require no physical field read.

The interface exposes no plane ordinal, byte offset, layout spelling, or
independent coefficient reconstruction. Portable IR and the interpreter select
the same owned decode temporaries. Physical layouts retain their existing field
read lowering and identities. Numerical helper computations can therefore name
coefficient and code rounding without duplicating format tables or expanding
the entry ABI with shadow weight tensors.
