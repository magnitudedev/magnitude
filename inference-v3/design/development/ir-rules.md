# TileLang IR authoring rules

**This guide requires each kernel to expose the facts needed for efficient compilation and to verify that lowering preserves its intended execution, before performance is used to judge a fixed algorithm.**

It complements [kernel optimization](kernel-optimization.md), which governs
algorithm selection and enclosing performance. [Portable kernels](../kernels.md)
define the backend boundary; [performance](../performance.md) defines evidence.

## Scope of the guarantee

No source style, compiler pass sequence, or finite checklist guarantees the
fastest equivalent implementation of an arbitrary algorithm. Native instruction
selection, register allocation, cache behavior, scheduling, and their interaction
with workload geometry remain relevant after TileLang lowers the program.

The enforceable guarantee is narrower and useful: **a qualified kernel has an
explicit execution contract, satisfies named structural properties in the
generated program, and has no unexplained violation of its intended schedule.**
Each qualification states its domain and compiler/target identity. A property
that has not been checked is unknown, not implicitly satisfied.

For a fixed algorithm, hold constant the required outputs, state transitions,
valid inputs, observable rounding boundaries, and allowed numerical error.
State separately whether reduction order, reassociation, intermediate
materialization, and redundant computation may change. Mathematical equality
over real numbers does not establish equivalence under the numerical contract.

Schedules, layouts, tiling, binding time, and storage placement remain choices
unless the contract fixes them. Their tradeoffs mean there may be several best
implementations for different shapes, occupancy distributions, or capabilities.

Use three levels of evidence:

| Level | What can be established | What it does not establish |
|---|---|---|
| Contract proof | An assumption, specialization, ownership rule, or transformation is valid on a declared domain | That a compiler exploits it |
| Lowering check | A specific guard, conversion, publication, dynamic fragment access, or other unwanted structure is absent; intended primitives are present | Final register residency, physical memory traffic, or minimum latency |
| Target measurement | A complete qualified implementation is faster under recorded conditions | Global optimality or performance on an unmeasured regime |

Algorithm comparisons must identify these levels. A slow implementation with an
unexplained lowering defect is not evidence that its algorithm is intrinsically
slow. Conversely, an algorithm whose unavoidable resource demand cannot meet
the target does not deserve unlimited tuning to make its compilation perfect.

## High-level principles

1. **Make facts available at the earliest valid stage.** Geometry, ranges,
   divisibility, ownership, and effects must reach the passes that need them.
   A host-side invariant hidden in a device buffer is invisible until expressed.
2. **Preserve useful structure.** Use typed regions, fragments, matrix operations,
   reductions, and explicit loops. Do not obscure them with premature flattening,
   opaque calls, generated source, or unnecessary mutable state.
3. **Pay dynamic costs only where the problem is dynamic.** Keep physical tile
   geometry fixed where possible; represent runtime occupancy and tails explicitly.
   A bounded runtime value need not become a compile-time constant.
4. **Choose ownership, location, and lifetime together.** Reducing instruction
   count while increasing communication or live state may lose. Keep reuse at
   the smallest scope shared by its consumers.
5. **Inspect the generated execution, not the spelling.** An allocation named
   fragment, an annotated vector loop, or a pipelined loop is an intention.
   Qualification checks its realization.
6. **Retain semantics while pursuing speed.** Preserve a promising algorithm
   while repairing numerical weaknesses. Include the repair's cost. Invalid
   memory access, races, and undefined state are not numerical tradeoffs.
7. **Turn established failures into structural regression checks.** A known
   compilation pathology should not require rediscovery through full-model runs.

## The end-to-end compilation boundary

```text
formula, precision, effects, invocation geometry
                    ↓
Python construction and static specialization
                    ↓
typed TileLang IR, buffers, regions, loops, assumptions
                    ↓
semantic verification and simplification
                    ↓
pipeline planning, layout inference, tile/reducer lowering
                    ↓
memory guards, allocation planning, address flattening
                    ↓
vectorization, storage reuse, unrolling, synchronization
                    ↓
host/device separation and target source generation
                    ↓
native compilation, pipeline creation, executable binding
                    ↓
ordered device execution and completion
```

This is a conceptual flow, not one universal pass order. Backends own their
ordered pipelines. In the inspected Metal pipeline, layout and tile lowering
precede safe-memory legalization; integer narrowing precedes buffer flattening
and index-width legalization; vectorization and storage rewriting precede the
main unrolling step; shared allocation merging precedes final shared-memory
synchronization. CUDA also performs an earlier unrolling step. Rules that depend
on pass order must be checked against the selected pipeline.

A later pass cannot be presumed to undo a conservative decision made earlier.
Follow a missing fact to its first relevant consumer, then inspect the final
device program for consequences that remain.

## Pattern conventions

Examples below are body fragments inside an authored `@T.macro` or
`@T.prim_func`, not standalone kernels. Uppercase dimensions denote construction-
time values unless stated otherwise. Buffers, thread ownership, and enclosing
launches must satisfy the stated preconditions. Only portable `tilelang.language`
constructs belong in Magnitude kernels. Backend diagnostics do not authorize
backend-specific dialects, compiler switches, or native calls in Ops.

Every pattern has a precondition, an intended compiler effect, and a check.
It is not a promise that its preferred spelling wins for every target.

## 1. Binding time and scalar state

### Separate construction-time choices from device decisions

Use Python values to select genuinely static schedule structure. Keep tensor
contents and per-invocation data dynamic unless independently known before
dispatch.

```python
# FULL_SEQUENCE is an established construction-time fact.
if FULL_SEQUENCE:
    first = 0
    stop = ROWS
else:
    first = offsets[sequence]
    stop = offsets[sequence + 1]
```

The first case should construct no offset-dependent device path. A buffer that
happens to contain `(0, ROWS)` is not equivalent compiler input. A static buffer
shape also does not make its contents static.

Static choices belong in executable identity. Use a small set of useful classes,
such as full and general, rather than compiling every changing token count,
expert count, or position. Specialization must not introduce a device-to-host
readback merely to discover a fact that can remain dynamic efficiently.

**Check:** selected variant, emitted branch/loads, cache separation, reuse, and
compilation outside warm invocation. Specialization trades startup/cache space
for simpler execution; it is not free.

### Use immutable expressions for facts and mutable storage for state

```python
# An invariant expression, not a loop-carried variable.
base = block * TILE

# An explicitly initialized per-thread recurrence.
total = T.alloc_var("float32", init=0.0)
for k in T.serial(K):
    total = total + values[k]
```

The eager frontend distinguishes immutable bindings from `local.var` stores.
Rebinding an ordinary name is not a reliable substitute for device mutation.
Conversely, using mutable buffers for every address expression can obscure
substitution, invariance, and bounds reasoning.

Initialize state explicitly; an allocation is not a portable initialization
guarantee. Address expressions should remain immutable when they do not change.
Do not infer physical registers merely from per-thread storage scope.

**Check:** frontend IR contains the intended binds, initialization, and carried
dependencies; immutable-rebinding warnings are understood and removed where they
reflect unintended state construction.

## 2. Domains, assumptions, and address arithmetic

### Express ranges in the coordinate being accessed

A weak representation hides the valid absolute row behind an offset:

```python
for local_row in T.serial(stop - first):
    output[first + local_row] = source[first + local_row]
```

When the producer guarantees `0 <= first <= stop <= ROWS`, make that contract
visible and iterate in absolute coordinates:

```python
T.assume(first >= 0)
T.assume(first <= stop)
T.assume(stop <= ROWS)
for row in T.serial(first, stop):
    output[row] = source[row]
```

These fragments assume a valid single-writer execution. They illustrate a range
representation, not a parallel copy schedule. The empty interval remains valid.

Safe-memory legalization asks the arithmetic analyzer to prove each global
access valid. A failed proof can add guarded loads with safe fallback values or
conditional stores. Repetition inside a hot loop can affect more than branch
count: surrounding address simplification and vectorization also change.

Bounds can instead be expressed through justified `min`/`max` expressions, but
clamping invalid inputs is a semantic decision. Do not silently map invalid route
IDs, offsets, or pages to valid data to make proofs easier.

**Check:** full-domain accesses lack redundant guards after legalization and in
final source. General/tail cases retain required protection. Buffer capacity
alone does not prove an offset, gather index, or count valid.

### Assumptions require an owner and a lifetime

For every assumption record:

| Obligation | Example |
|---|---|
| Who establishes it? | Admission validates host metadata; a routing producer constructs bounded IDs |
| What exact domain holds? | A route is `-1` or in `[0, ROWS)` |
| When does it hold? | After the producer completes and until that metadata is overwritten |
| Where must it be visible? | Before safe-memory, address, vectorization, or synchronization analysis |
| How is misuse detected? | Host validation, producer tests, and representative boundary tests |

```python
route = order[slot]                 # slot itself must be valid
if route >= 0:
    T.assume(route < ROWS)          # justified by the producer contract
    value = source[route, column]  # column has its own bounds obligation
```

Do not assert that all routes are nonnegative when padding uses `-1`. An
assumption about a memory load is not permission to reuse its value or constraint
after a possible write. Compiler analyses do not all have identical scope or
alias reasoning; verify the consuming pass.

Do not use negative-index syntax to implement a missing-data sentinel. Negative-
index legalization can translate provably negative coordinates relative to the
end of a dimension; some vector expressions require additional lane-wise
rewrites. Sentinel handling must explicitly prevent the access. Nonnegative
index proofs also avoid unnecessary negative-index machinery where applicable.

Some eligible host-evaluable assumptions become runtime checks. Device-data-
dependent or conditionally scoped assumptions do not universally become checked
preconditions. `T.assume` is not a general assertion or validation API.

### Preserve affine structure and prove the flattened address

Prefer a direct row/column relationship with known strides over repeated
flatten/divide/remainder reconstruction when the algorithm already knows the
coordinates. A power-of-two simplification must preserve signedness and the
valid input domain; replacing signed division with a shift is not universally
equivalent.

The relevant bound is the whole address calculation:

```text
element_offset + row × row_stride + column × column_stride
```

All intermediate products matter. The compiler can widen flattened expressions
when their ranges are unknown or exceed signed limits, even if each individual
index was declared `int32`. Earlier narrowing does not prevent later promotion.

Where wide arithmetic is genuinely required, widen before the potentially
overflowing operation:

```python
offset = T.cast(row, "int64") * stride + column
```

Casting an already overflowing product is too late. Conversely, do not force
every small local coordinate to 64 bits because a global base requires it.
Preserve small tile-relative expressions and the wide base relationship where
the normal buffer/view API can express them safely.

**Check:** explain wide arithmetic in hot loops using actual maximum offsets;
distinguish pointer width from index arithmetic; inspect both flattening and
subsequent legalization. Necessary wide indexing is not a qualification failure.

## 3. Full tiles, tails, masks, and dynamic occupancy

### Separate logical validity from physical tile shape

A physical matrix tile can remain static while its valid prefix is dynamic:

```python
valid_rows = metadata[tile, 1]
T.assume(valid_rows >= 0)
T.assume(valid_rows <= BM)
T.gemm(A_shared, B_shared, accum, valid_m=valid_rows)
```

The assumptions require a producer guarantee. The active accumulator rows must
already be initialized unless the operation clears them. Staged operands must
satisfy the primitive's access contract.

`valid_m` specifies an M-axis prefix, not arbitrary row masks, K tails, or
N tails. Its inactive output suffix is unspecified. Do not read that suffix or
treat it as zero unless independently initialized and preserved as required.
Consumers and publication must honor the same validity domain.

For an established full tile, omit runtime validity or supply its static extent.
For a dynamic full/tail split, place a uniform decision outside repeated work
when profitable; confirm that the full branch actually lowers without the
partial machinery. Branch duplication can increase code size, so it is a
schedule decision rather than an unconditional recommendation.

### Keep the interior simple and make the exceptional work explicit

```python
# block is uniform across this threadgroup.
if block < N // BLOCK:
    for i in T.Parallel(BLOCK):
        tile[i] = source[block * BLOCK + i]
else:
    for i in T.Parallel(BLOCK):
        index = block * BLOCK + i
        tile[i] = T.if_then_else(index < N, source[index], 0.0)
```

This assumes a `ceildiv(N, BLOCK)` grid, nonnegative indices, and a consumer
whose neutral padding value is zero. An omitted empty launch or an explicit
empty case is required for `N == 0` if the target does not accept zero-sized
grids. Divisibility known at construction may make the split unnecessary.

Choose padding from the operation: zero for sums and dot-product contributions,
negative infinity for a maximum over finite values, and the appropriate identity
for other reductions. Masking a softmax needs particular care for an entirely
masked row: infinity subtraction and normalization can otherwise produce NaNs.
Padding must be neutral under the declared exceptional-value semantics too:
zero multiplied by NaN or infinity is not necessarily an ignorable contribution.

**Check:** guards occur where validity is uncertain, not throughout a proven
interior; each lane initializes every shared/fragment element the consumer reads;
the final partial reduction tile is handled independently of output tails.

### Use conditional evaluation to protect invalid accesses

```python
value = T.if_then_else(index < N, source[index], 0.0)
```

Do not replace a guarded load with a value-selection expression that requires
both operands to be valid. A multiply by zero after an invalid load does not make
the access safe. Symbolic `and`/`or` describes IR boolean expressions; it must not
be relied upon as Python short-circuit protection for memory operations.

**Check:** inspect predication at the memory operation itself. Safety legalization
is not a blanket proof that shared/local accesses or opaque pointer accesses are
safe, nor that the fallback value implements the formula's masking semantics.

## 4. Loops, fragment indexing, and invariant work

### Unroll instruction coordinates, preserve long algorithm loops

```python
partial = T.alloc_local((LANES_PER_THREAD,), "float32")
for j in T.unroll(LANES_PER_THREAD, explicit=True):
    partial[j] = 0.0

for block_k in T.serial(K_BLOCKS):
    for j in T.unroll(LANES_PER_THREAD, explicit=True):
        partial[j] = partial[j] + contribution[block_k, j]
```

The inner extent is small and construction-time constant; each thread owns its
partial array. This preserves its scalar recurrence order while exposing fixed
array coordinates. The outer loop need not expand.

Explicit expansion makes IR indices constant before downstream native
compilation. An unroll annotation may instead leave a loop plus a pragma.
Both differ from hoping the native compiler recognizes a profitable loop.

On Metal, short-loop defaults use cost/extent/nesting thresholds. Remaining
constant loops longer than four iterations can receive an explicit
unroll-disable directive. The backend's matrix lowering explicitly expands
instruction-fragment coordinates independently of the outer reduction loop.
These are backend behaviors to verify, not thresholds to hard-code into portable
kernel policy.

**Check:** local/fragment indexing is static where scalar replacement requires it;
inspect loop annotations and final source, including disable directives. Bound
code growth and live values. Unrolling a long history loop can harm compilation,
instruction-cache behavior, or occupancy; register residency still needs native
evidence when it is the claimed mechanism.

### Hoist facts and repeated preparation only across valid dependencies

```python
# Safe only if these inputs remain invariant across the reduction.
route = order[tile]
for k in T.serial(K_BLOCKS):
    consume_tile(route, k)
```

Here `consume_tile` represents a statically authored macro with the stated
dependency, not an opaque runtime callback. Loading once removes repeated
preparation only if aliasing, writes, and required observation order permit it.
The same principle applies to scale lookup, packed metadata, address bases, and
schedule-kind decisions.

Loop unswitching can move an invariant branch outside a loop. It must prove the
condition independent of the loop variable and of buffers written in the loop;
opaque pointer effects can prevent that proof. An explicitly static schedule
choice is stronger than a runtime condition that merely happens not to change.

**Check:** expensive invariant work and branch decisions are outside the intended
loop in generated code. Hoisting can lengthen live ranges; do not trade a cheap
recomputation for excessive retained state without accounting for it.

## 5. Layout, vectorization, and copy regions

### State the lane-to-element relationship

Parallel work needs an ownership rule: which logical values each thread owns,
which are replicated, and how values move to a consumer with different ownership.
Use `T.Parallel`, fragments, and supported layout annotations to express it.

For a multidimensional parallel nest, a supplied loop layout describes the whole
nest and belongs to the outermost loop; adding unrelated per-dimension layout
annotations does not express that relationship. Layout validity is necessary but
does not establish good coalescing, balanced work, or minimal conversion.

The default free-layout scoring in the inspected compiler emphasizes fragment
register counts. Other scoring modes exist, but a heuristic is not exhaustive
schedule search or a prediction of whole-operation latency.

**Check:** inferred thread/value mapping, active threads, replication, collective
groups, and conversions at producer/consumer boundaries. Separate requested
memory operations from actual cache/DRAM transactions. Bank-conflict and occupancy
claims require an appropriate target model or measurements.

### Preserve contiguous vector opportunities

```python
# One thread owns V contiguous elements. base and the allocation/view satisfy
# the vector alignment contract; the entire group is inside the valid domain.
for j in T.vectorized(V):
    destination[base + j] = source[base + j]
```

Known stride one, base divisibility, extents, and layout support vectorization.
Aligned allocation does not imply every row or subview is aligned. Include
element size, row pitch, and view offsets in the proof.

The vectorizer combines memory, local-access, call, and control-flow constraints.
A multi-statement body may impose a more conservative combined width than a
simple copy. A lane-varying condition or incompatible operation can scalarize the
loop. Splitting a loop can recover width but introduce more traversal, storage,
or lost fusion; check the complete cost.

Matrix fragments have lane-local ownership restrictions. Metal legalizes wide
matrix-fragment vector loops to the elements actually owned by a lane. Do not
force a wider vector across that boundary.

**Check:** actual vector loads/stores and their addresses, not just the loop
annotation. A legal scalar tail is expected. Scalarization of a proven contiguous
interior needs an explanation or a lowering fix.

### Make copy extents unambiguous

```python
T.copy(source[row_base:row_base + BM, col_base:col_base + BK], tile)
```

This assumes a `BM × BK` destination and the desired boundary semantics. Prefer
explicit regions when shorthand leaves the transferred extent unclear. Two
scalar accesses may lower to one scalar assignment; a buffer and a head address
can describe a tile; mismatched extents can trigger normalization or different
iteration geometry. Copy is not a general broadcasting contract.

A copy can also perform a dtype conversion or introduce source-zero-fill and
destination guards. Ordinary zero-fill is not automatically the right reduction
identity. `T.copy` is not a promise of an asynchronous transfer or specialized
copy instruction on every target.

**Check:** inferred source/destination regions, actual iteration extents, casts,
padding values, access widths, and copy mechanism after tile lowering.

### Keep physical views distinct from logical matrix axes

For a transposed operand, its physical row/column offsets and strides still
describe its stored buffer. The transpose changes how those coordinates map to
logical M/N/K; it does not transpose the physical buffer declaration. Preserve
the physical region until the matrix primitive applies that mapping.

Use views and layouts that describe actual storage, including padding and
packing. Do not fabricate a smaller dtype, alias, or stride to make inference
select a desired instruction. A transpose, reshape, or swizzle may be a view,
register permutation, shared exchange, or full copy depending on its producer
and consumer; inspect the realized movement rather than assuming it is free.

**Check:** logical contraction dimensions match physical regions, transpose
flags, offsets, and strides; layout conversions are intentional; alignment holds
for the actual view, not only its base allocation.

## 6. Storage, matrix operations, and reductions

### Allocate by communication requirement

| Storage | Use when | Obligation |
|---|---|---|
| Immutable scalar expression | A value is a fact or pure address expression | Preserve dependencies without unnecessary mutation |
| Mutable scalar or local array | A thread owns changing state | Explicit initialization and bounded live state |
| Fragment | A logical tile is distributed across threads or consumed by tile operations | Compatible ownership, indexing, and primitive lowering |
| Shared memory | Threads genuinely exchange or reuse values | Complete initialization, valid participation, and synchronization |
| Global intermediate | A required inter-kernel boundary or justified materialization exists | Allocation, traffic, effects, and lifetime included in the operation |

Local and fragment allocations are not guarantees against spilling. Shared
allocations may merge; allocation scopes and last uses affect reuse and final
physical footprint. Lowering can introduce temporaries absent from source.

Keep transient operands short-lived and persistent state at the appropriate
scope. Excessive fusion or large tiles can retain many values simultaneously.
There is no universal rule that fewer allocations, more shared reuse, or more
occupancy is faster.

### Retain the accumulator across the complete contraction

```python
accum = T.alloc_fragment((BM, BN), "float32")
T.clear(accum)
for block_k in T.serial(T.ceildiv(K, BK)):
    # Stage this K tile, including required neutral padding.
    stage_operands(block_k, A_shared, B_shared)
    T.gemm(A_shared, B_shared, accum)
T.copy(accum, output_tile)
```

`stage_operands` is a macro with complete access and dependency semantics.
Supported dtypes, shapes, layout, and target capability are preconditions.
`T.gemm` accumulates here; clearing inside the loop would discard previous
contributions. Conversely, `clear_accum=True` is appropriate when each GEMM is
an independent result and no prior accumulator value is required.

A shared destination may lower to fragment load/compute/store around each GEMM,
while a compatible fragment destination can retain the accumulator. Publishing
an intermediate each iteration can therefore impose traffic and synchronization
even with the same arithmetic. Publish when a consumer actually needs it.

**Check:** intended matrix instruction family, operand and accumulator dtypes,
constant instruction coordinates, operand reuse, and no unintended repeated
accumulator publication. Confirm K/N tails separately from M-axis validity.

### Expose reductions as reductions

```python
# values is a (ROWS_PER_TILE, WIDTH) fragment with complete valid/neutral data.
sums = T.alloc_fragment((ROWS_PER_TILE,), "float32")
T.reduce_sum(values, sums, dim=1, clear=True)
```

Use a permitted reduction order and explicit seed/clear semantics. If the
contract fixes a sequential floating-point order, replacing it with a parallel
tree is a numerical algorithm change that needs separate qualification.

Shared-to-shared reductions can introduce input/output fragment copies; fragment
inputs/outputs can avoid those boundaries when ownership and consumers permit.
Cross-subgroup reductions need wider communication and participation than a
subgroup-local reduction. On the inspected Metal lowering, reductions across
SIMD groups require full threadgroup participation for their barriers.

Where reducer objects are used, keep a complete init/update/finalize epoch and
treat the reducer as opaque until finalization. Do not read its partial storage
as an ordinary tensor or use per-lane stores as an implicit reduction. Batching
several reductions can amortize communication but changes live-state demand.

**Check:** communication width, number of collectives, intermediate copies,
neutral elements, empty cases, clear/accumulate semantics, and permitted order.

## 7. Synchronization, dependencies, and pipelining

### Separate participation masks from data masks

```python
for i in T.Parallel(BLOCK):
    shared[i] = T.if_then_else(base + i < N, source[base + i], 0.0)
# All threads required by the following collective continue to participate.
consume_shared(shared)
```

The consumer is an authored macro or primitive whose accesses are visible to
TileLang. Let the normal dependency analysis establish required synchronization;
use an explicit portable synchronization operation only when its scope and
dependency are justified. Do not place a threadgroup collective inside a
lane-varying validity branch and assume inactive lanes are irrelevant.

Classify each condition: construction-time; block-uniform runtime;
subgroup-uniform runtime; or lane-varying. Runtime dependence does not itself
imply divergence. A subgroup-uniform branch is still insufficient for a
threadgroup-wide collective.

Barrier placement depends on producer/consumer conflicts, overwrite hazards,
loop-carried accesses, aliases, and storage reuse. Inspect it after final storage
merging as well as before. A barrier outside a branch may fail to separate two
conflicting operations inside it; hoisting is not automatically a repair.

**Check:** every required participant reaches each collective; every consumer
follows its producer; staging is not overwritten before use; warnings about
unsafe synchronization or uninitialized memory are resolved. A lower barrier
count is not evidence of a correct or faster kernel by itself.

### Pipeline only genuinely independent work

```python
for block_k in T.Pipelined(K_BLOCKS, num_stages=STAGES):
    stage_operands(block_k, A_shared, B_shared)
    T.gemm(A_shared, B_shared, accum)
```

This requests a schedule, not a guarantee of overlap. `STAGES` is a justified
construction-time choice. Compiler-inferred pipelining with zero stages is
disabled. A real recurrence dependency cannot be removed by requesting more
stages.

Pipeline injection can create buffer versions, stage predicates, prologues,
epilogues, and extra waits. Pure address bindings may be replayable; reads of
buffers written within the pipeline carry real dependencies. Stage counts and
manual schedules must respect them.

**Check:** final schedule, buffer versions and footprint, producer/consumer order,
barriers/waits, and the target mechanism that could overlap work. If overlap is
claimed, measure it. Extra buffering that only increases storage is not a win.

### Do not invent inter-block ordering

Independent threadgroups have no general ordering within a launch. A spin loop
waiting for another block can deadlock when the producer is not scheduled.
Atomics provide only their specified memory/order semantics, not a universal
grid barrier. Use supported cooperative mechanisms or a real ordered kernel
boundary when the algorithm requires global completion.

**Check:** atomics preserve dtype, address space, return-value semantics, and
required ordering; producer/consumer progress does not depend on unscheduled
blocks. Atomic contention remains a performance cost even with correct ordering.

## 8. Aliasing, precision, and arithmetic lowering

### Preserve alias information and declare real effects

Read-only and non-aliasing information can enable reuse and eliminate redundant
loads. False non-aliasing information invalidates those optimizations. Separate
buffers may refer to overlapping views of one allocation; names and distinct
parameters do not prove disjointness.

Use the actual supported alias contract. In the current public API, despite its
name, `T.annotate_restrict_buffers(x, y)` marks those parameters as non-restrict;
verify this meaning when using it. Do not add annotations speculatively. Keep
opaque pointer escapes out of numerical kernels: they weaken effect reasoning
and can prevent hoisting or safe storage reuse.

**Check:** actual backing ranges, declared effects, generated qualifiers, and
whether an in-place transformation preserves the formula's state/checkpoint
semantics. Read-only source annotations do not prove disjoint physical storage.

### Specify every precision boundary

```python
# If the formula requires a low-precision intermediate before a residual:
rounded = T.cast(accum[i, j], OUTPUT_DTYPE)
result[i, j] = T.cast(rounded, "float32") + residual[i, j]
```

Storage dtype, operand reconstruction dtype, multiplication dtype, accumulation
dtype, and publication dtype are separate choices. Moving a cast across a sum,
activation, or recurrence may change the contract. Copies may cast too.

Intrinsic lowering can form fused multiply-add operations. Native math modes
can reassociate or select approximate operations. The inspected Metal source-
compilation runtime enables fast math; explicit fast-math intrinsics also have
target lowering. A source expression alone therefore does not prove a separate
rounding after every arithmetic operator or strict IEEE exceptional behavior.

**Check:** required casts survive where observable; arithmetic/storage types,
contraction and math mode match the tested contract. Exercise cancellation,
extreme magnitudes, long reductions/recurrences, and special values when in
domain. Fix insufficient compiler control in TileLang rather than adding a
backend switch to Ops.

### Keep packed representations exact until a justified conversion

Use integer signedness and bit widths matching the encoding. Prove shifts,
masks, metadata strides, packing divisibility, and tail accesses. Bit
reinterpretation is different from numerical conversion, and native language
integer promotions must not change the intended width.

Decode coefficients and codes at the scope where consumers reuse them, retain
required precision, and stage only the needed tile. Whole-weight expansion or a
different quantization interpretation changes the cost/contract being compared.

**Check:** no duplicate metadata reconstruction in the hot loop, no hidden full
expansion, correct sub-byte view alignment, exact source interpretation, and
all conversion costs inside the measured operation.

### Repair accuracy within a promising fast approach

Classify the error before changing the schedule: conversion, accumulation,
cancellation, conditioning, normalization, approximation, or recurrence drift.
Explore selective widening, stable scaling, compensation, or mathematically
justified correction within the approach. Include additional products, state,
traffic, and synchronization when evaluating the repaired version.

A failed first numerical check does not disqualify a strong resource hypothesis.
An approximate fast prototype is not yet a valid same-contract speedup. Reject
the approach when its required repair removes its advantage or cannot meet the
contract, rather than reverting merely because the first attempt failed.

## 9. Native compilation and warm execution

### Preserve the program boundary

One `T.Kernel` is a device execution region. Several ordered regions can share a
native host entrypoint. That removes repeated host orchestration; it does not
fuse device work or remove intermediate traffic by itself.

Use the ordinary Ops/TileLang program path for binding, allocations, submission,
and completion. Stable resource binding is distinct from compile-time numerical
specialization: fixing a pointer does not make all values it addresses constants.
Cache keys must distinguish the actual program, static facts, target, compiler,
and relevant execution configuration.

**Check:** no tracing, compilation, pipeline-state creation, repeated static
argument preparation, or unplanned allocation in warm invocation. Verify cache
invalidation after source/compiler changes; a cache hit can conceal the absence
of a requested lowering trace. Do not clear unrelated caches as a routine audit.

### Treat target source as an intermediate artifact

The native compiler still chooses instructions, registers, spills, and scheduling.
Shader/pipeline creation can impose limits beyond the target's nominal maximum
threads or storage. Validate the compiled pipeline's applicable limits and retain
runtime/SDK/driver provenance.

The strongest useful source-level claims concern structure: an offset load is
absent; a vector access is present; an accumulator is not explicitly published
each iteration. Stronger claims—no spills, ideal occupancy, measured cache reuse,
or shortest execution—need native reports, counters, or controlled evidence.

## 10. Qualification and efficient enforcement

### Declare expected properties before inspecting results

For each kernel family, write a compact execution contract:

```text
Domain: supported shapes, strides, valid ranges, dtypes, aliasing and effects
Variants: finite specialization predicates and their owner
Work: algorithm, permitted order, expected primitive families
Ownership: threads per output and communication groups
Storage: persistent state, staging, intermediates and publication boundaries
Interior: expected predicates, index widths and vector accesses
Exceptions: tails, empty work, packed/sentinel data and numerical stress cases
Evidence: lowering checks plus the measurements needed for unresolved claims
```

This is a property specification, not a frozen shader string. Some kernels
legitimately require dynamic loops, scalar gathers, wide indices, atomics, or
global intermediates. Explain these instead of forcing every kernel through a
universal “zero guards, zero barriers” rule.

### Inspect the real pipeline at meaningful checkpoints

| Checkpoint | Inspect |
|---|---|
| Constructed IR | Static versus runtime values; mutable state; regions; effects; assumptions |
| Layout and tile/reducer lowering | Ownership, primitive family, copies, replication, accumulator representation |
| Safe-memory legalization | Which accesses acquire guards and what fact failed to prove them valid |
| Flattening and index legalization | Whole-expression bounds, introduced wide arithmetic, view offsets |
| Vectorization and unrolling | Actual widths, scalarization causes, fragment coordinates, code growth |
| Storage and synchronization | Final allocations, liveness/reuse, barriers, participation and loop hazards |
| Generated device and host code | Surviving unwanted work, casts, native operations and submission structure |
| Native executable | Applicable resource limits; register/spill information when available; numerical and timing behavior |

Use TileLang's shared pass instrumentation/LowerTrace around the normal selected
operation compilation. It can retain before/after IR and generated code from
the actual backend sequence. Source-only lowering avoids model execution and
native compilation, but still costs lowering time. Existing generated artifacts
are the first place to inspect; a missing pass snapshot justifies one affected
compilation, not a complete model rebuild.

Prefer IR-level structural checks over regex counts. Count accesses inside their
loop and control-flow context, distinguish intended tails from interiors, and
normalize irrelevant names. A tiny isolated pattern is useful for locating a
compiler defect, but the production composition must also preserve the property.

### Make regression checks conditional on the contract

| Reject qualification | Review rather than automatically reject |
|---|---|
| False specialization or unsupported assumption | Necessary dynamic input bounds |
| Invalid memory access, uninitialized reads, unsafe collective participation | Explained tail guards and padding work |
| Loss of a required observable rounding/state boundary | Necessary wide addresses or precision conversions |
| Missing native primitive required by the schedule | Legitimate scalar or subgroup work |
| Reappearance of a proven redundant interior guard/publication | A different layout or storage tradeoff with evidence |
| A benchmark that omits outputs, conversion, staging, or completion | Resource expansion justified by a faster complete operation |

Representative conformance cases include full tiles, one-element tails, zero
logical work where supported, packed offsets/sentinels, nonzero initial state,
allowed strides/views, boundary address ranges, and relevant numerical extremes.
Select cases that exercise different proof obligations rather than constructing
an exhaustive cross-product of dimensions.

After structural checks, test the smallest unresolved performance hypothesis
with compiled variants and identical inputs. Check the complete output and final
state. Account for native GPU time separately from host submission/completion;
do not add overlapping kernel times as if they were serial elapsed time.

Only a qualified winner proceeds to enclosing model measurements. Report its
absolute saved time and fraction of the parent, using the same token/context
window and matching controls. A source-level improvement or isolated speedup
does not by itself establish a whole-model gain.

## Acceptance rule

An implementation is ready for algorithm comparison when its valid domain and
precision/effects are explicit, its generated execution satisfies the declared
structural properties, necessary exceptions are explained, and the isolated
measurement includes all required work. It is ready for production acceptance
when the same properties hold in composition and the relevant enclosing checks
pass.

The maintained guarantee is **intentional, verified compilation on a declared
domain**. A claim of the most efficient implementation additionally needs a
specified search space or a proved lower bound that the implementation attains;
without that evidence, use “best measured qualified implementation,” not
“guaranteed optimal.”
