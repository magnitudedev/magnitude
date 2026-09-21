# Seismic language

Seismic source describes logical values, ownership, ordering, and semantic operations. It does not
describe physical blocks, storage classes, launch geometry, participant groups, staging, or
pipelines. Those are compiler and backend responsibilities. The normative specification is
`specs/26-09-19/seismic-logical-language-and-capabilities-spec.md`.

## Declarations

Every callable declaration is a function with a body:

```text
fn name[SHAPES](parameters) -> result where predicates:
    body

fn helper[SHAPES](parameters) -> result for metal
    requires metal.subgroup, metal.matrix
    where predicates:
    body

lower name[SHAPES](parameters) -> result for metal
    requires metal.matrix
    where predicates:
    body

native name for metal from "native/name.metal":
    threadgroups (ceil_div(N, 256), 1, 1)
    threads_per_threadgroup (256, 1, 1)
```

Canonical header order is `for`, `requires`, `where`.

- A `fn` without `for` is portable.
- A `fn for B` is a helper callable only by definitions for backend `B`.
- A `lower ... for B` is an alternative backend implementation of a portable function family.
- A `native name for B from "asset"` attaches one explicitly selected, top-level native
  implementation to an existing portable function. Its signature, effects, shapes, and ABI are
  derived from that function rather than repeated.
- Portable bodies and applicable lowerings remain equal candidates at each static call.
- A backend definition declares every capability namespace it uses. Requirements propagate through
  backend-helper calls, and unused requirements are errors.
- Source files end in `.seismic`; file names do not establish a backend or namespace.

There are no bodyless function declarations, target-neutral lowerings, export markers, lowering promises,
or `= portable` aliases.

## Values and ownership

```text
tensor[M, N] f32        # owned logical tensor
&tensor[M, N] f32       # shared borrow
&mut tensor[M, N] f32   # exclusive mutable borrow
index[N]                # integer with 0 <= value < N
range[N]                # half-open range within 0..N
```

An owned tensor moves when passed, assigned, or returned. Using its previous binding after the move
is an error. Shared borrows may overlap. An exclusive borrow cannot overlap another live borrow of
the same storage. Slicing produces a borrow; a mutable slice requires exclusive access and proved
disjointness from other live borrows.

Copies are explicit:

```text
let snapshot = to_owned(shared_slice)
let duplicate = clone(owned_value)
```

`to_owned` performs a deep copy from a borrow. `clone` duplicates an owned tensor. Implementations
may elide a physical copy only when ownership and value semantics are unchanged.

Borrowed tensors are not returned. Tensor and tuple results are owned. A backend may realize an
owned result using a hidden destination or legal storage reuse, but that ABI choice is not visible
in source.

## Bindings and mutation

```text
let value = expression
let mut state = expression
```

`let` is immutable. `let mut` permits reassignment or mutation; it does not create ownership or
write permission absent from the initializer. Ordered carried state follows from lexical scope and
use—there is no separate state declaration.

Functions either return owned results or mutate an explicit exclusive borrow:

```text
fn normalize[N](x: &tensor[N] f32) -> tensor[N] f32:
    let mut result = zeros[N](f32)
    parallel for i in 0..N:
        result[i] = x[i]
    return result

fn append[N](cache: &mut tensor[N] f32, at: index[N], value: f32):
    cache[at] = value
```

The checker proves that an owned value is fully initialized before it is read, moved, or returned.
Mutation through a borrow is permitted only through `&mut tensor` and must obey exclusivity.

## Indexing, ranges, and loops

`index[N]` and `range[N]` are semantic refinements. Arithmetic on an index produces an ordinary
integer unless it is proved or checked back into an index. Constructing a range proves or checks
`0 <= start <= end <= N`; bounds never clamp silently.

Point indexing reads or designates one element. Non-point indexing and slicing produce borrows.

```text
for i in 0..N:
    # ordered ascending visits; captured `let mut` state may be updated

parallel for i in 0..N:
    # logically independent visits
```

A `parallel for` body cannot update captured scalar or owned local state. It may write through an
exclusive tensor place when the checker proves distinct visits write distinct elements: every
loop binder must be established by a point index affine in it with a nonzero coefficient (once the
other binders on that axis are established; several binders on one axis are established together
when their coefficients form a mixed radix over the binders' ranges, as `i * C + j` with `j < C`),
or by a slice `c*v + d : c*v + d + len` with `len <= c`. Data-dependent, nonlinear, and
non-injective indices (`y[routes[i]]`, `y[i / 2]`, `y[i + j]`) are rejected. The other admitted
update is `atomic(add|max|min, place, value)`: portable, defined for f32, f16, bf16, i32 and u32
elements; `add` is the registry load/add/round/store and `max`/`min` are exact and
order-independent (a NaN operand is ignored). The reference applies atomic updates in visit
order. The compiler may group, vectorize, block, fuse, stage, pipeline, or distribute either loop
only while preserving its ordering contract.

## Calls and implementations

Calls resolve statically to a function family. For backend `B`, an occurrence considers applicable
portable bodies and applicable `lower ... for B` bodies. A backend-specific helper is considered
only when called from code for the same backend. Declaration order and file placement do not rank
candidates.

Shape predicates use `where`. A lowering may specialize shape relationships or element types only
while preserving the function contract.

A native implementation is never considered at a static call occurrence. Generated Rust exposes
the ordinary `for_device(device, precision)` and the explicit
`native_for_device(device)` in parallel. Both use the same generated `Args` and `Results`, but the
native handle permits only direct synchronous calls. Its launch expressions are closed integer
arithmetic over the attached function's shape dimensions.

The embedded source receives generated `SEISMIC_*` Metal macros. In addition to buffers, shapes,
strides, and scalar words, polymorphic element bindings and tensor ABI leaves receive canonical
registry-derived representation descriptors. Native source uses these macros rather than inferring
storage from logical extents or duplicating representation layout tables.

## Capabilities

The initial source-visible backend capability namespaces are:

```text
metal.subgroup     metal.matrix
cuda.subgroup      cuda.matrix
```

Capability calls use three-part paths, for example `metal.subgroup.simd_sum(value)` and
`metal.matrix.matmul(a, b, accumulation=f32)`. The registry assigns each member an exact typed
signature and semantic contract. Availability is the intersection of the device, toolchain, and
backend emitter; unsupported specialized candidates are removed before selection while portable
candidates remain available.

Logical matrix operations consume and return tensor values. Raw fragment members currently used by
the Metal library are a temporary migration bridge, not the durable language model.

## Reference behavior

The interpreter executes function bodies and defines deterministic reference behavior. Ordered
loops visit ascending coordinates. `parallel for` executes sequentially in the interpreter but
retains its independence requirement. Production implementations must be exact or carry numerical
evidence accepted by the caller's policy.

Source diagnostics cover ownership moves, borrow conflicts, incomplete initialization, invalid
indices and ranges, unordered mutation, capability declarations and availability, call coverage,
and numerical obligations.
