# Portable kernels

**A kernel is an authored portable TileLang schedule for one selected operation
or fused region. Magnitensor chooses the strategy; TileLang owns its compiler
meaning and target-specific realization.**

[Kernel optimization](development/kernel-optimization.md) defines how schedules
are pushed to their limits without weakening this boundary.

## Boundary

```text
semantic operation or region
        │ Magnitensor chooses algorithm, strategy and static parameters
        ▼
portable T.Kernel emission
        │ Magnitensor assembles the maximal compilation unit
        ▼
TileLang module
        │ TileLang lowers the whole unit for the target
        ▼
native executable
```

Magnitensor kernels use only the public TileLang language. They do not call a
target function, pass target-specific compiler flags, inspect generated IR or
branch on a backend name. TileLang and TVM values remain inside the kernel and
runtime packages and never cross Magnitensor's public tensor API.

## Capability-specialized portability

Portable does not mean one schedule for every device. Magnitensor maintains
several strategies and selects by backend-neutral capabilities reported through
TileLang.

| Capability fact | Schedule consequence |
|---|---|
| Subgroup geometry and exchange | Reduction ownership, vector width and cooperative traversal |
| Supported matrix shapes and dtypes | Fragment geometry and accumulation strategy |
| Fast memory scopes and capacity | Tile residency and pipeline depth |
| Asynchronous movement and barriers | Load/compute overlap and stage count |
| Atomics and synchronization | Grouping, reduction and routing algorithms |
| Alignment and launch limits | Vectorization, workgroup shape and tail handling |

Strategies are named by mechanisms such as fragment-tiled,
subgroup-streaming, split-reduction and persistent-weight. A strategy may be
ideal for one vendor today without containing that vendor's identity.

If a needed fact is absent, TileLang's public target capability is incomplete.
If a portable language operation cannot reach the required native mechanism,
TileLang's primitive or backend lowering is incomplete. Neither defect permits a
side channel in Magnitensor.

## Construction and compilation

A selected region contributes an ordinary Python emitter that calls statically
authored TileLang schedules. The compiler creates the ordered parameter plan and
binds region operands to graph values. The TileLang adapter passes that dynamic
ABI and a Python body callback to its public eager builder; the callback invokes every
selected emitter in dependency order while TileLang's eager builder is active.

This is programmatic IR construction, not source generation. Kernel bodies are
real Python checked and refactored with the rest of the codebase. Magnitensor does
not concatenate strings, invoke `exec`, synthesize Python signatures, or reach
through TileLang to TIR builders.

A true fusion has one `T.Kernel` region and no globally materialized interior.
An algorithm with required device-wide phases may use several ordered
`T.Kernel` regions in one authored schedule function. Magnitensor never recovers,
clones, splices or structurally compares existing PrimFuncs through private TVM
IR.

Already-finalized PrimFuncs are never recovered, cloned or combined. That would
make a region the compilation boundary prematurely and require private compiler
IR. Composition occurs from the selected lowerings through TileLang's public
programmatic `PrimFunc` and `IRModule` construction. Graph regions, selection and
operand planning remain entirely Magnitensor concepts.

## Native entrypoint

One compiled multi-function module is one TileLang program, not one device
kernel. TileLang may lower its private schedules' `T.Kernel` regions to several device launches, but
its execution adapter encodes them through one native host entrypoint and one
ordered stream or command-buffer context. A Python loop over device kernels does
not satisfy this contract.

Magnitensor supplies the unit and identifies immutable and dynamic operands.
TileLang owns ABI validation, generic partial binding and the backend-specific
realization of the pre-bound entrypoint. If an adapter cannot provide native
multi-launch or partial binding, the deficiency is a TileLang runtime gap rather
than permission to split the graph into Python calls.

## Numerical invariants

| Invariant | Reason |
|---|---|
| Accumulation and rounding boundaries are explicit | Real-number equivalence does not establish finite-precision equivalence |
| Encoded bits are reinterpreted before explicit decode arithmetic | Storage interpretation cannot be left to compiler accident |
| Padding and peers never enter a row's reduction | Batching and tiling cannot alter the mathematical result |
| State reads and writes match the operation's declared version | Fusion cannot reorder or expose tentative state |
| Unsupported tails are handled or reject applicability | A fast interior kernel is not a complete schedule |

Fusion eliminates storage and launches, not declared publication precision.
When a region covers distinct normalization, activation, multiplication, or
branch-add operations, their logical dtype boundaries remain observable even if
the values never leave registers. Arithmetic internal to a single operation is
governed by that operation's numerical contract, not invented intermediate nodes.

Representation decoding happens where its values are consumed. A quantized
matrix schedule decodes only the tile it reuses; it does not require a resident
dequantized copy unless that representation is selected and charged explicitly.

Small-row projection is selected by shape rather than by the enclosing prefill
or decode label. An exact-row direct schedule avoids executing masked matrix
instruction rows. Consecutive attention or recurrent projections that consume
the same activation form one lowering region and one `T.Kernel`; their separate
semantic outputs remain visible to downstream operations without requiring
separate launches.

Long-context decode attention groups every query head sharing one KV head into
the same partition workgroup. Subgroups scan disjoint history spans, retain
online-softmax state in registers, and reduce each query/key dot within the
subgroup without a workgroup barrier per history token. One bounded shared merge
combines subgroup results, and a second ordered kernel merges partitions.
Prefill uses streaming matrix products with query-row reuse. Scores remain inside
the streaming tile; bounded partial outputs and statistics may be used to combine
history partitions without materializing the score matrix.

Packed projections consume whole representation packets. Decode assigns packet
GEMVs to subgroups; prefill decodes directly into shared matrix tiles consumed by
`T.gemm`. Decoded weight arithmetic and logical publication dtype are separate:
an available FP32 matrix path preserves decoded coefficient precision, while
activation publication and gate/up rounding follow the logical tensor dtype.
Only the consumed tile is expanded, and selection charges its actual arithmetic
storage. Real-number reassociation of quantization coefficients is not assumed
to preserve the qualified whole-model precision contract.
Adjacent projections over one activation share a physical grid. Routed
decode combines selected and shared expert gate/up work and their down work into
two kernels. Routed prefill groups routes, combines routed/shared tiled gate/up,
combines their tiled down projections, and performs one final unpermute/shared
reduction.

Long recurrent prefill may prepare independent chunk-local delta systems and
carry the recurrent boundary through matrix contractions. Decode and small spans
retain register recurrence. Chunk boundaries are numerical scheduling choices,
not prefix-checkpoint boundaries. Both strategies expose the same output and
final state, preserve empty sequences, and respect reset decays without dividing
by cumulative decay products. Chunk scratch belongs in the ordinary workspace
ledger, and extra device stages remain inside the native program entry.

## Qualification

A kernel candidate is accepted only with independent numerical validation,
generated-code inspection and enclosing-region measurement. Emitting valid
source or improving an isolated instruction loop is insufficient; the candidate
must improve the selected region with its materializations and dispatches
included.

Target-specific tests belong with TileLang when they validate a capability,
primitive or lowering. Magnitensor tests validate that portable candidate
applicability and selection produce correct, performant execution across the
capability classes it supports.
