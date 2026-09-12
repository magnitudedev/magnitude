# Portable kernels

**A kernel is an authored portable TileLang schedule for one selected operation
or fused region. Magnitensor chooses the strategy; TileLang owns its compiler
meaning and target-specific realization.**

## Boundary

```text
semantic operation or region
        │ Magnitensor chooses algorithm, strategy and static parameters
        ▼
portable T.Kernel emission
        │ Magnitensor assembles the maximal compilation unit
        ▼
TileLang PrimFunc
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
`T.Kernel` regions in one authored `PrimFunc`. Magnitensor never recovers,
clones, splices or structurally compares existing PrimFuncs through private TVM
IR.

Already-finalized PrimFuncs are never recovered, cloned or combined. That would
make a region the compilation boundary prematurely and require private compiler
IR. Composition occurs before finalization from the selected lowerings. Dynamic
`PrimFunc` construction is a generic TileLang language responsibility; graph regions,
selection and operand planning remain entirely Magnitensor concepts.

## Native entrypoint

One compiled multi-kernel `PrimFunc` is one TileLang program, not one device
kernel. TileLang may lower its `T.Kernel` regions to several device launches, but
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

Representation decoding happens where its values are consumed. A quantized
matrix schedule decodes only the tile it reuses; it does not require a resident
dequantized copy unless that representation is selected and charged explicitly.

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
