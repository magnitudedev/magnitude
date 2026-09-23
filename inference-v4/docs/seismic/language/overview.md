# Language

Seismic describes the logical structure of a computation: its operations, values, dependencies, iteration, effects and publication. The compiler chooses how that structure runs on a target.

Source must contain enough structure to reach an optimal physical realization. It should expose independent work, exact contribution domains, recurrences, access patterns and necessary ordering. Authors should not have to prescribe workgroup sizes, scratch layouts, launch counts or pipeline stages to make those facts visible.

## What belongs in source

| Authored fact | Meaning |
| --- | --- |
| Scalar and tensor operations | The actual computation and its representation boundaries. |
| Bounded indices, ranges and shapes | Legal geometry, exact runtime values and bounds on them. |
| Ordered iteration and loop carries | Recurrences and required inter-iteration dependence. |
| Parallel iteration | Independence of work, subject to the checked ownership and effect rules. |
| Views and indirect indexing | Logical access geometry, including irregular access. |
| Mutable borrows, stores, atomics and checks | Observable changes, permissions, ordering and failure behavior. |
| Function bodies and authored alternatives | Reusable computation and explicitly supplied alternative structure. |
| Typed backend capabilities | Operations requiring a particular native semantic facility. |

The language must compose these forms: a helper inside a loop, a view of a computed tensor, a content-dependent branch, or a carry holding an intermediate tensor cannot require a separate escape path through the compiler.

## Freedom left to the compiler

The compiler may partition work, map it to participants, choose storage and representations within the numerical contract, stage data, share or recompute pure producers, select authored alternatives, and choose target operations that implement the actual authored contribution structure. It must preserve required reads, effects, versions, rounding and permitted outcomes.

An alternative algorithm belongs in an authored body. Equality of final answers alone does not permit the compiler to invent another algorithm. Supporting work such as indexing, communication and decoding is compiler-owned physical work and must be represented and accounted for.

Logical values do not imply allocations. An intermediate can remain computed until a consumer needs an element or addressable storage. Conversely, syntax that looks compact does not authorize skipping a producer and rebuilding a different value later.

## Semantic authority

The checked operation vocabulary owns typing, values, effects and numerical meaning together. The interpreter, physical constructors and backend operation handling consume that vocabulary. A new operation must define its full semantics and composition; a generic fallback arm cannot stand in for missing coverage.

The first applicable portable body defines the entry's reference behavior. Other bodies are choices whose whole-entry applicability must be established by [numerical construction](../compiler/numerics.md), in addition to structural and target correctness.

Read [values and types](values-and-types.md), [control and effects](control-and-effects.md), and [functions and capabilities](functions-and-capabilities.md) for the language contract. [Structure and accounting](structure-and-accounting.md) explains what source reveals to optimization. [Writing kernels](writing-kernels.md) shows how to expose it.

