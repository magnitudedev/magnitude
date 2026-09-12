# Inference operations

**An operation preserves one tensor-level mathematical contract through tracing,
reference evaluation and lowering. Models compose operations; Magnitensor may
lower one operation or a bounded region without changing that model code.**

## Semantic contract

| Property | Explicit meaning |
|---|---|
| Inputs and outputs | Shape, dtype, representation and observable layout |
| Mathematics | Values produced, reduction order where observable, and exceptional behavior |
| Precision | Accumulation, storage and rounding boundaries |
| Resources | Reads, writes, aliases and the version produced by mutation |
| Reference | Independent evaluation against which every lowering is checked |

An operation is recorded as one typed graph node when its mathematics is
distinct. Quantized projection, causal attention, delta recurrence, top-k
routing, routed experts and KV append therefore remain explicit operations.
They are not expanded into primitive arithmetic and rediscovered later.

## Placement

```text
distinct mathematical or state contract  ──▶ operation
composition of existing operations       ──▶ model function
optimization across adjacent operations  ──▶ region lowering
thread, tile or pipeline strategy         ──▶ portable kernel
native realization of a primitive        ──▶ TileLang lowering
```

A Qwen block is model composition. Its recurrence and expert evaluation are
operations. A fused norm/projection/recurrence/output/residual implementation is
a region lowering. None needs a model-named tensor operation or kernel.

Specificity is permitted where the mathematics is specific. A region used by
only one known architecture is still coherent when its conditions name the
operations, geometry, representation, precision and effects that make it valid.

## Lowerings

An operation or bounded connected region may have several implementations. Each
candidate declares:

- the graph nodes and boundary values it covers;
- accepted and produced representations and layouts;
- numerical and resource-effect equivalence;
- workspace and materialized outputs;
- a capability predicate and tuning identity;
- portable `T.Kernel` emission into a compiler-owned compilation unit.

A candidate does not own or finalize a `PrimFunc`. Finalization happens only
after the whole selected graph has been storage-planned and partitioned for
submission. This keeps independently selected lowerings composable without
post-hoc compiler-IR manipulation.

Applicability and preference are separate. Applicability proves that a candidate
is legal. Measured cost chooses among legal candidates. Selection happens over
the graph, so an individually fast operation cannot win by forcing a more
expensive conversion, intermediate or downstream schedule.

## Fusion

| Kind | Meaning |
|---|---|
| Semantic algorithm | One operation has an intrinsically fused implementation, such as streaming attention |
| Region fusion | One kernel implements a known producer-consumer subgraph |
| Cheap-operation fusion | Compatible pointwise, broadcast, view or reduction work is absorbed by an anchor |

A fused region is closed and convex. An internal value used elsewhere becomes a
region output; a dependency path cannot leave and re-enter the region. Dominance
and post-dominance govern branches and reconvergence. Mutable effects,
observable rounding, incompatible representations, unsupported reductions and
host observation are barriers.

Fusion produces one device kernel and removes interior global materialization.
Several ordered kernels behind one host invocation are a launch sequence, not a
fusion.

## State

Mutable resources use explicit versioned access. KV append consumes a visible
resource version and produces the version seen by subsequent attention. This
orders tensor work without making Magnitensor understand request acceptance,
checkpointing or prefix ownership.

The operation contract describes physical reads and writes. The model executor
retains the logical advance and publishes it only after the associated execution
completes and the generation method accepts it.

## Extension

A new operation arrives with its semantic contract, abstract evaluation,
reference evaluator and at least one portable lowering. A new optimized
candidate changes neither the operation API nor model code. A new model changes
the operation set only when it introduces genuinely new mathematics.

No operation prepares commands, allocates its own scratch, selects a backend or
submits work. Those responsibilities belong to whole selected graph lowering and
compiled execution described by [tensor-system.md](tensor-system.md).
