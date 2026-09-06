# Model composability

**Models compose computational blocks with explicit contracts. Implementations
can be compared and replaced independently, then fused and compiled together.**

This defines composition within a model executor. The engine coordinates requests;
the executor advances model state and owns the resources needed for that execution.
[Optimization](optimization.md) defines how implementation choices earn their place.
[Component identification](../components.md) defines stable IDs and assembly notation.

The model descriptions are assemblies of identified implementations:
[generic MLX-VLM](architectures/generic-mlx-vlm.md),
[Qwen3.5-family](architectures/qwen35.md), and [Gemma 4](architectures/gemma4.md).
Each tree resolves its nodes to component definitions with contracts, references,
independent checks and performance models. Shared definitions below are authoritative
across those assemblies; family-specific definitions remain in the architecture doc.

## Architecture, block, implementation

An **architecture** defines the model's computation: the arrangement of blocks,
weight interpretation, positional semantics, conditioning and state transitions.
It owns these semantics even when their implementation comes from upstream.

A **block** is a meaningful region of computation with an explicit contract.
Attention, recurrent mixing and feedforward are natural blocks; projection and
preparation or routed expert computation can expose useful nested boundaries.
Boundaries follow behavior and opportunities for reuse or independent comparison,
not every individual tensor operation.

An **implementation** realizes a block's contract using MLX operations, upstream
operators, custom kernels, or a composition of them.

| Contract property | What must be explicit |
|---|---|
| Computation | Required outputs, mathematical behavior and numerical equivalence requirements |
| Inputs | Weights, activations, positions, conditioning, dtype and layout requirements |
| State | Logical history consumed, state read or advanced, and observable resulting state |
| Support | Valid shapes, quantization formats, execution modes and storage capabilities |
| Lifetime | Mutation and aliasing rules, scratch requirements, and resources needed until device completion |

Replacing an implementation preserves this contract. Different physical state
representations may realize the same logical transition; their storage views and
conversion requirements must be explicit. Unsupported compositions fail during
binding or preparation, before model execution.

## Dependency roles

| Dependency | Role |
|---|---|
| MLX-LM / MLX-VLM | Upstream model execution, artifact compatibility and independent reference computations, exposed through adapters |
| MLX | Tensor representation, device execution, compilation and standard optimized primitives |
| Owned model code | Architecture semantics, block composition and supported implementation selection |
| Owned kernels | Specialized block implementations with demonstrated computational or execution benefits |

Choose the upstream reference for the exact artifact and required capabilities;
record the library and version. Using a second library requires an explicit role
and semantic reconciliation, rather than silently mixing its model conventions.
Upstream loading and selected operators can be reused without adopting upstream
generation loops or cache ownership throughout an optimized model.

Library-specific configuration, module and cache conventions stay at their
adapters. Shared computation depends on explicit tensors and contracts. Architecture
differences such as rotary conventions, normalization or expert routing remain
visible in the owning architecture rather than emerging from incidental imports.

## Composition without execution barriers

```text
Architecture:        embedding → [mixer → feedforward] × layers → readout
                                      │
Block implementation:    MLX operations / upstream operators / owned kernels
                                      │
Execution:                  fused and compiled tensor regions
```

A block boundary does not require a Python dispatch, GPU launch, synchronization
or materialized intermediate. Adjacent blocks may fuse; a complete resident model
step may compile together. Diagnostic access to a boundary must not impose its
cost on normal execution.

Resource preparation, physical placement and transaction bookkeeping surround
the tensor computation. State storage owns allocation and visibility; blocks
consume compatible views and perform their declared updates. Resources remain
owned until dependent device work completes. Streaming may require explicit
execution segments without moving I/O or residency policy into neural equations.

Prefill, decode and verification share model semantics while allowing different
implementations. Selection follows supported tensor geometry, precision and state
layout within the bound model implementation. [Batching](../engine/batching.md)
supplies compatible work; scheduling does not choose architecture kernels.

## Independent comparison and reuse

Meaningful block boundaries are invocable by comparison tooling with matched
weights, activations and logical starting state. A fused implementation compares
against the complete upstream region it replaces. Stateful comparisons include
resulting state, not only returned activations.

Comparisons compose from blocks to layers to the full model. Reference adapters
and diagnostic captures belong to validation tooling; production execution does
not require a parallel reference graph.

Share blocks across architectures when their behavior and contracts coincide.
Keep architecture-specific composition and semantic differences explicit. Necessary
specialization stays inside its owner; no universal architecture language or
runtime abstraction is required merely to make computations testable.

## Shared component definitions

These IDs describe current implementations. Each defines an independently invocable
boundary; dedicated benchmark coverage is not yet complete. References and derivations
below identify the required controls and performance models, not an assertion that
every component is already numerically or performance-qualified. Architecture docs
select concrete geometry, state layout and child configuration for these definitions.
Every shared contract binds to the [ceiling catalog](../performance/catalog.md#shared-model-contracts).
Each currently has one execution-efficiency dimension, identified by its contract
plus `/EXEC`, such as `MODEL:ATTENTION/EXEC`. Shape, query mode and residency select
operating points within that dimension. Individual resource costs explain its score.
Evidence identifies the selected implementation and revision; implementation changes
reset its scores and affected parent scores under the
[assessment rules](../performance.md#evidence-and-current-assessments).
The resource costs described below include diagnostic opportunities; only demands
justified by the linked derivation enter the optimistic theoretical ceiling.

### `MODEL:EMBEDDING:MAG:RESIDENT`

- **Contract / implementation:** Look up token rows in resident float or affine
  weights; the latter gathers encoded rows and dequantizes through MLX. Own lookup
  composition and execution dependencies, preserving vocabulary identity.
- **References / tests:** Independently loaded upstream embedding and direct indexing
  of independently dequantized rows. Check token order, repeats, dtype and values.
- **Performance / bounds:** Read selected encoded rows and metadata, write decoded
  activations; include gather/dequantization and dispatch. Model bytes at the relevant
  cache level rather than charging the entire embedding table per lookup. Repeated
  tokens may reuse cache lines. Full-vocabulary readout is separate work.

### `MODEL:EXPERTS:MAG:RESIDENT_GATHERED`

- **Contract / implementation:** Given hidden rows and expert assignments, return
  per-selected-expert outputs, before routing-weight reduction. Resident gate/up/down
  weights feed MLX quantized gathers and architecture-supplied activation. Assignment
  geometry selects sorted or unsorted execution; caller retains routing semantics.
- **References / tests:** Upstream expert module and a per-expert gather/matmul oracle.
  Match weights and activation; exercise assignment order, repeats, sparse/dense
  utilization and shapes on both sides of sorting selection.
- **Performance / bounds:** Selected weight bytes plus metadata, three projection
  costs, activation traffic and assignment sorting/restoration. Count unique expert
  reuse across rows and physical rereads separately. Small-row fusion can remove
  launches; larger groups can improve weight reuse. Include the enclosing router
  and reduction when claiming a feedforward improvement.

### `MODEL:ATTENTION:MAG:PAGED`

- **Contract / implementation:** Prepared Q and a logical paged KV view produce
  scaled, causal/windowed attention output. Owned `MTL` kernels through MLX compute
  softmax partials and combine them for supported short queries. Other geometries
  delegate to `MODEL:ATTENTION:MAG:GATHERED`. Storage append is outside this contract.
- **References / tests:** The gathered implementation at identical logical histories,
  plus an independent higher-precision attention equation oracle. Compare masks,
  row lengths, windows, fragmented views and output values; the child fallback
  cannot independently validate itself.
- **Performance / bounds:** For one row, equal K/V width and one query, one ideal
  KV traversal reads `2 × visible_tokens × kv_heads × head_width × element_bytes`.
  General queries require counting visible query-key pairs and tile/head reuse.
  Add query/output, mapping and partial-reduction traffic and roughly four arithmetic
  operations per query-head/key/head-coordinate pair. Derive bandwidth/compute bounds
  and reduction dependencies; physical rereads and poor occupancy explain gaps.

### `MODEL:ATTENTION:MAG:GATHERED`

- **Contract / implementation:** Gather logical paged histories, pad compatible rows,
  construct causal/window masks and call `MODEL:ATTENTION:MLX:DENSE`. This is an owned
  adapter around an upstream primitive, not an upstream paged implementation.
- **References / tests:** Independently materialized logical K/V and per-row attention
  equations. Test ordering, heterogeneous histories, padding and window boundaries;
  comparing with the paged path alone cannot establish a shared mask convention.
- **Performance / bounds:** Compose history gather reads/writes, padding/mask work and
  the dense child. Longer histories expose materialization costs even when the dense
  kernel is efficient. Account for actual temporary allocation and valid versus padded
  work. Replacing it must preserve the logical KV contract without hiding conversion cost.

### `MODEL:ATTENTION:MLX:DENSE`

- **Contract / implementation:** Upstream scaled dot-product attention over prepared
  dense Q/K/V, scale and supported mask. Runtime/primitive source is MLX.
- **References / tests:** Independent attention equations with higher-precision
  accumulation, declared output tolerance and matching causal/window visibility.
  Direct calls serve as a control for wrappers; they do not validate MLX against itself.
- **Performance / bounds:** Same mathematical attention work as the paged operator,
  with dense layout and the primitive's tiling/reuse. Model visible pairs, KV traffic,
  intermediates and actual arithmetic. Decode emphasizes history reads; prefill can
  reuse tiles. Peak hardware and measured primitive rates are different evidence.

### `MODEL:GATED_DELTA:MAG:FUSED_UPDATE`

- **Contract / implementation:** Prepared Q/K/V, decay, beta and initial matrix state
  produce outputs and final state. An owned `MTL` update holds state vectors across
  its token loop; a state-only form supports accepted-prefix reconciliation. It
  excludes projections, convolution and input/output preparation.
- **References / tests:** `MODEL:GATED_DELTA:LM:STANDARD` and an independent explicit
  recurrence. Compare every requested output, final state and prefix states under
  the same prepared inputs; include zero/full/partial accepted prefixes.
- **Performance / bounds:** Count matrix state load/store at invocation boundaries,
  prepared-input/output traffic and per-token recurrence arithmetic. The token loop
  is dependent; retaining state avoids a full external-memory round trip per token.
  Derive a resource/dependency model for actual state geometry, dtype and query width.
  Preparation and replay costs belong to the enclosing recurrent block.

### `MODEL:GATED_DELTA:LM:STANDARD`

- **Contract / implementation:** MLX-LM's gated-delta kernel under the same prepared
  input/output/state computation. Binding conventions do not change its source.
- **References / tests:** An independently expressed recurrence provides the oracle;
  the owned update supplies a differential control but is not itself proof of truth.
- **Performance / bounds:** Use the same recurrence arithmetic and state dimensions
  as the owned update, inspecting upstream's actual state traffic and query algorithm.
  Its measured rate is an attainable reference for the selected shape, not a hard
  bound. No model-level speedup follows from a local update comparison alone.
