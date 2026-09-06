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

Each type below owns its contract, dimension definitions and formula binding.
Parameters inherit the [origin/platform rules](../performance.md#dimensions-and-parameter-binding).
`JOIN` and `L` use the [resource algebra](../performance/derivations/resources.md#evaluation-algebra);
[neural regions](../performance/derivations/neural.md#named-region-bindings) supply the shared terms.
Implementation estimates use selected execution regions and matched local/parent
observations under [execution estimation](../performance/derivations/resources.md#execution-estimation).
References and tests describe controls; they do not assert current performance qualification.

### `MODEL:EMBEDDING`

**Contract.** Look up encoded vocabulary rows, returning the required floating activations with token order,
precision and lifetime preserved.

**Parameters.** Architecture: width `h`, vocabulary and row encoding. Workload: `m` input tokens, distinct row
identities and input/output residency.

**Composition.** `D_EMBED=EMBED(m,h,rows)` from [embedding](../performance/derivations/neural.md#embedding-and-elementwise-regions).
Union repeated rows; the vocabulary head is separate work unless joined by its parent.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:EMBEDDING/EXEC` | Elapsed seconds for `u=m` lookups through output readiness. | `L(D_EMBED)`; rate upper bound `m/L`. |

**Implementations and controls.**

#### `MODEL:EMBEDDING:MAG:RESIDENT`

- **Implementation:** Look up token rows in resident float or affine weights; the latter gathers encoded rows and
  dequantizes through MLX. Own lookup composition and execution dependencies, preserving
  vocabulary identity.
- **Reference / validation:** Independently loaded upstream embedding and direct indexing of independently dequantized rows.
  Check token order, repeats, dtype and values.

### `MODEL:EXPERTS`

**Contract.** Given hidden rows and distinct top-k expert assignments, return per-selected-expert
activations before routing-weight reduction. Preserve the supplied activation and numerical
contract.

**Parameters.** Architecture: hidden/expert widths `h,f_e`, encoded gate/up/down tensors and activation.
Workload: row assignments `m_e`, with `sum_e m_e=m*t`, output requirements and residency.

**Composition.** `D_EXPERTS=JOIN({MLP(m_e,h,f_e)} for m_e>0)` using
[projections/experts](../performance/derivations/neural.md#projections-and-experts). Union each selected expert’s
weights; count each required row/expert evaluation. Output geometry is `m*t*h` at this
boundary, but may reduce internally in a routed parent.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:EXPERTS/EXEC` | Elapsed seconds for `u=m*t` selected expert evaluations, including output readiness. | `L(D_EXPERTS)`; rate upper bound `u/L`. |

**Implementations and controls.**

#### `MODEL:EXPERTS:MAG:RESIDENT_GATHERED`

- **Implementation:** Given hidden rows and expert assignments, return per-selected-expert outputs, before
  routing-weight reduction. Resident gate/up/down weights feed MLX quantized gathers and
  architecture-supplied activation. Assignment geometry selects sorted or unsorted execution;
  caller retains routing semantics.
- **Reference / validation:** Upstream expert module and a per-expert gather/matmul oracle. Match weights and activation;
  exercise assignment order, repeats, sparse/dense utilization and shapes on both sides of
  sorting selection.

### `MODEL:ATTENTION`

**Contract.** Prepared Q and logically equivalent KV histories produce scaled causal/windowed attention.
Append and state ownership are external; layouts must be matched or explicitly adapted.

**Parameters.** Architecture: `h_q,h_kv,d_k,d_v`, dtypes and numerical contract. Workload: row count `b`,
query width `q`, old lengths `l_i`, window, scale, masks and boundary residency.

**Composition.** `D_ATTN=ATTN(geometry,visibility)` from [attention](../performance/derivations/neural.md#attention). It contains
unique visible KV, Q/output and optional conventional QK/weighted-V/softmax work. Layout
adaptation belongs to the selected implementation estimate; avoid compulsory score matrices
and duplicate KV-head reads.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:ATTENTION/EXEC` | Elapsed seconds for `u=bq` queries through attended-output readiness. | `L(D_ATTN)`; rate upper bound `bq/L`. |

**Implementations and controls.**

#### `MODEL:ATTENTION:MAG:PAGED`

- **Implementation:** Prepared Q and a logical paged KV view produce scaled, causal/windowed attention output. Owned
  `MTL` kernels through MLX compute softmax partials and combine them for supported short
  queries. Other geometries delegate to `MODEL:ATTENTION:MAG:GATHERED`. Storage append is
  outside this contract.
- **Reference / validation:** The gathered implementation at identical logical histories, plus an independent
  higher-precision attention equation oracle. Compare masks, row lengths, windows, fragmented
  views and output values; the child fallback cannot independently validate itself.
- **Benchmark controls:** `operator.attention-metal-16k` and `operator.attention-gathered-16k`
  compare completed append plus attention at 16K history. This combined boundary
  does not directly measure `MODEL:ATTENTION/EXEC`.

#### `MODEL:ATTENTION:MAG:GATHERED`

- **Implementation:** Gather logical paged histories, pad compatible rows, construct causal/window masks and call
  `MODEL:ATTENTION:MLX:DENSE`. This is an owned adapter around an upstream primitive, not an
  upstream paged implementation.
- **Reference / validation:** Independently materialized logical K/V and per-row attention equations. Test ordering,
  heterogeneous histories, padding and window boundaries; comparing with the paged path alone
  cannot establish a shared mask convention.
- **Benchmark controls:** `operator.attention-gathered-16k` is the gathered control for
  `operator.attention-metal-16k`; both include append, so neither isolates attention.

#### `MODEL:ATTENTION:MLX:DENSE`

- **Implementation:** Upstream scaled dot-product attention over prepared dense Q/K/V, scale and supported mask.
  Runtime/primitive source is MLX.
- **Reference / validation:** Independent attention equations with higher-precision accumulation, declared output tolerance
  and matching causal/window visibility. Direct calls serve as a control for wrappers; they do
  not validate MLX against itself.

### `MODEL:GATED_DELTA`

**Contract.** Prepared Q/K/V, decay, beta and initial matrix state produce requested outputs and final
state; a state-only reconciliation operation preserves the same update equations.

**Parameters.** Architecture: `h_v,d_k,d_v`, state/input dtypes and update equations. Workload: `b,q`,
prepared input values, initial/final observability and output versus state-only operation.

**Composition.** `D_DELTA=DELTA(geometry,inputs,state)` from
[recurrence](../performance/derivations/neural.md#gated-delta-recurrence). Only required initial/final state crosses
the boundary; state may stay local across tokens. Projections/convolution belong to enclosing
mixers.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GATED_DELTA/EXEC` | Elapsed seconds for `u=bq` updates through required output/state readiness. | `L(D_DELTA)` with the selected output obligation. |

**Implementations and controls.**

#### `MODEL:GATED_DELTA:MAG:FUSED_UPDATE`

- **Implementation:** Prepared Q/K/V, decay, beta and initial matrix state produce outputs and final state. An owned
  `MTL` update holds state vectors across its token loop; a state-only form supports
  accepted-prefix reconciliation. It excludes projections, convolution and input/output
  preparation.
- **Reference / validation:** `MODEL:GATED_DELTA:LM:STANDARD` and an independent explicit recurrence. Compare every
  requested output, final state and prefix states under the same prepared inputs; include
  zero/full/partial accepted prefixes.
- **Benchmark controls:** `operator.delta-owned` and `operator.delta-owned-prefill-512`
  time completed prepared-input recurrence for 3 and 512 tokens; corresponding
  `operator.delta-library` and `operator.delta-library-prefill-512` are upstream
  controls. These isolate the update, excluding enclosing mixer preparation.

#### `MODEL:GATED_DELTA:LM:STANDARD`

- **Implementation:** MLX-LM's gated-delta kernel under the same prepared input/output/state computation. Binding
  conventions do not change its source.
- **Reference / validation:** An independently expressed recurrence provides the oracle; the owned update supplies a
  differential control but is not itself proof of truth.
- **Benchmark controls:** `operator.delta-library` and `operator.delta-library-prefill-512`
  cover completed 3- and 512-token prepared-input updates, matched to the owned controls.
