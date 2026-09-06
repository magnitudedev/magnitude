# Model optimization

**An optimized model minimizes avoidable work and executes the remaining work
efficiently, with gains established both within blocks and in their composition.**

[Composability](composability.md) supplies the contracts and replaceable
implementations. This document defines performance properties and evidence;
the autonomous operation protocol governs development cycles.

The architecture trees select implementations; their component definitions are
the unit of analysis. Shared definitions live in
[composability](composability.md#shared-component-definitions); architecture-specific
ones live alongside the [Qwen](architectures/qwen35.md),
[Gemma](architectures/gemma4.md) and [upstream](architectures/generic-mlx-vlm.md) trees.
Every node resolves to its own reference, independent check and cost model. Parent
models use those same IDs rather than maintaining a separate list of unnamed costs.

## Performance belongs to an operating point

An architecture name or the presence of custom kernels does not establish
optimization. A performance claim identifies the implementation, artifact,
hardware, precision, batch size, query width, context lengths, state layout,
residency and requested outputs. Compilation and preparation costs have explicit
timing boundaries; steady execution and first-use latency are distinct properties.

Prefill, decode and verification have different reuse and parallelism. Their
implementations may differ while preserving the same model contract. Single-session
latency, aggregate throughput and memory efficiency remain separate properties;
an improvement in one does not silently compensate for a regression in another.

## Ceilings and headroom

Each identified model component has the reference and performance description
required by [component identification](../components.md). An architecture's model
composes those descriptions at block, layer and whole-model boundaries rather
than attaching one unexplained throughput target to the architecture. It
distinguishes three kinds of evidence:

| Evidence | Meaning |
|---|---|
| Reference target | A comparable implementation demonstrates an attainable rate |
| Resource bound | Required work and hardware capacity imply a throughput upper bound under stated assumptions |
| Execution diagnosis | Measured traffic, utilization, dispatch and dependencies explain the current gap and suggest opportunities |

For work requiring `D` bytes through a memory level with bandwidth `B`, and `F`
operations of a given kind with capacity `C`, execution time is at least
`max(D / B, F / C)`. Account separately for distinct compute resources and memory
levels. Compose bounds according to the dependency graph: serial work adds,
overlap must be feasible, and operations sharing a resource share its capacity.
An aggregate resource bound alone may miss a serial critical path.

For a selected component, instantiate its definition with actual rows, query width,
visible history, head/matrix geometry, precision and layout. Attach required bytes,
arithmetic and dependency assumptions to that ID. Evaluate the parent from its
selected child IDs and integration costs, then compare both parent and children
with their independent controls. Unknown bandwidth, reuse or overlap requires a
named discriminating measurement, not an invented numerical ceiling.

| Work | Required accounting |
|---|---|
| Projections and experts | Actual weight bytes including quantization metadata, activation traffic, dequantization and arithmetic; selected experts and reuse across rows |
| Attention | Query geometry, visible KV reads, new KV writes, attention arithmetic, windows, grouped heads and shared KV producers |
| Recurrence | State reads and writes, convolution history and update arithmetic; sequential decode versus parallel or chunked prefill |
| Composition | Intermediate materialization, copies, layout conversion, scratch, host graph construction, encoding, synchronization and exposed I/O |

Decode can be limited by weight traffic, KV/state traffic, host submission or
serial dependencies. Prefill can reuse weights across many tokens, making matrix
compute, attention I/O and workspace more significant. Neither classification is
assumed without accounting for the actual architecture and shapes.

Hardware capacity gives an optimistic bound, not a promise of attainment.
Measured bandwidth and operator latency diagnose current execution; they do not
limit future targets. Existing launch counts, layouts and intermediate tensors
are not inherently unavoidable work. Fusion, reuse, overlap and different
algorithms can change the bound itself. A claim of being near a ceiling requires
the assumptions and remaining gap to be explicit, beyond matching a reference.

## Structural efficiency

Implementation choices address identifiable sources of cost. These are recurring
opportunities, not mandatory fusions for every architecture or shape:

| Opportunity | Intended effect |
|---|---|
| Compile a stable tensor graph | Amortize graph construction and expose optimization across block boundaries; avoid needless retracing |
| Pack compatible projections sharing an input | Combine Q/K/V, gate/up or recurrent projections at load time; reduce dispatch and repeated input handling |
| Fuse preparation and epilogues | Combine normalization, positional transforms, gates, activations or residual work; eliminate dependent launches and intermediate traffic |
| Fuse routed expert computation | Combine compatible routing work, expert gate/up activation and weighted output reduction; exploit expert reuse at larger batches |
| Match attention to query and history geometry | Read valid history with useful head/tile reuse; avoid materialized score matrices, full-history gathers or scanning excessive spare capacity |
| Preserve state efficiently | Append new KV, reuse immutable prefixes, respect windows and update recurrent state without unnecessary history copies |
| Specialize matrix execution | Choose efficient quantized vector, narrow-matrix or grouped-matrix implementations for the actual shapes |
| Eliminate unused work | Compute only requested logits/features; avoid unneeded projections, repeated conversions and forced host synchronization |

Packing must respect quantization formats and weight ownership. Fusion must
preserve required rounding and state effects. Larger fused kernels can lose
parallelism or occupancy; compiled outputs and implicit contiguity requirements
can introduce copies. Judge the resulting execution, not source operation count.

Use standard MLX primitives when they efficiently realize the computation. Own a
kernel when a different algorithm, layout, fusion or shape specialization offers
a demonstrated benefit. A proven structural redesign can be pursued directly;
local tuning need not be exhausted first. Retain only mechanisms with an established
purpose, and keep necessary complexity behind the computational contracts.

## Comparable blocks and composite evidence

Every optimized region has an independent control for its declared behavior.
Pinned upstream computation supplies a model reference; a readable MLX operation
chain can supply a convenient development control after it is checked against
that reference. Shared helpers cannot independently validate themselves.

Inputs include representative model activations and states across supported
shapes, alongside targeted edge cases. Numerical requirements are established
before judging candidates. Compare activations and state transitions; diagnose
divergence rather than adjusting a gate to admit a speedup.

```text
Operator / fused region → stateful block → layer → full model → engine workload
```

Each enclosing comparison includes integration costs and accumulated numerical
effects. Local improvements do not add arithmetically: they may overlap, contend
for resources or move the bottleneck. Repeating one block with cached weights is
not evidence of full-model weight-stream performance. Isolated device timings
and execution through the normal compiled path answer different questions.

Accepted gains identify their supported operating range and protect affected
prefill, decode, verification, context and batch regimes. Qualification includes
cases outside those used for tuning. Stateful changes also preserve reuse,
independent row progress and accepted-boundary restoration. Full-model gains must
survive engine integration before supporting an engine-wide claim.

Measurements, failed controls, reference identities and bound assumptions remain
durable evidence alongside source identities, comparison conditions and raw results.
Small controlled comparisons support development; broader qualification supports
milestones. When implementations change, prior measurements remain historical
evidence rather than automatically qualifying the new composition.
