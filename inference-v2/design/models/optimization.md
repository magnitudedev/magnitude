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
implementations may differ while preserving the same model contract. These modes,
context lengths and batch sizes normally select samples within execution efficiency,
not separate dimensions. The [catalog](../performance/catalog.md#dimension-definitions)
declares a split only for meaningful independently moving outcomes, such as state
footprint and restoration time. Each dimension has a full `FAMILY:COMPONENT/DIMENSION`
ID, including a lone `/EXEC`. Protect unsampled regimes and auxiliary constraints;
one favorable sample or dimension cannot compensate for a regression elsewhere.

## Ceilings and headroom

The [MLX ceiling contract](../performance.md) defines the common denominator:
an optimistic theoretical bound for each declared dimension from unavoidable demands, with
perfect legal fusion/reuse and no unproved implementation overhead. Reference
rates remain comparisons; they never define the ceiling.

Each component contract binds to the [derivation catalog](../performance/catalog.md).
Source and variant select an implementation to measure, not a different standard
of theoretical efficiency. Bind geometry, state visibility, boundary residency and
profile capacities before evaluating a formula. No measurements are required to
write or inspect symbolic derivations; absent capacities and observations remain
explicitly unset.

Parent bounds combine required work and dependencies using the recursive rules,
not averages of child percentages or sums of standalone timings. A separate
measurement-based diagnosis explains actual copying, dispatch, contention and
other gaps. Unknown constraints loosen the upper bound rather than prematurely
limiting the target. A realizable implementation near the bound is useful evidence
of tightness, not a prerequisite for using an optimistic theoretical ceiling.

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
milestones. Any implementation change resets all its dimension assessments and those
of its actual parent compositions to `unmeasured`; unrelated components remain valid.
Prior samples remain historical. Follow the
[assessment rules](../performance.md#evidence-and-current-assessments) for fingerprints,
coverage and derivation-only re-evaluation. These are evidence requirements, not a
mandate to benchmark every component after every edit.
