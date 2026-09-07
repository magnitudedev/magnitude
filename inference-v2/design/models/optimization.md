# Model optimization

**An optimized model minimizes avoidable work and executes the remaining work
efficiently, with gains established within components and their composition.**

[Composability](composability.md) defines substitution contracts. The
[performance system](../performance.md) defines ceiling and implementation evaluation,
operating points and evidence. This document owns optimization choices and their
qualification; the autonomous operation protocol governs development cycles.

## Using component models

Start from the actual architecture tree and its type records, located through the
[catalog](../performance/catalog.md). Compare measured behavior with the bound at a
matched operating point, then use parent sensitivity to identify meaningful gains.
A local low-efficiency score alone does not identify the model bottleneck.

Keep reference comparisons and theoretical bounds distinct. A compiled graph,
custom kernel or faster isolated component does not establish a full-model gain.
Protect other query/context/batch regimes and independently meaningful dimensions;
one favorable sample cannot compensate for regressions elsewhere.

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
milestones. Follow the [assessment rules](../performance.md#stable-compositions-and-evidence)
for coverage, revision invalidation and derivation-only reevaluation. Evidence
requirements do not mandate benchmarking every component after every edit.
When adopting results, update affected tree annotations and their existing benchmark
IDs together under the [tree convention](../performance.md#tree-annotations-and-benchmark-references).
Clear invalidated values even when replacement measurements are deferred.
