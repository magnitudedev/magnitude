# Kernel optimization

**Treat optimization as an explanation of unnecessary cost, followed by an implementation
that removes it.** The goal is a better execution of the required computation, with a
clear reason it should be correct, faster and useful beyond the case that exposed it.

The guide has three concerns: understanding the opportunity, designing the resource
trade, and establishing which parts of the result are worth keeping.
[Kernel construction](../kernels.md) defines the composition boundaries;
[model optimization](../models/optimization.md) defines adoption in the enclosing system.

## Understand the opportunity

### Start from obligations, not the existing kernel

Write down what must be computed: outputs, numerical behavior and state effects.
Distinguish these obligations from the current algorithm, intermediate values, layout
and thread arrangement. Independent requests require independent results; they do not
require repeating preparation or decoding. Conversely, equivalent real-number equations
do not justify changing floating-point reductions, casts or multiply-add contraction.

Establish where faster execution matters. A slow-looking instruction or an inefficient
leaf is not necessarily the parent's bottleneck. Compare the current implementation with
an appropriate reference and the [theoretical model](../performance.md). References show
attainable performance; ceilings bound what is possible. Failed implementations cannot
justify lowering a ceiling. Tightening it requires a mathematical proof.

### Follow values to explain the cost

For the important inputs and intermediates, determine:

- **Production:** what creates the value, how often, and which operands actually change?
- **Consumption:** which outputs, requests, heads or tokens need the same value?
- **Residence:** where does it live, how large is it, and how long is it retained?
- **Execution:** which threads produce and consume it, and what dependencies connect them?

This connects arithmetic, memory, precision and scheduling into one picture. Sharing a
weight load may still leave repeated decoding. Removing an intermediate may lengthen
register lifetimes. A small mathematical expression may generate expensive indexing,
conversion or synchronization.

Use that picture to explain the gap: unnecessary demand on a resource, inefficient use
of its capacity, or a dependency preventing useful parallel execution. Count actual
instructions and traffic at the relevant memory level; requested loads are not necessarily
device-memory reads. Identify the largest plausible source of recoverable parent cost.

## Design the resource trade

### Change the cause of the cost

Choose a transformation from the explanation, rather than choosing a familiar technique
and searching for somewhere to apply it. The fundamental opportunities are:

| Opportunity | What to look for | What to change |
|---|---|---|
| Eliminate work | Unused outputs, redundant transformations, avoidable intermediates | Compute less, change the algorithm, or connect producers directly to consumers |
| Amortize work | Operands invariant across multiple consumers | Share the useful prepared value at the scope where those consumers meet |
| Compact values | Storage wider than the value's actual information content | Use an exact smaller representation; keep arithmetic precision independent |
| Rearrange execution | Poor access patterns, serial work, idle lanes or excessive live state | Change ownership, layout, tiling, fusion or partitioning |

These opportunities interact. Consider a different dataflow or representation before
sweeping parameters within an arrangement that cannot remove the main cost. Implement
coupled changes together when their benefit depends on each other.

### Account for what the transformation adds

Reuse exchanges recomputation for retained storage. Fusion exchanges intermediate traffic
and launches for longer lifetimes and a shared schedule. Larger tiles exchange repeated
loads for more live state and potentially less parallelism. Each can win or lose.

Choose the cached value, its representation, location and lifetime together. Registers,
threadgroup memory and device memory offer different sharing scopes and costs. Sometimes
recomputation or an explicit intermediate makes the overall execution cheaper.

State a candidate's argument concretely: **the cost removed, the cost introduced, why
correctness is preserved, and which operating conditions should benefit.** Prefer a
coherent alternative with a large predicted effect over accumulated special cases.

### Prove representation choices separately from scheduling

Storage, multiplication and accumulation precision are independent decisions. Determine
range and significant bits, including conversion and exceptional-value behavior. A
smaller exact representation introduces no approximation; changing arithmetic or accepting
rounding error requires separate numerical qualification.

For example, four-bit unscaled coefficients `q × 2^(4j)`, with `0 ≤ q ≤ 15` and
`0 ≤ j ≤ 3`, fit FP16 exactly: at most four significant bits and a maximum of 61,440.
Caching them compactly can make shared decoding affordable while products and accumulators
remain FP32. Fully dequantized weights do not inherit that proof. The transferable insight
is to retain the smallest sufficient shared information, not to lower precision globally.

## Establish what is worth keeping

### Verify the mechanism, not just the timing

After a meaningful candidate is implemented, use short comparisons to test its numerical
contract and predicted effect. Control values, layouts, requested outputs and compilation
boundaries. Synthetic inputs isolate execution questions; model inputs expose realistic
distributions. Choose evidence for the question instead of launching a broad campaign.

Source-level reuse does not guarantee machine-level reuse. Inspect generated execution
when results contradict the explanation: specialization, static indexing, register
allocation, copies and fusion can change the outcome. Do not infer a specific mechanism,
such as eliminated spills, from timing alone. Revise an unsupported explanation rather
than extending it with more tuning.

### Keep the benefit and contain the complexity

Confirm that the improvement survives composition and its integration costs. Protect
relevant batch, shape, dtype and layout regimes; one favorable workload does not establish
a general improvement. A repeated hot-weight kernel cannot stand in for a full model
streaming different weights.

Once a candidate works, remove supporting changes whose contribution is uncertain and
check that the benefit remains. Preserve mechanisms that earn their resource and complexity
cost. Express their numerical guarantees and execution requirements through reusable
boundaries so the same reasoning transfers to other kernels.

The resulting knowledge should explain **why the arrangement wins, where that explanation
holds, and what remains limiting**. Keep measurements and rejected experiments in the
existing performance records. Carry forward a justified implementation and a better cost
model, rather than a growing collection of benchmark-specific adjustments.
