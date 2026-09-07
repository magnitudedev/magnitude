# Resource and composition derivations

Executable formulation: [theory/resources.py](../../../performance/theory/resources.py).
The evaluator records the assumptions and bindings actually used for each bound.

This library defines demand descriptions, capacities and composition for
[component performance models](../../performance.md). Component records bind its
parameters and observation boundaries. All rates are theoretical upper capacities;
implementation measurements enter only the separate execution estimate.

## Capacity parameters

Numerical evaluation needs a versioned capacity profile, not an implementation
benchmark. For a memory interface of aggregate width `w` bits transferring at
`f_transfer` transfers/second, set the byte/second upper capacity
`B_upper = w*f_transfer/8`; effective interface throughput cannot exceed it. Use the full relevant interface, not one bank's width. For a known
execution resource with `n` units, at most `v` scalar-equivalent operations per unit
per cycle and maximum clock `f_max`, set `C_upper = n*v*f_max`. Published peak capacities
can supply these parameters when their units, device and precision match.

Cache/storage capacities constrain residency and simultaneous liveness. Different
precision/instruction pipelines have distinct upper rates; mutually exclusive uses
can share a tighter capacity inequality only when their contention is established.
Absent an authoritative count/rate, keep that capacity symbolic or omit that
constraint. Do not turn sustained measurements into a hard maximum. Do not infer
MLX/Metal instruction costs or a launch-latency floor from clock rate alone.

This profile is a theoretical envelope. Runtime/version fixes allowed semantics,
fusion and observation boundaries, while perfect legal Metal code is permitted.
Neither MLX primitive timings nor a reference engine's utilization determine it.

## Memory boundaries

`b` is row count, `q` new inputs per row, `m = bq`, `l_i` old history for row `i`,
`w` an attention window including the current position, and `s` element bytes.
Bandwidth/capacity values remain symbols supplied by a separately identified profile.

For a standalone operation, a dense input of `n` elements has logical size `ns`.
For a parent, producer outputs can remain internal and need not be transferred
through device memory. Use required unique inputs and required externally persisted
outputs at the chosen boundary; never sum every intermediate tensor's size.

If `W` distinct encoded input bytes must be accessed and at most `S` of them may
already reside above a given memory boundary, compulsory incoming traffic is at
least `max(0, W - S)`. This assumes the declared representation/input domain really
requires accessing those bytes; fixed zeros, structured weights, alternate encodings
or precomputed equivalent representations require a revised derivation. Cache
capacity and initial residency are explicit, not inferred from checkpoint size.

Required output persistence likewise specifies the memory level and completion
boundary. Merely returning an MLX array is not proof that it was evicted from cache
to DRAM. When persistence at a slower level is unproved, omit that write constraint
there. Counting logical bytes remains useful without claiming them as DRAM traffic.

## Evaluation algebra

A region contributes a demand description `D = (I, O, F, A)`:

- `I`: required external input information, keyed by identity and byte range.
- `O`: required externally persisted outputs/state, keyed the same way.
- `F`: arithmetic counts, partitioned by precision and operation resource.
- `A`: the algorithm/representation assumptions under which those demands are necessary.

`JOIN` composes descriptions: union shared byte ranges; add independently required
arithmetic; remove internal producer/consumer traffic; retain required external
features/checkpoints. It does **not** add elapsed times. Repeated tensor identity
removes duplicate input bytes, not distinct mathematical uses of its values.

For each memory boundary `r`, let `I_r` be the necessary incoming bytes and `S_r`
the maximum initially resident portion on its faster side. Let `O_r` contain only
writes proved necessary at that boundary. With upper bandwidth `B_r`:

```text
Q_r(D) = max(0, |I_r| - S_r) + |O_r|
L_data(D) = max_r Q_r(D) / B_r
L(D) = max(L_data(D), max_c F_c(D) / C_c, L_dependency(D))
U(D,u) = u / L(D)                         efficiency = L(D) / T
```

Read/write directions share a bandwidth only where the capacity definition says
so. A fused parent can use a different boundary description from a standalone
child. All missing throughput capacities are symbolic positive parameters; omitted
unproved demands contribute zero. `L_dependency` is zero unless an explicit
unavoidable dependency bound is supplied.

**The default data bound permits any equivalent MLX implementation under the fixed
representation/access contract. Arithmetic refinements below are conditional on
conventional dense dot products and the stated scalar equations.** They allow
packing, tiling, fusion and arbitrary scheduling, but are not proofs against every
possible algebraic algorithm. A record must say whether it uses the data bound or
an applicable refinement; changing that assumption creates a new derivation
assessment. Do not silently use a narrower algorithm class to claim saturation.

## Execution estimation

The implementation evaluation uses the actual selected execution plan `pi`, with
costs fitted from observations at matched operating points `theta`:

```text
T_hat_parent(theta) = TIME(pi, {estimated_region_cost_r(theta)}, obligations)
```

`TIME` evaluates that plan's resource/dependency schedule, including actual data
movement, host work, dispatch, completion and publication. It is not the theoretical
minimum over all possible plans. A serial chain of nonoverlapping complete stages
sums their costs; independent concurrent stages without contention use their maximum.
Shared-resource work requires its queue/overlap model or a joint observation. The
plan must declare which case holds; scalar child times alone do not determine it.

A primitive region fits its local metric from samples. A composite binds its
regions/children, sharing and invocation counts, then checks the composed prediction
against parent samples. Fusion across children creates a joint measured region;
do not sum isolated child timings or invent their separate fused costs. A parent's
observed residual can diagnose missing work, but is not an unavoidable theoretical
floor. Unknown region costs remain unknown.

For a footprint estimate, apply the live physical allocation union at each relevant
boundary, including the implementation's padding/scratch and in-flight retention.
For service objectives, evaluate the selected schedule/publication behavior using
the same workload/statistic as the theoretical evaluation. Do not substitute a
mean model for a percentile objective.

Parent sensitivity is a counterfactual evaluation of this same plan with specified
child costs or execution choices changed. It is an estimate, not proof of a speedup
or a percentage to add to other nodes. Changes that alter fusion/overlap, memory
feasibility or scheduling require reevaluating the plan itself.
