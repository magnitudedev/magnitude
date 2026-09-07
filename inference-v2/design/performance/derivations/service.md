# Service and generation derivations

Executable formulation: [theory/engine.py](../../../performance/theory/engine.py).
The evaluator records the assumptions and bindings actually used for each bound.

Component owners bind these formulas to admission, scheduling, batching, caching
and generation contracts. [Resource algebra](resources.md#evaluation-algebra) supplies
model demand and capacity accounting; [state derivations](state.md) supply storage
and restoration obligations. Metric definitions live in the component records.

## Workload relaxation

A workload `W` fixes arrivals, prompts, compatible prefixes, requested outputs,
model/numerical/sampling requirements, readiness, cancellations/stops and initial
residency. Its service contract fixes memory, admission capacity, chunk allowances
and fairness. A distribution supplies these jointly rather than as unrelated means.

Build a DAG of prompt/target/draft/repair work, sampling, state readiness and
publication. Model operations supply mathematical demand descriptions. Legal groups
respect executable model, state/conditioning, query width and phase compatibility.
For candidate schedule `s`, require:

```text
compulsory demand on resource r within any interval [a,b] <= C_r*(b-a)
start of dependent value use >= required value availability
live required information at t <= memory budget
request service starts only after external inputs are available
```

Only prove interval-local demands that hold for every legal execution. Whole calls
are not atomic merely because their values are dependent; work may tile, fuse and
overlap. Union shared weights in a group, count per-row arithmetic and independent
state, and prove any required repeated transfers across groups. Different query
widths alone do not prove repeated DRAM weight reads.

Let `R(W)` contain every legal execution, with optional optimistic relaxations:
perfect packing, instantaneous bookkeeping, internal pipelining and fractional
service choices. For a lower-is-better objective `J`:

```text
L_service(W,J) = inf_(s in R(W)) J(s)
U_rate = useful committed outputs / L_service(W,makespan)
```

This is an explicit constrained optimization model for a bound, not a measured
reference. Group enumeration instantiates a finite workload problem without MLX.
Label any offline knowledge of arrivals/outcomes granted as a relaxation; it does
not grant precomputed neural answers. A relaxed optimum need not be attainable.

Use the same observation interval and population statistic for bound and measurement.
Minimizing mean delay does not bound p95. Coordinatewise lower bounds can be combined
using the same monotone statistic to give a weaker bound, though their individual
optima need not coexist. Completion, response and publication-gap objectives are
parameters of this derivation, defined by their component owner.

## Time sharing and admission

For sustained contention with exact asymptotic decode share `s`, prefill/decode
service minima `L_P,L_D`, and sequential phase service:

```text
L_busy >= max(L_P/(1-s), L_D/s)
```

Each phase's allocated time must cover its required service. For finite intervals
with proved share errors `delta_P,delta_D`, use
`max(0,(L_P-delta_P)/(1-s),(L_D-delta_D)/s)`. Without finite error bounds omit that
refinement. Never impose contended shares on idle/uncontended time. An operation
may overrun a soft duration target; it is not a hard completion upper bound.

For a fixed FIFO admission workload, let `t_free` be the earliest feasible time
in the relaxed problem that admits a request after honoring prior FIFO obligations,
capacity and reclamation. Then `L_wait=max(0,t_free-arrival)`. Immediate feasibility
gives a zero wait floor. Actual bookkeeping time belongs to execution estimation.

## Prompt work and grouping

For a prompt chunk `q` after `l` inputs, full-attention pairs are
`q*l+q(q+1)/2`. Summing consecutive chunks equals the unchunked pair count; windowed
pairs follow [attention](neural.md#attention). Chunking does not invent semantic
pairs. Repeated weight reads or boundary work enter the ceiling only when proven
necessary under the memory/execution contract.

For a ready set, form legal groups and describe each by:

```text
D_group = JOIN(member model work, shared parameters once,
               independent required state, requested row outputs)
L_ready = inf_(relaxed legal groupings/schedules) completion bound
```

Do not sum standalone best group latencies across shared residency or overlap.
The grouping policy is tested by its effect on ready-work completion. Producing
group descriptors itself can have a zero time floor unless the contract requires
an observable transfer: `n` descriptors of `s` bytes across a boundary with initial
residency `S` give the conditional refinement `max(0,n*s-S)/B`.

## Prefix reuse

Enumerate compatible checkpoints `c` constructible before request `i` and usable
prefix lengths `h_ic`, excluding required anchors. Each checkpoint includes complete
state or reconstruction obligations and semantic namespace. Let `x_ct` mean retained
at event interval `t`, and `y_ic` mean selected for request `i`:

```text
H_max = max sum_(i,c) h_ic*y_ic
sum_c y_ic <= 1; y_ic <= x_c,arrival_i
retained physical union at each t <= cache budget
sum_c x_ct <= retained-entry limit
x_ct = 0 before c is constructible; leases forbid unsafe reclamation
```

This offline optimum is an optimistic upper bound on reusable prompt inputs skipped.
Fractional choices, omitted restoration cost or ignoring budget can loosen it further;
`sum_i max_c h_ic` is a budget-free upper bound. Shared backing is a union, not the
sum of checkpoint sizes. No reuse opportunity yields no meaningful reuse percentage.
Restoration, retention overhead and pressure still affect enclosing service metrics;
reuse count alone does not establish net performance benefit.

## Sampling and acceptance

Sampling demand binds eligible vocabulary, policy, constraints/history, randomness
and requested log probabilities. For arbitrary greedy logits all eligible candidates
must be considered; the conventional comparison count is `V_eligible-1`. Softmax,
filtering and requested normalization use the [local equation model](neural.md#projection-notation).
Exact top-k/sampling need not sort every score. Position-addressed randomness is a
semantic requirement, not a mandatory host fence. At a fused head/sampler boundary,
logits can remain internal; observation requirements determine external traffic.

Acceptance compares proposals with target samples through the first mismatch or
terminal proposal. With width `d` and accepted prefix `K`, a sequential model inspects
`min(K+1,d)` proposal/sample pairs, including all `d` on full acceptance. Rejected
suffixes and cumulative-product arrays are not inherently necessary. Bind required
inputs, accepted count/bonus outputs and any justified comparison capacity; a fused
boundary may lack a positive standalone floor.

## Generation rounds

Plain advancement joins target work, sampling, state obligations and publication.
For speculation, depth `d`, verification width `d+1` and acceptance `K` give `u=K+1`
committed outputs before stop/allowance truncation. Set `a_j=P(K>=j)`, `a_0=1`:

```text
E[u] = 1 + sum_(j=1..d) a_j
Pr(K=k) = a_k-a_(k+1) for k<d; Pr(K=d)=a_d
D_round(k) = JOIN(draft invocations, target verification(d+1),
                  sampling/acceptance, required repair(k), publication)
L_expected = sum_k Pr(K=k) * L(D_round(k))
U_round = E[u] / L_expected
```

Use a joint distribution when routes, lengths, stops or depth also vary. Do not
substitute `E[u/T]` or independent marginal means for correlated work/outcomes.
Without an acceptance model, the optimistic upper bound is
`(d+1)/inf_k L(D_round(k))`, capped by the allowed useful outputs and explicitly
relaxed to favorable acceptance. No measured acceptance rate is required.

Draft predictions retain their dependencies; proposal inputs are not known in
advance. Union shared target/drafter weights at the round boundary. For concurrent
requests, use compatible-group demands and the workload relaxation instead of
multiplying single-request rates. Required catch-up and deferred repair stay in the
round or enclosing workload that must pay them. Algorithm-specific draft graphs
belong to their component owners.

An upper bound exceeding another method's attained rate shows room, not a proven
speedup. An upper bound below an attained rate can rule out a candidate under the
same assumptions. References establish attainment and never define the ceiling.
