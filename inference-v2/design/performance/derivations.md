# Unavoidable-demand derivations

These formulas instantiate the [MLX ceiling definition](../performance.md). Each
section states its assumptions and relaxations. A work count is not automatically
an unavoidable resource demand; apply the memory boundary and algorithm conditions
before including it in a throughput bound.

## Notation and memory boundary

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

## Projections and experts

For a conventional dense projection `X[m,k] × W[n,k]^T`:

```text
logical input/output bytes = s_x m k + s_y m n
conventional scalar arithmetic = m n (2k - 1)       no bias
weight bytes, float = n k s_w
weight bytes, affine = n k p/8 + n ceil(k/g) (s_scale + s_bias)
```

The affine expression assumes aligned packed storage with `p` bits and group size
`g`; exact tensor-header extents override it for padding or mixed encodings. Do not
charge full floating-point weight materialization: dequantization can fuse with use.
Count weight bytes once across rows under ideal reuse, then apply the memory rule.
Only use the arithmetic term as a bound when conventional dense dot-product
execution is an explicit assumption; it is not a universal algebraic lower bound.

For gate/up/down MLP dimensions `h → f → h`, three projection counts give
`m [2f(2h - 1) + h(2f - 1)]` conventional scalar operations. Gate/up intermediates
need not leave the fused parent. Activation/transcendental resource constraints are
additional only when independently justified.

For MoE, `E` experts and `t` selected per row imply `mt` expert evaluations, but
only `e_unique` distinct expert weights, with
`t <= e_unique <= min(E, mt)` for nonempty rows with distinct top-k selections.
Use actual assignments for a trace-conditioned assessment; otherwise state the
range and use maximal reuse for the most optimistic bound. Do not grant foreknowledge
that changes routing semantics. Shared expert work and router work remain separate.
Sorting is an implementation choice, not unavoidable work. Duplicate hidden rows,
exact zero gates or structured weights can change arithmetic requirements.

## Embedding and elementwise regions

An embedding lookup requires the selected rows, not the complete vocabulary table.
With `v_unique` distinct requested token rows and width `h`, the selected float
payload is `v_unique h s`; affine payload follows the projection storage formula
for those rows. Output geometry is `m h`. Encoded row gathering and conversion may
fuse. Tied full-vocabulary projection still consumes the full head's relevant weights.

For residuals, normalization, gates, rotary transforms and soft caps, start with
unique boundary inputs and required outputs. Normalization also depends on a
reduction across its normalized dimension; count reduction/arithmetic only under
an explicit computational model. Neither one launch per operation nor intermediate
writes are mandatory. A fused pointwise region may add no compulsory DRAM traffic
beyond its parent's inputs/outputs; report the resulting loose bound honestly.

## Attention

For each row, query `j` (1 through `q`) attends `l_i + j` positions causally.
The number of visible query-key pairs, without query heads, is:

```text
P = sum_i [q l_i + q(q + 1)/2]                       full attention
P = sum_i sum_(j=1..q) min(w, l_i + j)               windowed attention
```

The union of visible key positions within a row is:

```text
V_i = l_i + q                                       full attention
V_i = min(l_i + q, w + q - 1)                       windowed attention
```

With `h_kv` KV heads, widths `d_k,d_v` and storage sizes `s_k,s_v`:

```text
unique logical KV payload = sum_i V_i h_kv (d_k s_k + d_v s_v)
query/output geometry     = b q h_q (d_k s_q + d_v s_o)
```

Assume each query head has ordinary content-dependent access to its visible history.
KV-head sharing permits reuse across query heads; do not multiply unique KV payload
by `h_q/h_kv`. Multiple queries can reuse tiles. At a fused producer/attention boundary,
new K/V may be available internally; for old-history reads subtract their contribution.
For a full model, count shared producer storage once and assume optimistic consumer
reuse unless a memory-capacity/I/O argument proves additional transfers.

Conventional QK dot products cost `h_q P (2d_k - 1)` scalar operations. The weighted-V
sum costs `h_q d_v (2P - bq)` when every query has at least one visible key. Softmax
is additional. These are counts for standard dot-product attention, not a claim
that all exact-attention algorithms need those exact operations or materialized scores.
The initial optimistic resource bound may omit unproved reduction/softmax costs.

Use these compulsory-data constraints and, when applicable, arithmetic constraints
in the resource maximum. Flash-style tiling can avoid a `q × history` score matrix;
never include that matrix as unavoidable traffic. Stronger I/O bounds need explicit
fast-memory capacity and algorithm assumptions. Paged-table lookup, padding, gathers
and partial-combine overhead are current costs, not automatic theoretical minima.

## Gated delta recurrence

For value heads `h_v`, key/value dimensions `d_k,d_v` and state element size `s_state`:

```text
matrix state elements per row = h_v d_v d_k
matrix state bytes per row    = h_v d_v d_k s_state
```

Under the standard update equations, each token decays state, projects remembered
values with K, forms a gated residual, updates state and projects output with Q.
A scalar-operation count for that equation order is
`b q h_v (7 d_k d_v)` when outputs are needed, excluding input preparation.
This is a conditional work count, not a bound across alternative chunked algorithms.
State-only reconciliation can omit Q/output work; it is a different output contract.

For a fused q-token invocation, initial/final state are the boundary state. The
matrix can remain local across the token loop; charging a full external-memory
state read/write for every token is unjustified. Apply the memory-level rule to
initial and final images. Convolution history and prepared Q/K/V/gates are separate
inputs; a full recurrent block may fuse their preparation. State dependence exists,
but a lower bound that assumes q whole kernels in sequence would exclude valid
chunked or parallel formulations without justification.

## State, loading and execution

For KV append, `new_tokens × h_kv × (d_k s_k + d_v s_v)` is new logical state per
producer. An immutable shared prefix requires no duplicate payload until a branch
changes data whose representation cannot remain shared. Full-page copy, dense
batch reconstruction and checkpoint cloning are not inherently necessary. When a
contract requires a physical copy of `N` bytes across a memory level, count both
its necessary read and write there; a logical view change does not imply that copy.

Retained sliding-window state and a query's temporary visibility are distinct:
`min(l + q, w)` retained positions versus up to `min(l + q, w + q - 1)` visible
positions during a wide advance. A capacity reservation is not traffic. Recurrent
state is fixed-size; repair can require recomputation, but the optimum may instead
retain accepted-boundary information within the allowed memory budget.

Cold loading requires absent artifact information to cross storage boundaries.
Apply `missing_required_bytes / storage_capacity`, with loading/conversion overlap
relaxed optimistically. Do not reuse that constraint for resident decode.

Default unavoidable host graph, allocation and launch costs are zero until a
specific MLX constraint proves otherwise. This intentionally loosens the ceiling:
one Python call or kernel per current component is not mandated by MLX. A positive
runtime floor must identify its required boundary and justify its minimum cost;
a measured incumbent overhead is only diagnostic evidence.

## Footprint and restoration

Footprint bounds count indispensable live information under the contract's allowed
representations, sharing and reconstruction, not the incumbent allocations. Logical
tensor bytes are a lower bound only where the contract requires that representation
to be materialized. Otherwise permit compression/recomputation optimistically and
record which positive bound remains justified.

For loading, let `W_resident` be the unique bytes of the required final materialized
weight representation. Since those weights must be live at completion:

```text
M_load_min >= W_resident
initial optimistic M_load_min = W_resident
```

Include metadata required by the encoding; tied weights count once. Temporary
conversion buffers add nothing until a positive minimum simultaneous footprint is
proved. Measure actual peak live allocations over the loading interval, including
staging and the final weights. Streaming instead of retaining all weights changes
the residency contract and requires a different binding.

For state, derive the union of information that must coexist at the selected
lifecycle boundary. When materialized in the required representation, a KV history
contributes `retained_positions × h_kv × (d_k s_k + d_v s_v)` per independent
producer; recurrent matrices contribute `b h_v d_v d_k s`, plus required convolution
history. Deduplicate shared prefixes, aliases and shared producers. Sum only state
that must coexist; do not sum memory peaks at different times.

Required checkpoint availability does not imply a complete stored image per
checkpoint. Sharing, deltas or replay may satisfy the same contract. Add only
indispensable retained reconstruction information, with its necessity justified;
do not assume a snapshot count from the current implementation. The minimum may
remain a loose bound on a required materialized subset. If that subset is itself
reconstructible at this boundary, relax it too. Unproved metadata/allocation floors
are zero; an entirely unresolved positive minimum yields no memory percentage.

For restoration, fix the advanced/accepted positions, required availability, allowed
initial information and memory budget. Minimize required movement/recomputation
over legal strategies. A current implementation's replay or full-cache copy is not
unavoidable if another legal checkpoint representation could eliminate it. A required
physical transfer uses the memory-boundary rule; provably required replay uses the
relevant neural/recurrent derivations. These demands feed the common resource rule.

An already available immutable checkpoint may permit a logical view change with no
established positive time floor. In that case `RESTORE` is unresolved, not a fabricated
percentage; retain the measured latency as diagnostic evidence. Include deferred
repair through readiness for the next use, so moving work outside the restore API
does not manufacture a speedup. Checkpoint creation/advance costs also remain part
of the enclosing workload evaluation.

The footprint minimum and restoration minimum may choose different legal strategies
under the same external obligations. Record constraints beside both; do not imply
their individual optima are jointly attainable.

## Parent and engine composition

Combine leaf demands with the [recursive rules](../performance.md#recursive-composition).
A fused layer can eliminate inter-block traffic; a model may still have compulsory
weight/state reads. Shared-memory capacity can tighten reuse assumptions, but absence
of that proof must not become an assumed reload at every child boundary.

A generation round committing `u` outputs has a ceiling `u / L_round`, where the
round includes unavoidable target/drafter/repair work under its declared method.
The most optimistic accepted-output count can give a loose upper bound. It is not
an expected rate: expected useful outputs and expected total round time must be
modeled over the same workload before deriving an expected-throughput bound.
Never treat measured acceptance as an architecture constant or add free future knowledge.

For engine throughput, fix offered work and the required latency/fairness/memory
constraints. Aggregate unavoidable service demand and use optimistic admissible
packing. Scheduling/metadata overhead is zero until constrained otherwise. Policy
bounds are workload-dependent; an exact percentage for a zero-cost administrative
operation is undefined. Show its measured excess cost through the parent when available.
