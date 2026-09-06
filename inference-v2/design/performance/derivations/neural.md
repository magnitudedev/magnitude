# Neural derivations

These reusable regions produce demand descriptions for the
[resource algebra](resources.md#evaluation-algebra). Component records supply
geometry, tensor identities, precision, output obligations and numerical contracts.
Architecture-specific sequences and child bindings remain in their owning documents.

## Projection notation

`P(m,k,n,W)` denotes a projection from `k` to `n` for `m` rows with encoded
tensor identity `W`. `MLP(m,h,f)` joins gate/up projections, the declared
activation/product and a down projection. The [projection derivation](#projections-and-experts)
defines their arithmetic and encoded-byte counts. Biases, if present, contribute
their parameter bytes and `mn` additions. Internal activations may remain fused.

Local nonlinear work is represented by its equations, never by an incumbent kernel
count. The following conventional refinements close the otherwise omitted regions;
`x` denotes a vector of length `n`. Special functions have separate symbolic upper
capacities, or their time contribution is omitted. Do not count an instruction both
as an FMA and as separate scalar instructions on the same capacity.

| Region | Required equation and conditional resource demand |
|---|---|
| Residual/scaling | `x+y`: `n` additions; `a*x`: `n` multiplications unless the scale is an identity |
| RMS norm | `x * rsqrt(sum(x*x)/n + eps)`, optionally times learned weights: `3n+1` scalar add/multiply operations without weights, `4n+1` with weights, plus one `rsqrt`; constant reciprocal `1/n` is precomputed |
| Rotary | For `r` rotated pairs: `(xc-ys, xs+yc)` gives `6r` scalar operations; sin/cos may be precomputed and shared under the residency budget |
| SiLU / sigmoid gate | `x/(1+exp(-x))` or `1/(1+exp(-x))`, then the required product; count the expression's operations or relax the special-function cost to zero |
| Approximate GeGLU | `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) * up`; use this numerical contract, not exact erf-GELU |
| Soft cap | `cap*tanh(x/cap)` for each requested logit; no extra external pass is compulsory |
| Stable softmax | Find max, exponentiate differences, reduce and normalize; conventional model has `n-1` comparisons, `n` exponentials, one reciprocal and `3n-1` add/multiply operations |
| Expert selection | All arbitrary router scores must be considered to choose exact top-k; sorting is not required. Its comparison floor may be relaxed to zero while retaining required score dependencies |

The learned norm weights, gate parameters and scale arrays join the parameter union.
At a fused boundary these regions often add no external traffic. Arithmetic tables
are refinements of a declared equation graph; algebraic simplification and allowed
numerical equivalences must be applied before treating an operation as necessary.

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

The standard state-only update removes the Q readout: its conditional arithmetic
is `bq h_v (5d_kd_v+d_v)`. These scalar counts were checked by independently counting
operations in the decay, remembered-value projection, gated residual, rank-one
update and optional Q readout equations.

## Named region bindings

`EMBED(m,h,rows)` selects unique encoded embedding rows; `ATTN(geometry,visibility)`
uses the attention equations; `DELTA(geometry,inputs,state)` uses the recurrence.
`P` and `MLP` are defined above. `NORM`, rotary, gates and elementwise terms select
the declared local equation in the projection-notation table. These names denote
mathematical regions with input/output identities, not implementation IDs.
Each region returns demands; apply `JOIN` and `L` from the resource algebra after
selecting the actual parent boundary. A special-function rate remains symbolic
or contributes no refinement until an upper capacity is established.
