# State and storage derivations

These formulas bind retained information, required movement and readiness to the
[resource algebra](resources.md#evaluation-algebra). Owners specify representations,
sharing, memory budgets, lifecycle boundaries and legal reconstruction strategies.

## Append and visibility

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

That retained-window formula describes a cache retaining the last query's visible
history. It is not a universal storage minimum: if only future queries matter,
the next token needs at most `w-1` preceding positions. Use `min(l+q,w-1)` for that
obligation, and add only separately required checkpoint/history visibility.

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

Footprint and restoration optima may choose different legal strategies under the
same obligations. Do not imply their separate optima are jointly attainable.

## Required live union

```text
M_required(t) = |union(required resident weights, live state,
                         required scratch, still-in-flight allocations at t)|
M_peak_min = max_t M_required(t)
KV_bytes_without_sharing = sum_(producer a,row i) n_ai h_kva (d_ka s_ka+d_va s_va)
recurrent_bytes = b h_v d_v d_k s_state
convolution_bytes = b (z-1) c s_conv
```

`n_ai` is the number of positions required by all consumers/checkpoint obligations,
not the current allocator's reservation. Deduplicate shared prefixes and producers;
recurrent and convolution arrays join the same union when simultaneously required.
Only enforce a materialized byte lower bound where the contract requires that
representation at that boundary. Reconstructible information may admit a smaller
bound. Page padding and fragmentation are not unavoidable under a logical contract.

A reservation does not add physical bytes. Scratch peaks at different times are
not summed. In-flight buffers stay live through their last required consumer.

## Restoration cases

| Available information and obligation | Time lower bound |
|---|---|
| Accepted state available through immutable views | `0`; no positive percentage denominator established |
| Every legal strategy must move at least `N_r` bytes across boundary `r` | `max_r N_r/B_r`; charge both directions only when both are necessary |
| Every legal strategy must reconstruct with at least demands `D_repair` | `L(D_repair)`; algorithm-conditional recurrence refinement comes from [neural equations](neural.md#gated-delta-recurrence) |
| Multiple legal representation/repair strategies | Minimize the derived bound over an exhaustive admissible set; an unmodeled legal strategy forces relaxation |

Denote this result `L_restore(initial,accepted,obligations,budget)`. A known replay
route is an achievable upper bound on restoration time, not its unavoidable minimum.
Snapshot spacing from today's implementation cannot establish a lower bound.
Memory constraints justify a refinement only with proof that faster representations
cannot fit. Deferred repair is included through next-use readiness; creation/advance
costs remain part of the enclosing workload. A zero floor is an analytical result
that cannot be made positive merely by timing an expensive implementation.

## Loading

```text
L_load = max(Q_storage/B_storage,
             max_r Q_conversion_r/B_r,
             F_conversion/C_conversion)
M_load_min = unique bytes of the required final resident weight representation
```

Missing required artifact information must arrive; loading and conversion may
pipeline. Initial page-cache residency and final representation are parameters.
Reinterpretation has zero conversion demand. Quantization metadata joins the
resident union; tied weights count once. Temporary staging adds to the theoretical
peak minimum only when simultaneous necessity is established. Streaming instead of
retaining all weights changes the contract. Do not carry storage-read constraints
into resident decode.
