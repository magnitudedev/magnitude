# Qwen3.5-family hybrid models

## Scope

**The model composes owned hybrid blocks and upstream operations. Resident
single-input execution compiles their tensor transitions together; state preparation
and publication remain outside compilation.** This covers the
accepted Qwen3.5-family text layouts, including compatible Qwen3.6 artifacts, dense
or routed feedforward, and converted affine weights.

MLX-LM supplies configuration and parameter containers. MLX-VLM supplies the
independent target reference and the explicitly adapted standard rotary calculation.
The program exposes residual features, not media conditioning. Source attribution
follows [component identification](../../components.md).

## Assembly

```text
MODEL:QWEN35:MAG:RESIDENT_COMPILED
├── embedding · MODEL:EMBEDDING:MAG:RESIDENT
├── layers.0.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.0.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.1.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.1.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.2.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.2.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.3.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.3.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.4.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.4.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.5.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.5.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.6.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.6.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.7.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.7.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.8.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.8.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.9.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.9.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.10.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.10.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.11.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.11.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.12.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.12.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.13.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.13.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.14.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.14.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.15.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.15.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.16.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.16.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.17.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.17.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.18.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.18.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.19.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.19.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.20.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.20.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.21.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.21.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.22.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.22.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.23.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.23.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.24.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.24.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.25.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.25.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.26.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.26.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.27.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.27.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.28.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.28.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.29.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.29.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.30.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.30.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.31.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.31.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.32.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.32.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.33.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.33.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.34.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.34.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.35.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.35.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── layers.36.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.36.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.37.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.37.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.38.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.38.mixer · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   └── update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
├── layers.39.feedforward · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   └── experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── layers.39.mixer · MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   └── attention · MODEL:ATTENTION:MAG:PAGED
│       └── fallback · MODEL:ATTENTION:MAG:GATHERED
├── readout · MODEL:QWEN35.READOUT:MAG:STANDARD
└── state · STATE:QWEN35:MAG:HYBRID
    ├── kv · KV:STORE:MAG:PAGED
    │   ├── append · KV:APPEND:MAG:CONTIGUOUS_RUNS
    │   └── branch · KV:BRANCH:MAG:COPY_ON_WRITE
    └── recurrent · STATE:RECURRENT:MAG:CHECKPOINTED
```

## Component definitions

Each type below defines its behavior and mathematical assumptions. Executable bindings
are owned by the [performance catalog](../../performance.md#ownership-and-component-records).
Parameters inherit the [origin/platform rules](../../performance.md#dimensions-and-parameter-binding).
`JOIN` and `L` use the [resource algebra](../../performance/derivations/resources.md#evaluation-algebra);
[neural regions](../../performance/derivations/neural.md#named-region-bindings) supply the shared terms.
Implementation estimates use selected execution regions and matched local/parent
observations under [execution estimation](../../performance/derivations/resources.md#execution-estimation).
References and tests describe controls; they do not assert current performance qualification.

### `MODEL:QWEN35`

**Contract.** Consume text inputs through the configured hybrid layer graph, producing requested
logits/features and valid attention/recurrent state.

**Parameters.** Architecture: actual layer types, all child geometry/weights, numerical contract and
dense/routed configuration. Workload: `b,q,l_i`, requested residual features/logit rows, state
and residency.

**Composition.** Select the attention or recurrence contract per layer, then its feedforward. The [shared
embedding](../composability.md#modelembedding), local norms/residuals, requested readout and
external features join once. Shared state uses the [hybrid state contract](#stateqwen35).
Current Python graph construction belongs to implementation estimation.

```text
D_QWEN = JOIN(D_EMBED,
  {input_norm_j, mixer_j, residual_j,
   feedforward_norm_j, D_QFF_j, residual_j}_j,
  requested D_QHEAD, requested residual features, required final state)
mixer_j = D_QA_j or D_QR_j according to the artifact
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:QWEN35/EXEC` | Elapsed seconds for `u=bq` consumed inputs through all requested outputs/state. | `L(D_QWEN)`; rate upper bound `bq/L`. |

**Implementations and controls.**

#### `MODEL:QWEN35:MAG:RESIDENT_COMPILED`

- **Implementation:** Compile the resident single-input layer assembly, including
  independent batched rows, requested residual features, readout and functional
  attention/recurrent updates. Embedding, attention projection/finish, routing and
  expert math are shared with layerwise execution. Wider inputs and unsupported
  storage/operator compositions use the layerwise implementation below.
- **State boundary:** State storage prepares already-reserved writable addresses
  and pinned buffer views. Tensor execution returns new buffer versions; state
  storage installs the complete validated result as a tentative boundary. Existing
  transactions own acceptance, rejection and completion lifetime. Compilation
  neither allocates physical pages nor grants writes to retained prefixes.
- **Specialization:** Positions and physical addresses are tensor operands. Cache
  specialization follows batch size, physical capacity, requested outputs and the
  attention launch horizon. Pad page maps to attention partitions so ordinary
  storage-page growth does not retrace the whole model; retain at most four compiled
  geometries. Padding changes neither causal visibility nor required attention splits.
- **Reference / validation:** Compare complete outputs and logical state with the
  layerwise implementation, including mixed positions, page growth, rejection,
  requested features and changing output requirements. Preserve eager sigmoid-gate
  arithmetic under fusion. Stock MLX-VLM remains the independent model reference.


#### `MODEL:QWEN35:MAG:LAYERWISE`

- **Implementation:** Advance the configured hybrid layer sequence and state, returning requested logits/features.
  Python builds the layer graph each forward; only recurrent regions compile. Compatible rows
  share an arena with independent positions. Ordinary residual and normalization order is
  preserved.
- **Reference / validation:** Separately loaded stock MLX-VLM target; use MLX-LM as a second control with
  positional/numerical conventions reconciled. Compare layer residuals, logits and logical
  state, then free generation and changing batches.


### `MODEL:QWEN35.ATTENTION`

**Contract.** Apply Qwen gated attention: Q/output-gate and K/V projections, Q/K normalization, rotary
transforms, history append, attention, output gating and projection.

**Parameters.** Architecture: hidden width `h`, `a=h_q*d`, `k=h_kv*d`, encoded projection/norm parameters and
rotary semantics. Workload: `m=bq`, history lengths, positions and required KV persistence.

**Composition.** Bind [projection/local equations](../../performance/derivations/neural.md#projection-notation), the [shared attention
contract](../composability.md#modelattention) and [KV
append](../../engine/components.md#kvappend). The extra `a` projected values are the output
gate. New KV may feed attention internally while meeting future-state obligations; count it
once.

```text
D_QA = JOIN(P(m,h,2a,W_q_gate), P(m,h,k,W_k), P(m,h,k,W_v),
  Q/K norm + rotary, D_ATTN, sigmoid/output gate, P(m,a,h,W_o), required new KV)
F_projections = m[(2a+2k)(2h-1) + h(2a-1)]
new KV logical bytes = m*h_kv*d*(s_k+s_v)
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:QWEN35.ATTENTION/EXEC` | Elapsed seconds for `u=m` mixer inputs through output and required KV readiness. | `L(D_QA)`. |

**Implementations and controls.**

#### `MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS`

- **Implementation:** Project Q plus output gate, K and V separately; normalize Q/K, apply paired rotary transforms,
  append KV, run the selected attention child, then gate and project its output. MLX operation
  composition. The default paged child reuses each KV read across two query heads for
  single-token execution when the head geometry permits; wider query blocks retain
  per-head execution.
- **Reference / validation:** Stock Qwen gated attention with matched weights and logical history. Compare prepared Q/K/V,
  gate, output and appended state; use independently computed rotary values to diagnose upstream
  convention differences.


### `MODEL:QWEN35.RECURRENCE`

**Contract.** Apply projected convolutional preparation, gated-delta state update and gated normalized
output projection; preserve accepted-boundary reconstruction obligations.

**Parameters.** Architecture: `h`, key heads/width `h_k,d_k`, value heads/width `h_v,d_v`, kernel width `z`,
actual retained convolution window `z-1`, encodings and state dtype. Workload: `m=bq`, initial
state and requested outputs/checkpoints.

**Composition.** Bind [projections/local equations](../../performance/derivations/neural.md#projection-notation), direct depthwise
convolution and [shared recurrence](../composability.md#modelgated_delta). Prepared tensors
may fuse; checkpoint requirements do not mandate an image after every token. Decay is
`exp(-exp(log_rate)*softplus(decay_projection+time_bias))`; constant terms may be prepared
once.

```text
a=h_k*d_k; v=h_v*d_v; c=2a+v
D_QR = JOIN(P(m,h,c,W_qkv), P(m,h,v,W_gate),
  P(m,h,h_v,W_beta), P(m,h,h_v,W_decay), depthwise_conv(c,z), SiLU,
  Q/K norms/scales, decay/beta transforms, D_DELTA,
  gated output norm, P(m,v,h,W_o), required convolution/recurrent state)
F_projections = m[(c+v+2h_v)(2h-1) + h(2v-1)]
F_conv = m*c*(2z-1)                  conditional direct convolution
matrix bytes = b*h_v*d_v*d_k*s_state
convolution bytes = b*(z-1)*c*s_conv
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:QWEN35.RECURRENCE/EXEC` | Elapsed seconds for `u=m` recurrent-mixer inputs through output/state readiness. | `L(D_QR)`; alternative chunked algorithms retain the common data bound unless separately derived. |

**Implementations and controls.**

#### `MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION`

- **Implementation:** Separate QKV, output-gate and decay/beta projections feed convolution, normalization and a
  replaceable gated-delta update, followed by gated normalization/output projection. The tensor
  region compiles; state staging and transaction effects remain outside it. The default update
  is the shared `MODEL:GATED_DELTA:MAG:FUSED_UPDATE`; its upstream alternative is
  `MODEL:GATED_DELTA:LM:STANDARD`.
- **Reference / validation:** Complete MLX-LM recurrent block and an independent gated-delta equation oracle. Compare
  prepared inputs, convolution history, matrix state and output across one/many inputs, batching
  and accepted-prefix restoration.


### `MODEL:QWEN35.FEEDFORWARD`

**Contract.** Produce the configured dense or routed/shared-expert feedforward output, preserving Qwen
routing, normalization and gating semantics.

**Parameters.** Architecture: `h,f`, or expert count `E`, top-k `t`, expert widths and shared width
`f_shared`; encoded tensors, SiLU and routing normalization policy. Workload: `m`, assignments
or declared route distribution/range and residency.

**Composition.** Dense uses `MLP`; routed uses the [shared expert contract](../composability.md#modelexperts)
plus router/reduction/shared branch. [Projection/local
equations](../../performance/derivations/neural.md#projection-notation) supply norms, sigmoid and softmax. Union expert
weights, count all row/expert evaluations; selected routes are workload conditioning, not an
architecture constant.

```text
D_QFF_dense = MLP(m,h,f)                         SiLU gate
D_QFF_routed = JOIN(P(m,h,E,W_router), full-score softmax, top-k,
  optional selected-weight renormalization, D_EXPERTS, weighted reduction,
  MLP(m,h,f_shared), P(m,h,1,W_shared_gate), sigmoid + gated branch addition)
F_weighted_reduction = m*h*(2t-1)                conventional scalar model
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:QWEN35.FEEDFORWARD/EXEC` | Elapsed seconds for `u=m` feedforward rows through output readiness. | `L(D_QFF)` for the configured branch. |

**Implementations and controls.**

#### `MODEL:QWEN35.FEEDFORWARD:LM:DENSE`

- **Implementation:** Pass through the bound upstream gated dense MLP.
- **Reference / validation:** Independent MLX-VLM MLP and explicit gate/up/activation/down equations with the same weights;
  compare output before the enclosing residual.


#### `MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED`

- **Implementation:** Router softmax, top-k and optional renormalization; selected expert evaluation; weighted
  reduction plus a sigmoid-gated shared MLP. Uses the shared expert child; routing and
  combination remain separate MLX operations.
- **Reference / validation:** Complete upstream routed/shared MLP. Compare assignments, probabilities, selected outputs and
  final sum with representative routing patterns.


### `MODEL:QWEN35.READOUT`

**Contract.** Normalize final residuals and project requested rows to vocabulary logits, preserving
tied/separate head semantics.

**Parameters.** Architecture: hidden/vocabulary widths `h,V`, encoded head and norm parameters. Workload:
`m_out`, requested logits and residency.

**Composition.** `D_QHEAD=JOIN(final RMS norm,P(m_out,h,V,W_vocab))` using [projection
equations](../../performance/derivations/neural.md#projection-notation). Union tied vocabulary/embedding storage at the
target boundary. Omit absent readout; it does not receive a 100% score.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:QWEN35.READOUT/EXEC` | Elapsed seconds for `u=m_out` requested logit rows through readiness. | `L(D_QHEAD)`. |

**Implementations and controls.**

#### `MODEL:QWEN35.READOUT:MAG:STANDARD`

- **Implementation:** Final upstream norm followed by tied embedding projection or the separate language head;
  compute logits only when requested.
- **Reference / validation:** Stock final norm/head from identical residuals; check tied weights, precision and
  requested-output behavior independently of the transformer.


### `STATE:QWEN35`

**Contract.** Combine paged attention and recurrent/convolution state into one checkpoint/transaction with
independent row progress and valid accepted-boundary restoration.

**Parameters.** Architecture: attention producer and recurrent/convolution geometry, encodings and state
formats. Workload: required retained positions `n_ai`, shared histories, live checkpoints,
advanced/accepted positions, observation instant and memory budget.

**Composition.** Join [KV storage](../../engine/components.md#kvstore) with [recurrent
state](../../engine/components.md#staterecurrent). [Live-union and restoration
formulas](../../performance/derivations/state.md#required-live-union) remove aliases and permit legal reconstruction.
Advance/snapshot creation and transient peaks remain enclosing-workload costs and constraints.
More images may improve restoration while increasing retained memory.

```text
M_live = sum_(attention a,row i) n_ai*h_kva*d_a*(s_ka+s_va)
       + sum_(recurrent r) b*[h_vr*d_vr*d_kr*s_state + (z_r-1)*c_r*s_conv]
M_min = required unique materialized union, not M_live times checkpoint count
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `STATE:QWEN35/MEM` | Retained physical bytes at the specified lifecycle boundary, counting shared backing once. | `M_live` below is the initial lower bound when those representations must be materialized; add only proven checkpoint information. Efficiency `100*M_min/M`. |
| `STATE:QWEN35/RESTORE` | Seconds from the specified advanced state to accepted-state readiness, including deferred repair before next use. | `L_restore(initial,accepted,obligations,budget)` from [restoration cases](../../performance/derivations/state.md#restoration-cases); efficiency `100*L/T`. |

**Implementations and controls.**

#### `STATE:QWEN35:MAG:HYBRID`

- **Implementation:** Combine paged attention history and recurrent images into one logical checkpoint. Append or
  tentatively advance both, preserve row independence and resolve each accepted boundary without
  exposing rejected state.
- **Reference / validation:** Independently advanced upstream KV/recurrent caches and explicit prefix replay; compare
  logical contents after branching, restore and unequal verification acceptance, including
  budget and lifetime failures.


### `MODEL:QWEN35.MTP`

**Contract.** Consume draft tokens and previous hidden conditioning through the attached prediction head,
with independent native state and borrowed target vocabulary.

**Parameters.** Architecture: hidden width `h`, attached layer count/geometry, encodings and borrowed
vocabulary identity. Workload: `m` draft inputs, conditioning, requested outputs, native cache
histories and residency; proposal depth is separate from attached layer count.

**Composition.** Join shared embedding, two input norms, combination projection, attached layers, output norm
and requested head. The current loader sets `full_attention_interval=1`: each attached layer
uses Qwen gated full [attention](#modelqwen35attention), norms/residuals and configured
[feedforward](#modelqwen35feedforward), with [native
checkpoints](generic-mlx-vlm.md#statecheckpoints). Target recurrence is absent.
Accepted-output throughput belongs to
[speculation](../../engine/components.md#generationspeculation).

```text
D_MTP = JOIN(D_EMBED, two input norms, P(m,2h,h,W_combine),
  {input_norm_j,D_QA_j,residual_j,ff_norm_j,D_QFF_j,residual_j}_attached_j,
  output norm, requested vocabulary projection, required native state)
F_combine = m*h*(4h-1)
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:QWEN35.MTP/EXEC` | Elapsed seconds for `u=m` draft inputs through requested logits/features/state. | `L(D_MTP)`; repeated dependent predictions bind one invocation per draft step. |

**Implementations and controls.**

#### `MODEL:QWEN35.MTP:MAG:CONDITIONED`

- **Implementation:** Combine normalized token embedding and previous hidden conditioning, run attached MLX-LM
  decoder layers with native caches, then normalize and project through the borrowed target
  vocabulary. The head owns its state; proposal acceptance and repair belong to
  [speculation](../../engine/speculation.md).
- **Reference / validation:** Matching upstream MTP drafter with identical head weights, quantization, conditioning and
  logical positions. Compare hidden outputs, logits and cache transitions independently before
  testing full speculative generation.


## Qualification

Prepared region controls capture actual layer inputs in an untimed forward, then use
`performance.benchmarks.regions` to compare the selected operation with a borrowed-weight
upstream control. State and outputs are checked independently. Parent measurements run
without capture wrappers or retained diagnostic intermediates.

Model replay, generated continuation, batched prefill, hybrid restoration and MTP use the
corresponding functions in `performance.benchmarks`. Protect dense/routed variants, long
contexts and multi-input execution. Each benchmark records its precise numerical contract;
reference throughput is a comparison and never a ceiling.
