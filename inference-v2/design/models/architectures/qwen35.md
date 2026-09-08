# Qwen3.5-family hybrid models

## Scope

**The model composes owned hybrid blocks and upstream operations. Resident
single-input execution compiles their tensor transitions together. Both execution paths
use one architecture equation for the residual stream, features and readout; state
preparation and publication remain outside compilation.** This covers the
accepted Qwen3.5-family layouts, including compatible Qwen3.6 artifacts, dense
or routed feedforward, and converted affine weights.

MLX-VLM supplies the full conditional configuration, language parameters, vision
encoder, and explicitly adapted rotary calculation. Qualified MLX-LM primitives
remain usable beneath owned operations and the attached drafter. The complete
model binds [input semantics](../inputs.md) alongside its decoder. Source attribution
follows [component identification](../../components.md).

Owned operations follow [model composability](../composability.md) and
[kernel construction](../../kernels.md). Qwen owns recurrent/attention selection,
routing, SwiGLU and residual ordering; shared numerical implementations own tile
execution. Generated specializations remain beneath these semantic components.

## Assembly

```text
MODEL:QWEN35:MAG:LAYERWISE
├── Optional image preparation · MODEL:QWEN35.PREPARATION:MAG:IMAGES
├── Optional image encoder/projector · MODEL:QWEN35.VISION:VLM:STANDARD
├── Embedding · MODEL:EMBEDDING:MAG:RESIDENT
├── Repeated hybrid layer
│   ├── Mixer (selected per layer)
│   │   ├── Attention · MODEL:QWEN35.ATTENTION:MAG:GROUPED_PROJECTIONS
│   │   │   └── MODEL:ATTENTION:MAG:PAGED
│   │   │       └── Fallback · MODEL:ATTENTION:MAG:GATHERED
│   │   └── Recurrence · MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   │       └── Update · MODEL:GATED_DELTA:MAG:FUSED_UPDATE
│   └── Feedforward (selected by configuration)
│       ├── Dense · MODEL:QWEN35.FEEDFORWARD:MAG:DENSE
│       └── Routed · MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│           └── Experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
├── Readout · MODEL:QWEN35.READOUT:MAG:STANDARD
├── Eligible single-input execution · MODEL:QWEN35:MAG:RESIDENT_COMPILED
│   └── Shares the embedding, layer components and readout above
└── Hybrid state · STATE:QWEN35:MAG:HYBRID
    ├── KV · KV:STORE:MAG:PAGED
    └── Recurrent checkpoints · STATE:RECURRENT:MAG:CHECKPOINTED

Optional attached drafter · MODEL:QWEN35.MTP:MAG:CONDITIONED
├── Borrowed target embedding · MODEL:EMBEDDING:MAG:RESIDENT
└── Draft state · STATE:CHECKPOINTS:MAG:NATIVE
```

## Component definitions

Image preparation supplies projected embedding spans and three-axis coordinates.
Causal image spans can cross prompt chunks; only intersecting feature slices are
injected. Physical KV positions still count decoder inputs. Continued rotary positions
retain their model-defined offset through batching, restore, verification and repair.
Generated single-token execution uses the same compiled decoder with explicit rotary
positions. The paired MTP head consumes aligned successor embeddings during prompt
catch-up, including image features, together with the preceding target residual.

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
  expert math are shared with layerwise execution. A continuous residual stream fuses
  each update with the following RMS normalization, including the final readout norm.
  Wider inputs and unsupported
  storage/operator compositions use the layerwise implementation below.
- **State boundary:** State storage prepares pinned, read-only history and bounded
  writable append buffers. Tensor execution returns only changed buffer versions; state
  storage installs the complete validated result as a tentative boundary. Existing
  transactions own acceptance, rejection and completion lifetime. Compilation
  neither allocates physical pages nor grants writes to retained prefixes.
- **Specialization:** Positions and physical addresses are tensor operands. Cache
  specialization follows batch size, physical capacity, requested outputs and the
  attention launch horizon. Bucket page-map width independently of allocator slabs
  and append capacity; retain at most four compiled geometries.
  Actual positions govern causal visibility. Combined append backing exposes contiguous
  K/V views and requires one bounded write per producer.
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

#### `MODEL:QWEN35.ATTENTION:MAG:GROUPED_PROJECTIONS`

- **Implementation:** Pack compatible Q/gate, K and V projections into one allocation and operation.
  Narrow execution fuses unpacking, Q/K normalization and paired text rotary transforms;
  append KV, run the selected attention child, then gate and project its output.
  The default paged child reuses each KV read across two query heads for
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

- **Implementation:** Compatible QKV, output-gate and decay/beta projections share one packed operation.
  Narrow execution fuses convolution, normalization and gate preparation before the
  replaceable gated-delta update, followed by upstream gated normalization and output projection. The tensor
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

#### `MODEL:QWEN35.FEEDFORWARD:MAG:DENSE`

- **Implementation:** Single-row affine execution fuses gate/up projections and SiLU
  activation, then applies the bound down projection. Other geometries use the bound
  upstream gated MLP. Both consume the same parameter tensors.
- **Reference / validation:** Independent MLX-VLM MLP and explicit gate/up/activation/down equations with the same weights;
  compare output before the enclosing residual.


#### `MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED`

- **Implementation:** Compatible router/shared-gate projections share one packed operation.
  Narrow execution fuses softmax, top-k, optional renormalization and the shared gate.
  Compatible resident routed and shared experts execute together, including their
  weighted sum. The architecture supplies routing and the shared coefficient; the
  expert kernels preserve projection, activation and BF16 accumulation boundaries.
  Other geometries and streamed experts retain separate shared execution.
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

### `MODEL:QWEN35.PREPARATION`

**Contract.** Translate ordered source images and the artifact's chat representation
into bounded prepared pixels, image geometry, and an expanded decoder layout. The
processor identity includes the artifact, preparation implementation and pinned
processor behavior. This is CPU input interpretation, separate from neural encoding.

`MODEL:QWEN35.PREPARATION:MAG:IMAGES` applies the artifact's PIL image processor,
validates its output schema, and expands one placeholder per image into the required
soft-token span. It exposes no device cache or generation-loop state.

### `MODEL:QWEN35.VISION`

**Contract.** Encode prepared image patches and project them into decoder-width
features without autoregressive state. Independent images remain isolated when
packed into one upstream encoder invocation. Output slices retain their own
completion and allocation obligations.

`MODEL:QWEN35.VISION:VLM:STANDARD` uses the pinned VLM vision tower and projector.
The captured parameter record contains the actual weights and shares bounded feature
retention through [model inputs](../inputs.md). `MODEL:QWEN35.VISION/EXEC` observes
encoder/projector execution through completed outputs. It is an opaque upstream
workload record; it supplies no fabricated analytical efficiency denominator.
