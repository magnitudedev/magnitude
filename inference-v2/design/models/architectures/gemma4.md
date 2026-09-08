# Gemma 4

## Scope

**An owned assembly that preserves Gemma's normalization, branching and KV-sharing
semantics while making each computational region independently comparable.**
Layerwise and resident execution share the architecture equation; compilation and
state publication bind that equation without defining a second model loop.
The current binding accepts supported Gemma 4 text configurations with affine
weights, dense/routed branches and optional per-layer inputs. MLX-VLM supplies
configuration, parameter containers and the independent model reference.

IDs follow [component identification](../../components.md). The complete model binds
[image input semantics](../inputs.md) alongside the decoder. Owned execution and the
generic upstream adapter require separate numerical qualification.

Image features replace scaled token embeddings before per-layer projection. Media
positions use the declared zero vocabulary input in the per-layer branch. With vision
bidirectionality enabled, each image's soft-token span is indivisible: local attention
sees the complete image block while retaining its lower sliding-window bound; global
attention stays causal. Rectangular queries after a cached text prefix obey the same
rule. Producer sharing remains unchanged. After image consumption, ordinary text
continuation uses the resident compiled path without image operands.

Owned operations follow [model composability](../composability.md) and
[kernel construction](../../kernels.md). Gemma retains its normalization, GeGLU,
scaling, input branches and producer-sharing semantics while reusing compatible
contraction and attention implementations. Kernel plans do not redefine that assembly.

## Assembly

```text
MODEL:GEMMA4:MAG:LAYERWISE
├── Optional image preparation · MODEL:GEMMA4.PREPARATION:MAG:IMAGES
├── Optional image encoder/projector · MODEL:GEMMA4.VISION:VLM:STANDARD
├── Resident decode · MODEL:GEMMA4:MAG:RESIDENT_COMPILED
├── Embedding · MODEL:EMBEDDING:MAG:RESIDENT
├── Optional input preparation · MODEL:GEMMA4.INPUTS:MAG:PER_LAYER
├── Repeated layer
│   ├── Attention · MODEL:GEMMA4.ATTENTION:MAG:SHARED_KV
│   │   ├── KV producer or shared producer reference · MODEL:GEMMA4.KV:MAG:PRODUCER
│   │   └── MODEL:ATTENTION:MAG:PAGED
│   │       └── Fallback · MODEL:ATTENTION:MAG:GATHERED
│   ├── Feedforward · MODEL:GEMMA4.FEEDFORWARD:MAG:BRANCHED
│   │   ├── Dense · MODEL:GEMMA4.MLP:MAG:GEGLU
│   │   └── Optional expert branch · MODEL:GEMMA4.EXPERT_BRANCH:MAG:ROUTED
│   │       └── Experts · MODEL:EXPERTS:MAG:RESIDENT_GATHERED
│   └── Optional input application · MODEL:GEMMA4.INPUTS:MAG:PER_LAYER
├── Readout · MODEL:GEMMA4.READOUT:MAG:SOFTCAPPED
└── KV state · KV:STORE:MAG:PAGED
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

### `MODEL:GEMMA4`

**Contract.** Consume text through Gemma’s configured normalized, branched layer graph and shared KV
producers, returning requested logits/features and valid state.

**Parameters.** Architecture: layer geometry and producer map, local/global windows, enabled routed/input
branches, weights and scales. Workload: `m=bq`, row histories, output/feature requirements and
residency.

**Composition.** Join shared embedding, optional [input preparation/application](#modelgemma4inputs), each
[attention](#modelgemma4attention)/[feedforward](#modelgemma4feedforward) region and requested
[readout](#modelgemma4readout). Count each physical KV producer once for storage/writes and
each consumer’s mathematical attention. Repeated input-component IDs do not repeat full
preparation.

```text
D_GEMMA = JOIN(scaled D_EMBED, optional input preparation,
  {input_norm_j,D_GA_j,post_attention_norm_j,residual_j,
   D_GFF_j,residual_j,optional input application_j,residual_if_inputs_j,
   layer_scale_j}_j, requested readout/features, required final producer state)
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4/EXEC` | Elapsed seconds for `u=m` consumed inputs through requested outputs/state readiness. | `L(D_GEMMA)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4:MAG:LAYERWISE`

- **Implementation:** Scale embedding, prepare optional layer inputs, run configured layers with their
  norms/residuals and layer scalars, then read out. Python builds the graph each forward; there
  eligible resident single-token forwards use the compiled composition below.
- **Reference / validation:** Independently loaded stock MLX-VLM Gemma. Compare layer residuals, requested features, logits
  and logical KV with matched inputs and weights; exercise dense/routed variants and optional
  features explicitly.

#### `MODEL:GEMMA4:MAG:RESIDENT_COMPILED`

- **Implementation:** Compile the same neural bindings into a tensor transition.
  Each KV producer writes one bounded append image; later shared readers consume
  that updated image with their own query and window. State preparation, validation,
  installation and resource lifetime remain outside compilation. Wide or streamed
  execution uses the layerwise path after sealing live append images.
- **Reference / validation:** The layerwise composition and independent MLX-VLM
  execution, with fixed weights, inputs and logical history. Cover shared readers,
  unequal row lengths, rejected tokens, append rollover, forks and feature-only output.


### `MODEL:GEMMA4.ATTENTION`

**Contract.** Project/normalize/rotate Q, optionally produce KV, read the declared producer with the layer’s
window and project attended output.

**Parameters.** Architecture: layer `j` widths `a_j=h_qj*d_j`, `k_j=h_kvj*d_j`, hidden width `h`, source
`p(j)`, window `w_j`, weights and transforms. Workload: `m=bq`, histories, positions and
source residency.

**Composition.** Use [KV production](#modelgemma4kv) only when `p(j)=j`, plus [shared
attention](../composability.md#modelattention). Shared source consumers keep independent
query/attention arithmetic. Union visibility across readers at the parent; matching source
geometry and semantics remain required.

```text
D_GA_j = JOIN(P(m,h,a_j,W_q), Q norm/rotary,
  [D_GKV_j if p(j)=j], ATTN(Q_j,KV_p(j),window=w_j), P(m,a_j,h,W_o))
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.ATTENTION/EXEC` | Elapsed seconds for `u=m` attention-block rows through output and required state. | `L(D_GA_j)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4.ATTENTION:MAG:SHARED_KV`

- **Implementation:** Normalize/project Q and apply the layer's rotary transform; optionally invoke its KV producer;
  read that source with the declared window, run the attention child and output projection.
  Readers must match the producer's geometry and attention semantics. Uses MLX operations around
  the child.
- **Reference / validation:** Stock attention from identical hidden input and logical source KV. Compare Q, visible history,
  attended values and projected output across local/global layers, different row lengths and
  shared-reader layers.

### `MODEL:GEMMA4.KV`

**Contract.** Produce transformed K/V and append once to a shared source; K=V shares the raw projection,
preserving the distinct key/value transforms.

**Parameters.** Architecture: `h`, `k_j=h_kvj*d_j`, `e_j=1` for a separate V projection or `0` for raw K=V,
encoded tensors and transforms. Workload: `m`, positions and required visibility/persistence.

**Composition.** Join [projections and local equations](../../performance/derivations/neural.md#projection-notation) with [KV
append](../../engine/components.md#kvappend). Readers have no producer invocation. Raw
projection sharing does not alias differently normalized/rotated output branches.

```text
D_GKV_j = JOIN(P(m,h,k_j,W_k), e_j*P(m,h,k_j,W_v),
  K/V norms, K rotary, required new KV)
F_projections = m*(1+e_j)*k_j*(2h-1)
new KV bytes = m*h_kvj*d_j*(s_k+s_v)
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.KV/EXEC` | Elapsed seconds for `u=m` new KV positions through required source readiness. | `L(D_GKV_j)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4.KV:MAG:PRODUCER`

- **Implementation:** Project raw K and, when needed, V; normalize branches, apply key rotary transform and append
  once to the source's paged history. K=V shares the raw projection, not the differently
  transformed key/value outputs.
- **Reference / validation:** Upstream preparation and cache append with identical hidden input/positions. Compare K/V and
  stored history; verify readers add no duplicate writes and branching preserves earlier
  prefixes.

### `MODEL:GEMMA4.FEEDFORWARD`

**Contract.** Apply Gemma input normalization, required dense branch, optional separately normalized routed
branch, combination and final normalization.

**Parameters.** Architecture: dense/routed configuration and each child’s weights/geometry/norms. Workload:
`m`, routes or their explicit distribution/range and residency.

**Composition.** `D_GFF=JOIN(input_norm,D_GMLP,optional_dense_norm,optional_D_GEXPERT,branch_add_if_routed,output_norm)`.
[Dense](#modelgemma4mlp) and [expert](#modelgemma4expert_branch) work coexist on routed
layers. Use [local equations](../../performance/derivations/neural.md#projection-notation); fusion can remove traffic
but not required branches.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.FEEDFORWARD/EXEC` | Elapsed seconds for `u=m` feedforward rows through output readiness. | `L(D_GFF)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4.FEEDFORWARD:MAG:BRANCHED`

- **Implementation:** Apply the architecture's input norm and dense child; on routed layers apply the dense branch
  norm and add the expert branch; apply the final feedforward norm. Preserve branch-specific
  normalization order.
- **Reference / validation:** Complete upstream Gemma feedforward from matching hidden inputs. Check branches separately,
  then their sum and final normalization.

### `MODEL:GEMMA4.MLP`

**Contract.** Compute gate/up GeGLU and down projection using Gemma’s declared approximate activation and
numerical behavior.

**Parameters.** Architecture: `h,f`, encoded gate/up/down tensors, bias/activation settings. Workload: `m` and
boundary residency.

**Composition.** `D_GMLP=MLP(m,h,f)` with [approximate GeGLU](../../performance/derivations/neural.md#projection-notation). Packed
projections and fused epilogues can share input reads and eliminate intermediate
materialization.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.MLP/EXEC` | Elapsed seconds for `u=m` dense MLP rows through output readiness. | `L(D_GMLP)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4.MLP:MAG:GEGLU`

- **Implementation:** Separate gate/up projections, approximate GeGLU activation
  and down projection using MLX operations and upstream weight modules.
- **Reference / validation:** Upstream dense MLP and explicit projection/activation equations; compare outputs before
  surrounding norms and residuals.

### `MODEL:GEMMA4.EXPERT_BRANCH`

**Contract.** Normalize router input, select top-k scores, normalize selected scores, apply expert scales
and combine independently normalized expert outputs.

**Parameters.** Architecture: `h,E,t`, expert geometry, router scale/epsilon, per-expert scales and norms.
Workload: `m`, assignments or distribution/range, outputs and residency.

**Composition.** `D_GEXPERT=JOIN(router_norm/scale,P(m,h,E,W_router),top_k,selected_score_softmax,expert_scales,expert_input_norm,D_EXPERTS,weighted_sum,output_norm)`.
[Shared experts](../composability.md#modelexperts) retain assignment semantics. [Local
equations](../../performance/derivations/neural.md#projection-notation) preserve Gemma’s selected-score softmax,
distinct from Qwen’s full-score softmax.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.EXPERT_BRANCH/EXEC` | Elapsed seconds for `u=m` routed-branch rows through normalized output. | `L(D_GEXPERT)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4.EXPERT_BRANCH:MAG:ROUTED`

- **Implementation:** Normalize router input, select top-k scores, softmax selected scores and apply per-expert
  scales. The shared expert child evaluates and reduces selected experts from the separately
  normalized input; the branch normalizes the combined result.
- **Reference / validation:** Upstream router and routed branch. Compare expert selection, weights, scaled outputs and
  branch result with real routing distributions and ties.

### `MODEL:GEMMA4.INPUTS`

**Contract.** Prepare auxiliary layer inputs once and apply each configured gate/projection/norm
contribution to its layer.

**Parameters.** Architecture: hidden width `h`, per-layer width `p`, consumer count `N`, auxiliary
embedding/projection tensors and scales. Workload: `m`, token identities, required consuming
layers and residency.

**Composition.** Preparation joins `EMBED(m,N*p,rows)`, `P(m,h,N*p,W_prepare)`, per-layer-width norm and scaled
combination. Each application joins gate `P(m,h,p,W_gate_j)`, approximate GeGLU/product,
`P(m,p,h,W_apply_j)` and output norm. Use [neural
equations](../../performance/derivations/neural.md#projection-notation); logical prepared workspace is not compulsory
DRAM traffic. Preparation/application probes are subregions of one dimension.

```text
D_GINPUTS = JOIN(preparation_once, {application_j}_consumers)
F_projections = m*[N*p*(2h-1) + sum_j(p*(2h-1)+h*(2p-1))]
logical prepared workspace = m*N*p*s
```

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.INPUTS/EXEC` | Elapsed seconds for `u=m` inputs through preparation and all required layer contributions. | `L(D_GINPUTS)`; absent features have no invocation or score. |

**Implementations and controls.**

#### `MODEL:GEMMA4.INPUTS:MAG:PER_LAYER`

- **Implementation:** Prepare auxiliary embedding and projected normalized layer inputs with their scales; at each
  layer apply its gate, projection and norm before the residual addition. Preparation and
  application are observable boundaries of this one feature component; neither stage needs to
  execute when absent.
- **Reference / validation:** Upstream input preparation and per-layer contribution from identical token embeddings and
  hidden states; test configured scales and geometry.

### `MODEL:GEMMA4.READOUT`

**Contract.** Apply final norm, tied/separate vocabulary projection and configured optional tanh soft cap to
requested logits.

**Parameters.** Architecture: `h,V`, head/norm tensors and positive optional cap. Workload: `m_out`, output
requirements and residency.

**Composition.** `D_GHEAD=JOIN(final_norm,P(m_out,h,V,W_vocab),optional_soft_cap)` using [local
equations](../../performance/derivations/neural.md#projection-notation). Union tied embedding/head weights. The cap need
not be an extra external pass.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:GEMMA4.READOUT/EXEC` | Elapsed seconds for `u=m_out` requested logit rows through readiness. | `L(D_GHEAD)`. |

**Implementations and controls.**

#### `MODEL:GEMMA4.READOUT:MAG:SOFTCAPPED`

- **Implementation:** Final norm, tied or separate vocabulary projection and configured optional tanh soft cap.
  Return only requested logits.
- **Reference / validation:** Upstream readout from the same residual, covering tied weights and configurations with and
  without a cap.

## Qualification

Current scores follow the [assessment rules](../../performance.md#stable-compositions-and-evidence).

Existing components have tests and diagnostic comparisons, but not every named
boundary yet has a dedicated independently qualified benchmark function. The tree
identifies where those measurements belong; it does not claim they already exist.

Historical custom 16K comparisons left numerical parity open. Native Gemma
long-context qualification applies to the generic adapter, not this assembly.
Test KV-sharing, K=V, window crossings, per-layer inputs, batching and prefix
restoration before broad claims, and retain the selected component IDs with results.

Evidence: `sessions/26-09-06/evidence/cycle-005/comparison-16384.json` and
`sessions/26-09-06/evidence/cycle-006/comparison.json`, relative to the monorepo root.

Further kernel specialization must preserve dense/routed branches, optional inputs
and KV sharing. Media tensors remain separate from owned text parameter loading.

### `MODEL:GEMMA4.PREPARATION`

**Contract.** Translate ordered source images and the artifact's chat representation
into bounded prepared patches, patch coordinates, soft-token counts and an expanded
decoder layout. This CPU interpretation is distinct from per-layer decoder inputs
and from vision encoding.

`MODEL:GEMMA4.PREPARATION:MAG:IMAGES` applies the artifact's PIL image processor,
validates its exact output schema and expands the image boundary markers and soft
tokens. Processor identity covers the artifact and preparation behavior.

### `MODEL:GEMMA4.VISION`

**Contract.** Encode prepared patches and project them into decoder-width image
features. Compatible patch geometries can share an encoder batch with independent
row outputs. Encoding has no autoregressive state and does not choose decoder
attention visibility.

`MODEL:GEMMA4.VISION:VLM:STANDARD` composes the pinned VLM vision tower and image
projector. Its captured parameter record contains both modules and shares bounded
feature retention through [model inputs](../inputs.md). `MODEL:GEMMA4.VISION/EXEC`
observes encoder/projector execution through completed outputs. It is an opaque
upstream workload record with no fabricated analytical efficiency denominator.
