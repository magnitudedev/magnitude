# Gemma 4

## Scope

**An owned assembly that preserves Gemma's normalization, branching and KV-sharing
semantics while making each computational region independently comparable.**
The current binding accepts supported Gemma 4 text configurations with affine
weights, dense/routed branches and optional per-layer inputs. MLX-VLM supplies
configuration, parameter containers and the independent model reference.

IDs follow [component identification](../../components.md). The program exposes
text inputs and residual features, not media execution. Current owned execution
must not inherit qualification from the generic upstream adapter.

## Assembly

```text
MODEL:GEMMA4:MAG:LAYERWISE
├── MODEL:EMBEDDING:MAG:RESIDENT
├── MODEL:GEMMA4.INPUTS:MAG:PER_LAYER                when configured
├── repeated layer assembly: attention → residual → feedforward → residual
│   │                       → optional input contribution → layer scaling
│   ├── MODEL:GEMMA4.ATTENTION:MAG:SHARED_KV
│   │   ├── MODEL:GEMMA4.KV:MAG:PRODUCER             producer layers only
│   │   └── MODEL:ATTENTION:MAG:PAGED
│   │       └── MODEL:ATTENTION:MAG:GATHERED         fallback
│   │           └── MODEL:ATTENTION:MLX:DENSE
│   ├── MODEL:GEMMA4.FEEDFORWARD:MAG:BRANCHED
│   │   ├── MODEL:GEMMA4.MLP:MAG:GEGLU
│   │   └── MODEL:GEMMA4.EXPERT_BRANCH:MAG:ROUTED    routed layers only
│   │       └── MODEL:EXPERTS:MAG:RESIDENT_GATHERED
│   └── MODEL:GEMMA4.INPUTS:MAG:PER_LAYER            consumes prepared inputs
└── MODEL:GEMMA4.READOUT:MAG:SOFTCAPPED
```

Shared embedding, attention and expert IDs resolve to the
[shared component definitions](../composability.md#shared-component-definitions).
The per-layer input component prepares shared input once and contributes at each
layer; the repeated ID does not mean repeated full preparation. KV consumers use
an earlier producer's stored history through explicit connections, not copied
subtrees. Physical state is an input dependency of the neural components.

## Components

### `MODEL:GEMMA4:MAG:LAYERWISE`

- **Contract / implementation:** Scale embedding, prepare optional layer inputs, run
  configured layers with their norms/residuals and layer scalars, then read out.
  Python builds the graph each forward; there is no whole-model compiled step.
- **References / tests:** Independently loaded stock MLX-VLM Gemma. Compare layer
  residuals, requested features, logits and logical KV with matched inputs and
  weights; exercise dense/routed variants and optional features explicitly.
- **Performance / bounds:** Compose selected children with the actual normalization,
  residual and scaling sequence. Add exposed graph/encoding and state costs. Count
  unique KV producers for writes/storage, each consumer for reads. Derive prefill
  and decode bounds from the configured global/local layer mix, not layer count alone.

### `MODEL:GEMMA4.ATTENTION:MAG:SHARED_KV`

- **Contract / implementation:** Normalize/project Q and apply the layer's rotary
  transform; optionally invoke its KV producer; read that source with the declared
  window, run the attention child and output projection. Readers must match the
  producer's geometry and attention semantics. Uses MLX operations around the child.
- **References / tests:** Stock attention from identical hidden input and logical
  source KV. Compare Q, visible history, attended values and projected output across
  local/global layers, different row lengths and shared-reader layers.
- **Performance / bounds:** Q/output projection work plus child attention reads and
  computation; producer cost only for writer layers. Shared KV does not make later
  reads free. Local visibility is window-bounded; global visibility grows with context.
  Producer-aware projection packing and preparation fusion are unimplemented opportunities.

### `MODEL:GEMMA4.KV:MAG:PRODUCER`

- **Contract / implementation:** Project raw K and, when needed, V; normalize branches,
  apply key rotary transform and append once to the source's paged history. K=V
  shares the raw projection, not the differently transformed key/value outputs.
- **References / tests:** Upstream preparation and cache append with identical hidden
  input/positions. Compare K/V and stored history; verify readers add no duplicate
  writes and branching preserves earlier prefixes.
- **Performance / bounds:** Count one or two input projections as configured,
  normalization/rotary passes and new KV bytes. Bound projection and write work by
  arithmetic/bandwidth with their dependencies. Sharing eliminates duplicate production
  and storage; it does not eliminate the consumer attention work.

### `MODEL:GEMMA4.FEEDFORWARD:MAG:BRANCHED`

- **Contract / implementation:** Apply the architecture's input norm and dense child;
  on routed layers apply the dense branch norm and add the expert branch; apply the
  final feedforward norm. Preserve branch-specific normalization order.
- **References / tests:** Complete upstream Gemma feedforward from matching hidden
  inputs. Check branches separately, then their sum and final normalization.
- **Performance / bounds:** Compose both enabled child paths, their normalization
  passes and combination. Overlap is limited by shared resources and dependencies.
  A faster expert path cannot be credited with eliminating dense-branch work.

### `MODEL:GEMMA4.MLP:MAG:GEGLU`

- **Contract / implementation:** Separate gate/up projections, approximate GeGLU
  activation and down projection using MLX operations and upstream weight modules.
- **References / tests:** Upstream dense MLP and explicit projection/activation
  equations; compare outputs before surrounding norms and residuals.
- **Performance / bounds:** Three projection costs plus activation/intermediate traffic.
  Model quantized bytes and matrix geometry for decode versus prefill reuse. Packed
  gate/up and fused GeGLU epilogues can remove launches and traffic; those variants
  are not present in this ID's implementation.

### `MODEL:GEMMA4.EXPERT_BRANCH:MAG:ROUTED`

- **Contract / implementation:** Normalize router input, select top-k scores, softmax
  selected scores and apply per-expert scales. Evaluate the shared expert child from
  its separately normalized input, reduce weighted outputs and normalize the result.
- **References / tests:** Upstream router and routed branch. Compare expert selection,
  weights, scaled outputs and branch result with real routing distributions and ties.
- **Performance / bounds:** Router projection/selection plus expert child, reduction
  and normalization traffic. Include assignment sorting and cross-row expert reuse.
  Derive bounds from selected expert work; preserve Gemma's routing semantics when
  considering fused kernels rather than substituting Qwen's routing equations.

### `MODEL:GEMMA4.INPUTS:MAG:PER_LAYER`

- **Contract / implementation:** Prepare auxiliary embedding and projected normalized
  layer inputs with their scales; at each layer apply its gate, projection and norm
  before the residual addition. Preparation and application are observable boundaries
  of this one feature component; neither stage needs to execute when absent.
- **References / tests:** Upstream input preparation and per-layer contribution from
  identical token embeddings and hidden states; test configured scales and geometry.
- **Performance / bounds:** Count auxiliary embedding reads and shared projection once,
  then gated projection/norm work per consuming layer. Preparation reuse matters;
  counting the full auxiliary embedding operation at every layer overstates work.
  Model live prepared tensors as workspace, separately from per-layer traffic.

### `MODEL:GEMMA4.READOUT:MAG:SOFTCAPPED`

- **Contract / implementation:** Final norm, tied or separate vocabulary projection
  and configured optional tanh soft cap. Return only requested logits.
- **References / tests:** Upstream readout from the same residual, covering tied
  weights and configurations with and without a cap.
- **Performance / bounds:** Norm and vocabulary projection plus an output-sized pass
  when capped. Vocabulary size and output-row count determine work. Consider fusion
  with the producer only where it preserves rounding and avoids extra materialization.

## Performance composition

Every contract in this tree resolves to its [ceiling binding](../../performance/catalog.md#gemma-contracts).
The [common definition](../../performance.md) provides an optimistic theoretical
bound per declared dimension, independent of source/variant. References and current
implementation costs diagnose gaps; they do not limit that bound. Parent accounting
allows fusion and shared-data reuse before counting unavoidable demands.

Every component defined here has one `/EXEC` dimension: execution efficiency at
the selected operating point. Query mode, context length and batch size select
samples; resource utilization and workspace remain diagnostics/constraints.
Shared components use their own [catalog definitions](../../performance/catalog.md).
No state or memory percentage is inferred from a KV producer's execution score.

Apply the [shared component models](../composability.md#shared-component-definitions)
using each producer/consumer's actual head geometry and dtype. A sliding attention
window limits visible reads but does not prove physical storage is window-bounded.
Producer count, consumer count and retained allocation must remain separate.

The current tree has no whole-step compilation or fused Gemma projection/GeGLU
path. Those candidates must be identified as new implementations of these contracts
and compared locally and in the parent model under [optimization](../optimization.md).
Dense and routed variants, optional layer inputs and local/global geometry change
which component controls the achievable rate; no architecture-wide numerical
ceiling has yet been established.

## Qualification

Current assessments follow the [evidence/reset rules](../../performance.md#evidence-and-current-assessments):
any implementation change makes its scores and affected parent scores `unmeasured`.
Historical observations remain tied to their original fingerprints and operating points.

Existing components have tests and diagnostic comparisons, but not every named
boundary yet has a dedicated independently qualified benchmark subject. The tree
identifies where those subjects belong; it does not claim they already exist.

Historical custom 16K comparisons left numerical parity open. Native Gemma
long-context qualification applies to the generic adapter, not this assembly.
Test KV-sharing, K=V, window crossings, per-layer inputs, batching and prefix
restoration before broad claims, and retain the selected component IDs with results.

Evidence: `sessions/26-09-06/evidence/cycle-005/comparison-16384.json` and
`sessions/26-09-06/evidence/cycle-006/comparison.json`, relative to the monorepo root.
