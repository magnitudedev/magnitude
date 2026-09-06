# Qwen3.5-family hybrid models

## Scope

**The current model is an assembly of owned hybrid blocks and upstream operations;
whole-model compiled/fused decode remains an optimization gap.** This covers the
accepted Qwen3.5-family text layouts, including compatible Qwen3.6 artifacts, dense
or routed feedforward, and converted affine weights.

MLX-LM supplies configuration and parameter containers. MLX-VLM supplies the
independent target reference and the explicitly adapted standard rotary calculation.
The program exposes residual features, not media conditioning. Source attribution
follows [component identification](../../components.md).

## Assembly

```text
MODEL:QWEN35:MAG:LAYERWISE
├── MODEL:EMBEDDING:MAG:RESIDENT
├── repeated layer assembly: norm → mixer → residual → norm → feedforward → residual
│   ├── mixer: one of
│   │   ├── MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
│   │   │   └── MODEL:ATTENTION:MAG:PAGED
│   │   │       └── MODEL:ATTENTION:MAG:GATHERED       fallback
│   │   │           └── MODEL:ATTENTION:MLX:DENSE
│   │   └── MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION
│   │       └── MODEL:GATED_DELTA:MAG:FUSED_UPDATE
│   └── feedforward: one of
│       ├── MODEL:QWEN35.FEEDFORWARD:LM:DENSE
│       └── MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│           └── MODEL:EXPERTS:MAG:RESIDENT_GATHERED
└── MODEL:QWEN35.READOUT:MAG:STANDARD

State dependency: STATE:QWEN35:MAG:HYBRID
Optional drafter: MODEL:QWEN35.MTP:MAG:CONDITIONED
                  └── STATE:CHECKPOINTS:MAG:NATIVE
```

Shared embedding, attention and expert IDs resolve to the
[shared component definitions](../composability.md#shared-component-definitions).
Native checkpoints resolve to the [generic adapter](generic-mlx-vlm.md#statecheckpointsmagnative).
The repeated layer assembly expresses ordering, not another runtime dispatch.
Norms, projections and residuals inside a component remain testable regions without
requiring an ID for every tensor operation. This tree selects resident weights;
streaming is a separate composition to identify and qualify explicitly.

## Components

### `MODEL:QWEN35:MAG:LAYERWISE`

- **Contract / implementation:** Advance the configured hybrid layer sequence and
  state, returning requested logits/features. Python builds the layer graph each
  forward; only recurrent regions compile. Compatible rows share an arena with
  independent positions. Ordinary residual and normalization order is preserved.
- **References / tests:** Separately loaded stock MLX-VLM target; use MLX-LM as a
  second control with positional/numerical conventions reconciled. Compare layer
  residuals, logits and logical state, then free generation and changing batches.
- **Performance / bounds:** Compose embedding, each selected mixer/feedforward,
  norms/residuals and readout along the dependency graph. Include graph construction,
  encoding and state costs. Compare with the PoC's compiled/fused route at matched
  geometry; it is an attainable reference, not V2 evidence or a hardware ceiling.

### `MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS`

- **Contract / implementation:** Project Q plus output gate, K and V separately;
  normalize Q/K, apply paired rotary transforms, append KV, run the selected
  attention child, then gate and project its output. MLX operation composition.
- **References / tests:** Stock Qwen gated attention with matched weights and logical
  history. Compare prepared Q/K/V, gate, output and appended state; use independently
  computed rotary values to diagnose upstream convention differences.
- **Performance / bounds:** Projection weight traffic and arithmetic, normalization/
  rotary passes, new KV writes, child attention cost and gated output projection.
  Input width, head geometry and context determine the model. Packed projections
  and fused preparation can remove launches/intermediates; verify the enclosing
  block because a changed layout can add copies at the child boundary.

### `MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION`

- **Contract / implementation:** Separate QKV, output-gate and decay/beta projections
  feed convolution, normalization and a replaceable gated-delta update, followed by
  gated normalization/output projection. The tensor region compiles; state staging
  and transaction effects remain outside it. The default update is the shared
  `MODEL:GATED_DELTA:MAG:FUSED_UPDATE`; its upstream alternative is
  `MODEL:GATED_DELTA:LM:STANDARD`.
- **References / tests:** Complete MLX-LM recurrent block and an independent gated-delta
  equation oracle. Compare prepared inputs, convolution history, matrix state and
  output across one/many inputs, batching and accepted-prefix restoration.
- **Performance / bounds:** Count projection traffic, convolution work and recurrent
  matrix reads/writes. State size is independent of total context; query count changes
  update work and algorithmic parallelism. Derive bounds from these quantities and
  their dependency chain. Fused preparation and a different multi-input update are
  opportunities; compilation alone does not imply those fusions exist.

### `MODEL:QWEN35.FEEDFORWARD:LM:DENSE`

- **Contract / implementation:** Pass through the bound upstream gated dense MLP.
- **References / tests:** Independent MLX-VLM MLP and explicit gate/up/activation/down
  equations with the same weights; compare output before the enclosing residual.
- **Performance / bounds:** Two input projections, gating and one output projection.
  Model weight bytes and roughly `2 × rows × input_width × output_width` arithmetic
  per projection, plus activation traffic. Decode favors weight/launch efficiency;
  prefill can reuse weights. Gate/up packing and epilogue fusion need a distinct
  implementation rather than relabeling this upstream path.

### `MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED`

- **Contract / implementation:** Router softmax, top-k and optional renormalization;
  selected expert evaluation; weighted reduction plus a sigmoid-gated shared MLP.
  Uses the shared expert child; routing and combination remain separate MLX operations.
- **References / tests:** Complete upstream routed/shared MLP. Compare assignments,
  probabilities, selected outputs and final sum with representative routing patterns.
- **Performance / bounds:** Compose router, expert child, shared MLP and reduction;
  independent branches may overlap but contend for bandwidth. Count selected expert
  bytes and cross-row reuse, not total resident parameters. Fused routing, gate/up
  activation and weighted combination target exposed launches and intermediates.

### `MODEL:QWEN35.READOUT:MAG:STANDARD`

- **Contract / implementation:** Final upstream norm followed by tied embedding
  projection or the separate language head; compute logits only when requested.
- **References / tests:** Stock final norm/head from identical residuals; check tied
  weights, precision and requested-output behavior independently of the transformer.
- **Performance / bounds:** Norm traffic plus vocabulary projection weight traffic
  and arithmetic. Vocabulary size and projected row count set the work; omit it
  when not requested, but count it in generation. No derived numerical ceiling yet.

### `STATE:QWEN35:MAG:HYBRID`

- **Contract / implementation:** Combine paged attention history and recurrent images
  into one logical checkpoint. Append or tentatively advance both, preserve row
  independence and resolve each accepted boundary without exposing rejected state.
- **References / tests:** Independently advanced upstream KV/recurrent caches and
  explicit prefix replay; compare logical contents after branching, restore and
  unequal verification acceptance, including budget and lifetime failures.
- **Performance / bounds:** KV storage grows with attention history; recurrent images
  are fixed-size but snapshots/replay cost work. Count new writes, required branch
  copies, snapshot bytes and replayed inputs separately. Derive copy bounds from
  byte traffic and repair cost from the recurrent component, without double-counting
  writes already included in neural block timings.

### `MODEL:QWEN35.MTP:MAG:CONDITIONED`

- **Contract / implementation:** Combine normalized token embedding and previous hidden
  conditioning, run attached MLX-LM decoder layers with native caches, then normalize
  and project through the borrowed target vocabulary. The head owns its state;
  proposal acceptance and repair belong to [speculation](../../engine/speculation.md).
- **References / tests:** Matching upstream MTP drafter with identical head weights,
  quantization, conditioning and logical positions. Compare hidden outputs, logits
  and cache transitions independently before testing full speculative generation.
- **Performance / bounds:** Sum conditioning/projection and dependent draft-layer work,
  state traffic and readout. Draft depth creates a serial chain. End-to-end cost per
  committed output includes draft, target verification and repair divided by expected
  committed tokens; acceptance alone is insufficient. Comparative qualification is open.

## Performance composition

Shared component definitions give local byte/compute models. Apply them to the
actual layer mix: attention reads visible history, recurrence advances fixed-size
state, and MoE reads selected weights. Derive prefill and decode bounds separately
under [optimization](../optimization.md). Shared physical resources and dependent
stages prevent summing isolated best-case rates into a full-model ceiling.

The current tree has no whole-step compilation, packed projection path or fused
MoE implementation. Those changes must become explicitly identified substitutions,
with independently validated regions and enclosing model measurements. A better
execution graph can remove costs; present dispatch counts are not an immutable floor.

## Qualification

The IDs above label current implementation responsibilities; dedicated benchmark
subjects for every boundary are not yet established. Component definitions state
what must be compared, not that every comparison has already passed.

Historical 16K decode matched stock continuations; 32K divergence and timing
variability remained unresolved. No broad custom speedup is established. Record
future results against the selected IDs, source revision and full child configuration;
protect dense/MoE variants, long contexts, batch changes and multi-input execution.

Evidence: `sessions/26-09-06/evidence/cycle-004/comparison.json`, relative to the
monorepo root. The PoC is a separate reference implementation, not this source state.
