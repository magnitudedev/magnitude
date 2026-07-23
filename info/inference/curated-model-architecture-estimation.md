# Curated model architectures and generation estimation

This document is the architecture reference for every checkpoint family in Magnitude's curated
local-model catalog. It records how each decoder works, which work is active for one generated
token, how sequence state scales with occupied context, and how ICN should estimate baseline
generation throughput without running model weights.

The catalog changes over time. Any catalog change must update the coverage table and either map the
new checkpoint to a reviewed architecture below or add a new architecture section. Similar parameter
counts or marketing names are not enough to reuse an architecture model.

## Estimation target

The first-pass estimate is baseline autoregressive decode:

- one target model;
- one sequence;
- one accepted output token;
- a specified occupied context;
- the resolved llama.cpp execution plan and tensor placement; and
- no prompt processing, sampling, transport, concurrent scheduling, or model load time.

Speculative decoding is deliberately separate. Baseline does not execute an external draft model,
DFlash, or an MTP/NextN draft head. A checkpoint may store MTP tensors without executing them in its
ordinary decoder graph; storage is relevant to fit and download size but not baseline token time.
Likewise, multimodal encoders do not contribute to text-token decode after their input embeddings
have been produced.

The analytical estimate has four time components:

```text
token time =
    executed weight-operation time
  + sequence-memory time
  + architecture-specific operation time
  + cross-device transfer time

tokens/second = architecture efficiency / token time
```

All components use the exact quantization, native device placement, and backend selected by the
resolved plan. Uncertainty describes how much of the graph is represented by direct calibration; it
does not remove an otherwise finite best estimate.

## Catalog coverage

| Catalog checkpoint | Runtime family | Decoder shape | Reviewed baseline treatment |
| --- | --- | --- | --- |
| Qwen3.5 4B | Qwen3.5 dense hybrid | 24 Gated DeltaNet + 8 full-attention layers | Recurrent state + full-attention KV |
| Qwen3.5 9B | Qwen3.5 dense hybrid | 24 Gated DeltaNet + 8 full-attention layers | Recurrent state + full-attention KV |
| Qwen3.6 27B | Qwen3.5 dense hybrid | 48 Gated DeltaNet + 16 full-attention layers | Recurrent state + full-attention KV |
| Qwen3.6 35B-A3B | Qwen3.5 hybrid MoE | 30 Gated DeltaNet + 10 full-attention layers; 256 routed experts | Recurrent state + top-8 experts + shared expert + full-attention KV |
| Gemma 4 E2B | Gemma 4 dense with per-layer embeddings | 28 sliding + 7 global-attention layers | Row lookups + 512-token sliding windows + global KV |
| Gemma 4 12B | Gemma 4 dense | 40 sliding + 8 global-attention layers | 1,024-token sliding windows + global KV |
| Gemma 4 26B-A4B | Gemma 4 MoE | 25 sliding + 5 global-attention layers; 128 routed experts | Top-8 experts + shared expert + 1,024-token sliding windows + global KV |
| Gemma 4 31B | Gemma 4 dense | 50 sliding + 10 global-attention layers | 1,024-token sliding windows + global KV |
| Laguna S 2.1 118B-A8B | Laguna transformer MoE | 36 sliding + 12 global-attention layers; 256 routed experts | Top-10 experts + shared expert + 512-token sliding windows + global KV |
| Qwen3.5 122B-A10B | Qwen3.5 hybrid MoE | 36 Gated DeltaNet + 12 full-attention layers; 256 routed experts | Recurrent state + top-8 experts + shared expert + full-attention KV |
| Nemotron 3 Super 120B-A12B | Nemotron-H LatentMoE hybrid | Interleaved Mamba-2, MoE, and attention anchors | Mamba state + latent top-22 experts + shared expert + attention KV |
| DeepSeek V4 Flash 284B-A13B | DeepSeek V4 compressed-attention MoE | 43 layers using raw, CSA, and HCA attention; 256 routed experts | Hash/top-6 experts + compressed state + sparse index + gathered KV |
| Nemotron 3 Ultra 550B-A55B | Nemotron-H LatentMoE hybrid | Interleaved Mamba-2, MoE, and attention anchors | Mamba state + latent top-22 experts + shared expert + attention KV |
| GLM 5.2 753B-A40B | GLM MLA/DSA MoE | 78 main layers; dense lead-in then MoE; shared sparse indexers | Dense lead-in + top-8 experts + shared expert + index scan + gathered MLA KV |

The table describes the published checkpoints as reviewed on 2026-07-23. Exact GGUF metadata and
the pinned runtime remain authoritative for the artifact being assessed.

## Common accounting rules

### Executed tensors, not stored tensors

Memory fitting charges every stored tensor. Generation performance charges only operations executed
by the baseline decoder graph.

- A dense matrix used once contributes one calibrated matrix-vector operation.
- A packed routed-expert matrix contributes only the selected expert slices.
- A token, position, type, per-layer, or hash-table lookup contributes the rows actually fetched.
- A shared expert, router, gate, latent projection, normalization, and output head is always active
  when the baseline graph executes it.
- A tensor referenced multiple times contributes its execution multiplicity, even if stored once.
- MTP/NextN, DFlash, multimodal projector, vision, and audio tensors contribute zero baseline decode
  work when their graph is not executed.

Tensor names may help diagnose a native workload, but ICN must not derive architecture semantics
from a catalog ID. Workload facts come from the no-allocation native model and graph selected for
the exact artifact.

### Active expert math

For a uniform packed expert pool with `N` experts and top-`K` routing, the active bytes for one
expert tensor are:

```text
active routed bytes = ceil(packed operation bytes × K / N)
```

This ratio is applied independently to every executed routed tensor. It is not applied to the
entire checkpoint. Embeddings, attention, routers, shared experts, dense lead-in layers, latent
projections, normalization, and the output head remain always active.

The estimator must use per-layer routing metadata when `N` or `K` differs by layer. Hash routing
changes how experts are selected, not how many expert matrices execute. Expert-parallel placement
changes the devices and transfer cost, not the active-byte ratio. A reported “active parameter”
headline is a useful cross-check but is not the formula.

### Attention and recurrent state

For ordinary full attention at occupied context `C`, a layer reads approximately:

```text
full KV bytes/token = C × (key row bytes + value row bytes)
```

For sliding-window attention with window `W`:

```text
sliding KV bytes/token = min(C, W) × (key row bytes + value row bytes)
```

Grouped-query, multi-query, unified K/V, and MLA change the native K/V row widths. The estimator
uses those native widths rather than reconstructing them from nominal head counts.

Recurrent layers do not multiply state rows by context. They read and update a fixed state for the
current sequence on every token. Their cost includes recurrent-state reads and writes, convolution
state, and architecture-specific scan/update operations. Treating that state as a full-context KV
cache is catastrophically pessimistic; ignoring it entirely is optimistic.

Sparse attention has two distinct costs:

1. selecting positions, often by scanning a compressed index whose depth grows with context; and
2. gathering and attending the fixed or bounded top-`K` positions.

Only the gather is context-capped by top-`K`. An estimator that models sparse attention as a simple
sliding window misses the index scan; one that models it as dense full attention overstates the
gather.

### Quantization calibration

Every executed weight operation is matched to calibration for its actual GGML tensor type, backend,
device, and dense-versus-routed operation.

- Q4, Q5, Q6, Q8, BF16, F16, IQ, and MXFP4 are distinct operation classes.
- A QAT checkpoint still uses the GGUF tensor types present in the artifact. QAT is principally
  fidelity evidence; it does not justify substituting a different runtime type.
- Packed MoE kernels use routed calibration. Ordinary matrix calibration cannot stand in for
  `MUL_MAT_ID` without additional uncertainty.
- MXFP4 must use an exact supported kernel or the closest measured backend operation with widened
  uncertainty; its nominal four-bit storage alone does not predict throughput.
- KV cache types are calibrated separately from weight types.
- Tiny vectors, elementwise gates, recurrent scans, sparse indexers, cache compression, and attention
  kernels are not equivalent to large matrix-vector reads. Their analytical costs or architecture
  allowances must be added instead of pretending each is another large matmul.

The model-free calibration measures the actual enabled backend. It is therefore recomputed when the
native build, backend set, device topology, or calibration policy changes.

### Context scaling

Configured context is capacity; occupied context is the tokens currently available to attention.
Generation curves are evaluated at occupied depths no greater than the configured or trained limit.

- Dense global attention grows linearly in decode traffic with occupied context.
- Sliding attention stops growing at its window.
- Recurrent state is constant with occupied context.
- Compressed and sparse attention grows according to compression depth plus bounded gathered
  positions.
- Model weights do not grow with context.
- Context allocation and fit may grow even where recurrent or sliding decode traffic does not.

The estimate should be monotonic where the architecture requires it: increasing occupied context
must not improve the same placement and quantization. A flat curve is valid for a fully recurrent
model or a fully saturated sliding-window model.

### Hardware placement and replacement

The same artifact must be recalculated for every hardware topology.

- Each tensor operation uses calibration for the device that executes it.
- Each cache/state operation uses the device that owns that memory.
- Unified-memory CPU and accelerator devices share one capacity domain but retain different
  operation calibrations.
- Partial offload, expert offload, and split models include backend-boundary transfers.
- Discrete GPU placement must not use system RAM bandwidth for GPU-resident work or count system and
  VRAM capacity as one domain.
- Replacing a GPU, changing enabled backends, or changing tensor placement invalidates the estimate.
- Cross-domain penalties are based on actual transfers, not merely the presence of two device names.

## Qwen3.5 and Qwen3.6 dense hybrid

### Checkpoints

- Qwen3.5 4B
- Qwen3.5 9B
- Qwen3.6 27B

These use the same Qwen3.5 text architecture despite the Qwen3.6 product name. Their repeating
decoder pattern is three Gated DeltaNet layers followed by one full gated-attention layer.

| Checkpoint | Main layers | Recurrent layers | Full-attention layers | Full-attention Q/KV heads |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5 4B | 32 | 24 | 8 | 16 / 4 |
| Qwen3.5 9B | 32 | 24 | 8 | 16 / 4 |
| Qwen3.6 27B | 64 | 48 | 16 | 24 / 4 |

Gated DeltaNet performs single-token decode through a fixed recurrent matrix state and a short
causal-convolution state. It is linear in prefill length but constant in stored sequence depth for
each decode token. Full-attention layers retain ordinary KV rows for the entire occupied context.

Baseline estimation therefore includes:

1. all main-decoder dense projections and FFNs;
2. recurrent-state read, update, and write traffic for each DeltaNet layer;
3. convolution-state traffic for each DeltaNet layer;
4. full-context K/V traffic for only every fourth layer; and
5. the ordinary output projection.

It excludes the optional MTP decoder block. If the GGUF stores that block alongside the trunk, its
bytes affect fit but not baseline token time.

The architecture should receive recurrent/hybrid uncertainty because matrix calibration does not
directly measure the DeltaNet update kernel. It must not receive a context-proportional charge for
recurrent state.

## Qwen3.5 and Qwen3.6 hybrid MoE

### Checkpoints

- Qwen3.6 35B-A3B
- Qwen3.5 122B-A10B

These retain the three-recurrent/one-full-attention pattern and replace every dense FFN with sparse
MoE. Both use 256 routed experts, top-8 selection, and an always-active shared expert.

| Checkpoint | Main layers | Recurrent / full | Hidden size | Routed experts | Selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 40 | 30 / 10 | 2,048 | 256 | 8 |
| Qwen3.5 122B-A10B | 48 | 36 / 12 | 3,072 | 256 | 8 |

For each MoE layer, routed gate/up/down pools receive the `8/256` active fraction. The router and
shared-expert gate/up/down projections are always active. Attention and recurrent projections are
also always active. The total/active parameter labels are only a cross-check because embeddings,
attention, and shared experts do not obey the routed fraction.

The baseline formula combines the Qwen recurrent-state treatment with routed backend calibration.
MTP is excluded until a separate speculative estimate is requested.

## Gemma 4

### Dense checkpoints

- Gemma 4 E2B
- Gemma 4 12B
- Gemma 4 31B

### MoE checkpoint

- Gemma 4 26B-A4B

Gemma 4 interleaves several local sliding-attention layers with one global-attention layer. The
native layer pattern and per-layer K/V widths are authoritative because global layers can use
different head dimensions or K/V sharing from local layers.

| Checkpoint | Layers | Sliding / global | Window | MoE routing |
| --- | ---: | ---: | ---: | --- |
| E2B | 35 | 28 / 7 | 512 | Dense |
| 12B | 48 | 40 / 8 | 1,024 | Dense |
| 26B-A4B | 30 | 25 / 5 | 1,024 | 128 experts, top-8, plus shared expert |
| 31B | 60 | 50 / 10 | 1,024 | Dense |

E2B uses per-layer embeddings. Those large tables are row lookups, not full matrix operations.
Charging their stored bytes on every token would make the effective model appear much slower than
it is. Multimodal encoders are not part of text-token baseline decode.

For 26B-A4B, only the routed expert pools receive the `8/128` fraction. The shared expert and the
rest of the decoder remain active. Its roughly 4B active headline is consistent with this structure
but does not replace tensor-level accounting.

Gemma QAT GGUFs should use their actual quantized operation calibration. QAT supports a stronger
fidelity claim than ordinary Q4; it does not make the tensor a different speed class by itself.

## Laguna S 2.1

Laguna S 2.1 is a conventional transformer MoE with 48 layers: 12 global-attention layers and 36
512-token sliding-window layers in a repeating one-global/three-sliding pattern. It has 256 routed
experts with top-10 token-choice routing plus one always-active shared expert. Attention uses grouped
queries with eight K/V heads and 128-dimensional heads.

Baseline estimation includes:

- exact Q4_K_M or Q8_0 tensor calibration;
- `10/256` of each routed expert pool;
- the complete shared expert and router;
- full occupied-context K/V reads on 12 layers; and
- at most 512 tokens of K/V reads on 36 layers.

The optional Laguna DFlash model is a separate speculative target and contributes nothing to the
baseline estimate. Attention output gating and router elementwise work receive an architecture
allowance beyond calibrated matrix reads.

## Nemotron 3 Super and Ultra

### Checkpoints

- NVIDIA Nemotron 3 Super 120B-A12B
- NVIDIA Nemotron 3 Ultra 550B-A55B

Nemotron-H is a hybrid Mamba-2/attention/LatentMoE architecture. Mamba blocks hold fixed recurrent
SSM and convolution state. A smaller number of attention anchors retain full-context KV. LatentMoE
projects routed computation from the model hidden dimension into a smaller latent dimension, runs
many small selected experts there, and projects back. Routers, latent projections, and shared
experts remain active.

Super has 88 main layers, 512 routed experts, top-22 selection, a 1,024-dimensional routed latent
space, and a full-width shared expert. Ultra uses the same family at larger scale, with 512 routed
experts, top-22 selection, a 2,048-dimensional latent space, and a full-width shared expert. Exact
block ordering and state dimensions come from the artifact's native hyperparameters.

Baseline estimation includes:

1. Mamba projections and fixed SSM/convolution-state traffic on Mamba blocks;
2. `22/512` of executed latent expert pools;
3. always-active latent input/output projections, router, and shared expert;
4. full-context KV traffic only on attention anchors; and
5. the main output head.

MTP weights are excluded. MXFP4_MOE uses routed MXFP4 calibration on a backend that supports the
actual operation. Mamba state commonly remains F32 or F16 even when weights are quantized, so its
traffic must use state dtype rather than checkpoint weight precision.

The principal analytical uncertainty is Mamba/LatentMoE kernel efficiency, not active-expert count
or context scaling. Treating Nemotron as a standard transformer MoE would incorrectly allocate KV
traffic to Mamba blocks and omit recurrent state.

## DeepSeek V4 Flash

DeepSeek V4 Flash is a 43-layer, 284B-total/13B-active MoE. It uses 256 routed experts with top-6
selection and one shared expert. The first three layers use token-ID hash routing; later layers use
learned routing. Hash routing still executes six experts, but its token-ID-to-expert table is a row
lookup rather than a full matrix operation.

Its attention is neither ordinary dense attention nor simple sliding attention:

- raw local layers retain a 128-token local window;
- Compressed Sparse Attention (CSA) maintains overlapping state compressed at ratio 4, scans a
  compressed index, selects up to 512 positions, and gathers their compressed KV;
- Heavily Compressed Attention (HCA) maintains state compressed at ratio 128 and attends that
  compressed history; and
- Manifold-Constrained Hyper-Connections add always-active projections and mixing work.

The published layer compression pattern contains raw bootstrap/final layers and alternating CSA/HCA
layers. The exact per-layer ratio from the GGUF controls estimation.

For one layer at occupied depth `C`:

```text
raw local depth = min(C, 128)
CSA stored depth ≈ ceil(C / 4)
CSA gathered depth = min(index top-K, CSA stored depth)
HCA attended depth ≈ ceil(C / 128)
```

CSA also pays the index scan over compressed history; it is not just a 512-token window. HCA pays
attention over its heavily compressed depth. CSA and HCA each read their fixed F32 compressor state
and write the retained row for the new token; CSA does the same for fixed indexer state. That
state-update cost does not grow with context. Native compressed row sizes, attention/indexer head
widths, and state types determine bytes. The estimator adds weight operations, fixed state updates,
index scan, gather, and attention allowances separately.

The FP4/FP8 mixed checkpoint requires tensor-level type calibration. The optional NextN/MTP head is
excluded from baseline.

## GLM 5.2

GLM 5.2 is a 1M-context MLA/Deep Sparse Attention MoE. It has 78 main layers, three dense lead-in
layers, then sparse MoE layers with 256 routed experts, top-8 selection, and one shared expert.
Multi-head latent attention stores compressed KV rather than decompressed per-query-head KV.

Deep Sparse Attention uses an indexer to select up to 2,048 historical positions. IndexShare
computes a full index on every fourth sparse-attention layer and reuses that selection for the next
three layers. Estimation therefore separates:

- compressed MLA KV storage and per-token update;
- a context-growing index scan only where an index is computed;
- bounded gathers/attention over at most 2,048 selected positions on every sparse-attention layer;
- dense FFNs in the first three layers;
- `8/256` routed expert pools in later layers; and
- always-active routers and shared experts.

A stored shared indexer is counted according to execution frequency, not once per consumer layer
and not merely once because its weights are stored once. Preserved NextN/MTP tensors are excluded
from baseline.

## Confidence and validation

Confidence is evidence coverage, not a product eligibility switch.

- **High:** exact tensor type/device calibration; standard dense or sliding attention; no
  uncalibrated architecture-specific kernel dominates token time.
- **Moderate:** active MoE routing, recurrent state, latent experts, or sparse/compressed attention
  is analytically represented but kernel efficiency is inferred from related calibrated operations.
- **Low:** placement spans physical memory domains, an exact quantized operation is unavailable, or
  architecture-specific work relies heavily on conservative analytical bandwidth/compute bounds.

Every estimate still returns lower, expected, and upper rates. Wider bounds are required when more
of token time comes from analytical rather than direct synthetic calibration.

Architecture fixtures must verify:

- catalog coverage has no unmapped checkpoint;
- only main-decoder tensors are charged for baseline;
- row lookups never charge a complete table;
- increasing selected experts cannot improve speed;
- shared experts remain fully active;
- recurrent state does not scale with context;
- sliding layers stop scaling at their window;
- sparse index scans and compressed histories scale with their architecture;
- MTP/DFlash exclusion is stable;
- changing quantization or placement selects corresponding calibration; and
- increasing occupied context never improves an otherwise identical estimate.

## Primary architecture sources

- Qwen3.5 4B model card and configuration:
  https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.6 27B configuration:
  https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json
- Qwen3.6 35B-A3B configuration:
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json
- Gemma 4 26B-A4B model card and configuration:
  https://huggingface.co/google/gemma-4-26B-A4B
- Laguna S 2.1 model card and configuration:
  https://huggingface.co/poolside/Laguna-S-2.1
- Nemotron 3 Super architecture:
  https://docs.nvidia.com/nemotron/latest/nemotron/super3/pretrain.html
- Nemotron 3 Super deployment architecture:
  https://docs.nvidia.com/nemotron/latest/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html
- Nemotron 3 Ultra architecture:
  https://docs.nvidia.com/nemo/automodel/nightly/recipes-e2e-examples/nemotron-3-ultra
- DeepSeek V4 Flash model card:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- DeepSeek V4 runtime architecture:
  https://huggingface.co/docs/transformers/model_doc/deepseek_v4
- GLM 5.2 model card and configuration:
  https://huggingface.co/zai-org/GLM-5.2

The pinned llama.cpp model loaders and graph builders are the execution authority used to reconcile
these published descriptions with an exact GGUF artifact.
