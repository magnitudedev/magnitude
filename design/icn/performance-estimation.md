---
applies_to:
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - inference/native/llama-cpp-rs/llama-cpp-2/src/model/params/fit.rs
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/wrapper_common_fit.h
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/wrapper_common_fit.cpp
---

# ICN generation-performance estimation

ICN estimates single-user generation throughput for model profiles before model weights are
downloaded. The estimate is advisory evidence for hardware-tailored catalog policy. It never
authorizes loading, changes a memory-fit result, or substitutes for runtime timing from an installed
model.

## Ownership and boundary

The pinned bindings expose native model-workload facts from the same no-allocation model, context
parameters, tensor placement, and backend identities used by memory fitting. They also expose a
bounded model-free ggml calibration. The bindings do not calculate token rates, assign confidence,
choose context points, or define fallback policy.

`icn-hardware` owns the generation formula, active-expert arithmetic, context curve, calibration
matching and fallback rules, uncertainty bounds, confidence, estimator version, and typed failure
policy. ICN model management owns calibration lifecycle, evidence-key construction, and caching.
ICN returns complete evidence through model assessment. ACN recommendation policy may rank those
results but must not reconstruct model workload or hardware throughput.

The estimator may use only:

- GGUF metadata and the tensor directory;
- tensor types, shapes, byte sizes, roles, and no-allocation buffer placement retained by llama.cpp;
- native model hyperparameters, including routed and shared-expert structure;
- the resolved execution profile and native memory-fit result;
- a bounded, model-free calibration of enabled ggml backends; and
- the normalized hardware topology supplied by the shared hardware service.

It must not download or read tensor payloads, run a model decode, inspect prompts, depend on an
installed copy of the artifact, or infer behavior from a product catalog name. Remote sparse-header
materialization and a complete local artifact therefore produce the same model workload.

## Workload definition

The initial estimator describes baseline autoregressive decode for one sequence and one generated
token at a time. It includes the target model, its configured KV types, attention policy, and the
exact fitted placement. It excludes prompt processing, sampling, transport, speculative decoding,
MTP acceptance, and concurrent-session scheduling. Those exclusions are serialized with the result
rather than left implicit.

An estimate is a curve over occupied context depth. Occupied depth is distinct from configured
context capacity. ICN requests the standard points 8,192, 32,768, 100,000, and 200,000 tokens,
discarding points above the effective configured or trained limit and de-duplicating clamped values.
These points are performance observations, not alternative model configurations.

## Native model workload facts

The bindings return every stored tensor exactly once with its canonical name, tensor type, native
buffer-device identity, complete storage bytes, native access class, bytes touched by that access
before routed-expert selection, and whether ordinary target-model decode executes it. Native access
classes distinguish always-active tensors, routed-expert pools, and row lookups. Untied input,
positional, per-layer, and hash-routing embeddings are row lookups; a token embedding shared with
the output projection remains an always-active full matrix operation. Stored MTP/NextN tensors are
explicitly not baseline operations. The bindings report total and selected expert counts but do not
apply their ratio.

For every layer, the bindings return native K/V row bytes, configured cache types, fitted KV-device
identity, attention head width and fixed-state type, sliding-window limit, recurrent flag and fixed
recurrent-state rows, cache-compression ratio, and sparse-index row width and execution flag. The
workload summary includes the canonical native architecture, MTP/NextN layer count, MLA/KV rank,
and sparse-index dimensions and top-K. It also reports native hybrid and recurrent model
classifications. These are facts, not an estimate: the binding payload contains no token rates,
efficiency constants, uncertainty bounds, or confidence labels.

## ICN workload calculation

`icn-hardware` charges always-active operations and row lookups by their native operation bytes.
Routed expert pools are charged by applying selected experts divided by total experts with checked,
round-up integer arithmetic; router tensors and shared experts remain always active. Missing or
inconsistent expert metadata makes performance unavailable rather than falling back to an
active-parameter ratio over the complete model.

KV traffic is calculated from the native per-layer K/V row bytes. Full-attention layers scale with
occupied depth and sliding-window layers are capped by their native window. Recurrent layers add no
context-dependent KV traffic; they instead charge one fixed state read and write per generated token
using their native state type and fitted device. Compressed attention scales stored depth by its
native compression ratio. Sparse attention separately charges its context-growing index scan and
its top-K-bounded cache gather. DeepSeek V4 compressed layers additionally charge a read of the
native F32 compressor state—and CSA index state—plus the retained row written for the new token;
this traffic does not grow with context. Specialized K-only caches may expose one zero ordinary K/V
row; ordinary attention with only one missing row remains invalid. Hybrid models apply these rules
independently per layer. Native layer identities must be unique; duplicate identities make the
workload invalid rather than double-counting it.

Host/device placement is part of the workload: every tensor and every layer's KV traffic uses the
calibration for its actual native device. A configuration that fits through partial offload or
expert buffer overrides must not inherit the estimate for a fully accelerated configuration.
Phase 1 does not directly calibrate the small activation transfers at backend boundaries; a
placement spanning multiple physical memory domains therefore receives a conservative efficiency
penalty and low confidence. Multiple native execution-device identities do not alone establish
such a placement. In particular, ordinary CPU and Metal ownership on Apple Silicon remains one
unified-memory-domain placement and receives no cross-domain penalty; each operation is still
charged against its actual CPU or Metal calibration.

## Model-free backend calibration

Calibration uses bounded synthetic ggml tensors and the same backend operations used by the pinned
runtime. It contains no model weights and performs no tokenization or decode. It measures effective
dense matrix-vector work and routed `MUL_MAT_ID` work for the weight and KV-cache tensor types
supported by the curated catalog. Each sample includes the backend's actual arithmetic, tensor
reads, dequantization, dispatch, and synchronization cost for the synthetic operation.
Synthetic weight buffers exceed typical consumer shared-cache working sets, so calibration measures
streaming operation throughput rather than repeatedly crediting a small cache-resident tensor.

Calibration is performed at most once for one native-build, enabled-backend, topology, and
calibration-policy identity within an ICN process. The result is serializable so isolated planner
workers reuse it. Calibration failure is non-fatal: memory fitting continues and performance is
reported as unavailable. Calibration has explicit time and temporary-allocation bounds and releases
all native resources before returning.

## Estimate and confidence

In `icn-hardware`, time for each native operation class is derived from its calibrated effective
tensor-read rate and dispatch overhead. Routed and dense matrix operations use different calibration
evidence. K and V traffic is charged independently for every layer, with sliding-window limits and
the layer's actual native device. The reciprocal of total predicted seconds per token is the raw
generation rate. Estimator v5 applies ICN-owned upper efficiency factors of `0.82` for ordinary
dense decode and `0.75` for routed decode. Recurrent, sparse-attention, and compressed-attention
workloads cap that factor at `0.72`, `0.68`, and `0.65` respectively to cover state-update,
selection, gather, compression, and elementwise work outside calibrated matrix and memory traffic.
Cross-memory-domain placement applies a further `0.88` factor. These are versioned estimator policy:
any change requires a new ICN estimator method identity and cache evidence.

Every available result contains finite, positive lower, expected, and upper rates with
`lower <= expected <= upper`. Bounds incorporate calibration dispersion and workload coverage.
Estimator v5 starts with at least 12% uncertainty, weights observed calibration spread by `1.5`,
and widens routed and cross-memory-domain estimates further. Confidence is high, moderate, or low
and is lowered by missing exact operation calibration, routed-expert uncertainty,
cross-memory-domain placement, or architecture work represented by a conservative related
calibration. Unified CPU/accelerator device ownership is not itself a reason to lower confidence.
When exact routed or quant calibration is absent but the same fitted device has valid related
calibration, ICN returns a bounded hardware-specific estimate with lower confidence rather than
suppressing the model. Structurally malformed workloads still return a typed unavailable result;
zero, NaN, infinity, and an unqualified point estimate are invalid.

The evidence identity includes the ICN estimator method, native workload schema, native calibration
schema, artifact content, execution policy, native build, enabled backends, topology, capacity
policy, effective placement, and requested context points. Evidence identities are cache inputs,
not cache schema versions.

## Failure and lifecycle semantics

Performance is independent of fit success. A calibration, workload, arithmetic, or decoding failure
cannot change `Fits`, `DoesNotFit`, incompatibility, or native fit diagnostics. A profile without a
runnable placement has no generation estimate. Operational failures needed to produce a memory fit
remain ordinary operation failures; estimator-specific failures become typed unavailable evidence
and retain their method, code, and message through the public assessment response.

Preview caches the composite profile assessment through the model-management cache. Cache reads and
writes retain the no-fail behavior of disposable caches. Volatile calibration samples are not a new
durable authority. Loading always replans, and installed-model runtime timing remains authoritative
for observed performance.

## Acceptance criteria

- Preview never downloads or reads model tensor payloads to estimate performance.
- Native memory-fit outcomes and byte accounting are unchanged when performance is unavailable.
- The bindings expose workload and calibration facts but contain no throughput formula, confidence
  assignment, context policy, or recommendation semantics.
- Dense, routed-expert, shared-expert, sliding-window, recurrent, compressed-attention,
  sparse-index, unified-memory, CPU-only, and cross-memory-domain calculations have deterministic
  fixture coverage.
- Apple Silicon CPU and Metal workload ownership resolves to one performance memory domain and
  never receives a cross-domain penalty solely because both native devices appear in the plan.
- Increasing active tensor traffic, routed experts, or per-layer KV context depth cannot improve
  an otherwise identical estimate.
- Recurrent state is charged once per token and never multiplied by occupied context.
- Stored MTP/NextN tensors do not affect baseline target-model throughput.
- Sparse index scans continue to scale with their indexed history after the gathered attention
  depth reaches top-K; compressed histories scale by their native compression ratio.
- Calibration is bounded, reusable across profiles and artifacts, and never runs concurrently for
  the same evidence identity.
- Malformed native values and incomplete MoE metadata produce typed unavailable results. Missing
  exact operation calibration uses a conservative same-device fallback and lowers confidence.
- Remote sparse and complete local forms of the same artifact produce identical model workloads.
- Recommendation policy consumes only a complete estimate point matching the exact selected product
  context. Clients present the selected evidence but never reinterpret workload or throughput.
