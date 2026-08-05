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

# Generation-performance estimation

## Contract

ICN estimates single-user decode throughput at several occupied-context depths for one exact
assessed profile. The ordered estimates are advisory recommendation evidence. They never change
capacity, authorize loading, or replace observed runtime timing.

| Layer | Responsibility |
|---|---|
| Native bindings | Exact workload, placement, and model-free calibration facts |
| `icn-hardware` | Formula, uncertainty, confidence, and failure policy |
| ICN model service | Assessment lifecycle, identity, and caching |
| ACN | Recommendation ranking only |

Performance estimation is part of model assessment. On an exact cache miss, obtaining its workload
requires the same native model open and profile-specific context graph as memory assessment. The
final arithmetic is cheap; obtaining exact inputs is not.

## Scope

Each sample models baseline autoregressive decode for one sequence and one generated token at its
requested occupied-context depth. It excludes prompt processing, sampling, transport, speculative
acceptance, and concurrent scheduling. The serving profile still owns capacity and fit; performance
samples do not create additional serving configurations.

No tensor payload is read and no inference or model benchmark runs.

## Native evidence

The no-allocation planner reports:

- each stored tensor once: type, storage, operation bytes, access class, execution role, placement;
- routed/shared expert counts and roles;
- architecture-specific attention storage as either conventional K/V rows, one MLA latent row, or
  no attention row;
- sliding-window, recurrent, compressed-attention, and sparse-index facts; and
- exact native device identity for every operation.

These are facts, not token-rate estimates. Malformed or internally inconsistent shapes are request
defects. MLA never fabricates a V row; conventional attention never accepts a partial K/V pair.

## Calculation

`icn-hardware` calculates per-token traffic from the exact workload:

- always-active tensors and row lookups use native operation bytes;
- routed pools apply selected/total expert ratios with checked round-up arithmetic;
- attention traffic uses each layer's exact row shape and occupied depth;
- sliding windows cap depth;
- recurrent layers charge fixed state once per token;
- compressed and sparse attention apply their native compression and index/gather terms; and
- every term uses calibration for its actual fitted device.

Time is derived from calibrated operation throughput and dispatch cost. The reciprocal of total
predicted seconds is the expected rate. The same exact native workload is evaluated at each requested
depth, so additional samples require no additional context graph. Versioned efficiency factors cover
unmeasured elementwise, selection, gather, compression, and cross-domain work.

Every result contains finite positive lower, expected, and upper rates with
`lower <= expected <= upper`, plus `high`, `moderate`, or `low` confidence. Calibration dispersion,
fallback evidence, routed uncertainty, unusual architecture work, and cross-physical-domain
placement widen bounds or lower confidence. Unified CPU/Metal ownership alone does not.

## Calibration interaction

Hardware calibration measures bounded synthetic dense and routed operations for enabled backends.
It is model-free and established by the persistent planning-worker pool before ICN readiness. Untimed
warm-up proves synchronized backend execution and may trigger CUDA PTX JIT before timed samples.

Assessment never calibrates lazily. Missing exact operation calibration may use a conservative
same-device fallback with lower confidence. Missing all applicable evidence, invalid metrics, or
invalid arithmetic fails the assessment request.

## Identity and caching

Performance evidence from a stable-topology-validated `Fits` result is cached with its exact
assessment. Identity covers:

- target content and workload schema;
- exact profile and ordered performance sample depths;
- calibration method and concrete metric digest;
- native build, enabled backends, topology, stable capacity, placement, and execution policy; and
- estimator method.

Hardware-calibration elapsed wall time and live free memory are excluded. A warm exact-assessment
cache hit performs no native workload extraction or estimation. Native `DoesNotFit` evidence is not
reused while fallback placement can observe process-local free memory.

## Conformance

- Preview and assessment read no tensor payload and run no model decode.
- Performance failure never changes native capacity evidence into another domain result.
- Bindings contain no token-rate formula, confidence, profile, or recommendation policy.
- Dense, routed, recurrent, conventional-attention, MLA, compressed, sparse, unified-memory, and
  cross-domain paths have deterministic fixture coverage.
- Increasing active traffic cannot improve an otherwise identical estimate.
- Performance samples are strictly ordered by context and end at the configured context.
- Multiple performance depths for one profile reuse one exact native workload.
- Recurrent state is never multiplied by context depth.
- MTP/NextN storage does not affect baseline target decode unless explicitly executed.
- One same-target assessment batch reuses its native model across all missing profiles.
- Warm exact-cache reads perform no native planning.
