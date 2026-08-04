---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-server/**
  - inference/crates/icn-api/**
  - inference/native/llama-cpp-rs/**
  - packages/acn/src/local-model*.ts
  - packages/acn/src/local-provider-offering-projection.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/model-menus/**
---

# ICN model calibration

## Definition

**Calibration = hardware characterization + model assessment.**

```text
synthetic native operations ──> hardware characterization ─┐
                                                          ├─> calibrated model evidence
GGUF metadata + target/profile/policy ──> model workload ──┘
```

Neither input is sufficient alone:

- hardware characterization has no model-specific workload;
- GGUF metadata has no machine-specific execution rate.

## Terminology

| Term | Meaning |
|---|---|
| Calibration | Complete production of model-specific hardware evidence |
| Hardware characterization | Reusable measurement of native device operations |
| Model assessment | Combination of GGUF workload, characterization, topology, profile, and policy |
| Fit | A memory result: an allocation fits or does not fit capacity |
| Profile selection | Selection among completed assessment results |
| Configuration materialization | Creation of an offering from current calibration evidence |

“Fitting” must not name a process, API, lifecycle, or product status.

## Stage 1: hardware characterization

| Input | Output |
|---|---|
| Enabled native devices and backends | Backend and physical-device identity |
| Supported tensor storage types | Effective tensor bytes/second per tensor type |
| Dense and routed synthetic operations | Separate dense and routed rates |
| Bounded timed samples | Launch overhead, dispersion, convergence, and stability |
| Native build and measurement policy | Characterization identity |

Properties:

- Executes synthetic operations; consumes no GGUF data or model weights.
- Includes reads, arithmetic, dequantization, dispatch, and synchronization.
- CUDA warmup proves execution and triggers PTX JIT.
- Runs once per native-process evidence identity; concurrent callers share it.

Invalidated by a change to native build, enabled devices/backends, operation schema, or measurement
policy.

## Stage 2: model assessment

| GGUF-derived workload | Resulting calibrated evidence |
|---|---|
| Active tensor bytes by device, tensor type, and dense/routed/lookup class | Expected, lower, and upper generation rates |
| Total and selected experts; always-active and routed work | Confidence and uncertainty |
| KV, sliding-window, compressed, sparse-index, and recurrent traffic by context | Context-dependent performance curve |
| Model, KV/context, compute, auxiliary, and speculative allocations | Memory accounting by physical domain and device limit |
| Exact device placement | Compatibility and fit result |
| Immutable target and serving profile | Configuration and assessment identities |

Assessment consumes GGUF metadata and tensor descriptors, never tensor payloads. Identical model
content, profile, topology, characterization, native build, and policy produce identical evidence.

## Scope

| Target source | When assessed | Model input |
|---|---|---|
| Curated catalog | Initial product calibration, at every curated profile | Release-bundled compact planner GGUF |
| Discovered model matching current catalog evidence | Not reassessed | Existing catalog calibration |
| Discovered model without matching evidence | On demand when a configuration is needed | Installed GGUF metadata |

Merely discovering a model on disk does not add it to initial catalog calibration.

## Download lifecycle

```text
catalog calibration
      ↓
selected target/profile
      ↓
download → verify content → inspect capabilities
      ↓
validate calibration identity
      ├─ current ──> materialize provider offering
      └─ stale ────> recompute only invalidated calibration stage
```

Rules:

- A verified catalog download does not trigger a second assessment.
- Compact planner and complete installed GGUF forms of one immutable artifact must produce
  identical assessment evidence; release generation proves this equivalence.
- Curated installation does not perform an unconditional maximum-context search.
- Materialization references existing evidence; it does not duplicate memory or throughput logic.

Calibration identity covers:

- immutable target content and workload schema;
- native build and concrete hardware characterization;
- topology and effective placement;
- serving profile and capacity policy; and
- estimator policy.

## State and failure contract

| State | Calibration? |
|---|---|
| Hardware characterization | Yes |
| Model assessment | Yes |
| Download or content verification | No |
| Capability inspection | No |
| Offering persistence or provider reconciliation | No |
| Configuration materialization | No |
| Loading | No |

- Every calibration operation is bounded and terminates in typed success or failure.
- Failure or timeout cannot remain indefinitely `Preparing` or become an empty success.
- Materialization failure is reported separately and does not invalidate calibration evidence.
- Cache loss recomputes only the smallest missing stage.

## Acceptance criteria

- Every machine-specific estimate combines exact model workload with measured characterization.
- Initial calibration assesses curated target/profile pairs, not all discovered files.
- Installing a currently calibrated catalog target performs no assessment or context search.
- Changed inputs invalidate only affected evidence.
- Product state always identifies the active calibration substage.
- Non-calibration work is never presented as calibration.
