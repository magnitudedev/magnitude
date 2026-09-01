---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-api/**
  - packages/icn/src/hardware/**
  - packages/icn/src/models/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
---

# Model assessment and ranking

ICN owns assessment-material resolution, assessment demand, template-derived capabilities,
compatibility, memory, placement, performance, filtering, scheduling, cache reuse, native
concurrency, deadlines, and assessment publication. ACN owns catalog ranking and projects native
assessment state. Clients only render the result.

Assessment Material is the compact, immutable GGUF and bundle evidence required to assess a model
without installed tensor payloads. It preserves the effective default and named chat templates,
BOS/EOS identities and exact token strings, add-token flags, tensor directory, component roles, and
content identities. Companion projector material preserves its bounded GGUF metadata and tensor
directory, so native assessment can establish modalities without an authored capability flag.
Catalog desired, catalog effective, and discovered targets resolve this same input shape.

The automatic ICN pool assesses catalog desired material when not installed, effective material
when installed, and only `Ready` discoveries. It publishes a read-only revisioned snapshot with
independent catalog and discovery source slices. Exact work identity guards publication, so removed
or superseded models cannot retain stale results. Packages, bundles, and serving configurations do
not cross the boundary. Whole-source failures are observed and retried with bounded background
backoff. Each exact target is attempted once; any target failure settles as `Dropped`, is never
retried, and is omitted by ACN. Catalog drops emit an OpenTelemetry error; discovered drops are
silent.

One planning-worker job opens the target Assessment Material once without tensor allocation. That
same native model handle supplies effective-template/tool/reasoning analysis and every requested
context-graph measurement. The job also establishes projector modalities and speculative-decoding
compatibility. One deadline covers the job, and one flat assessed result publishes capabilities,
template fingerprint, and profile evidence together while retaining the exact native runtime
configuration required by residency. There is no template worker, template timeout, template
cache, inventory capability state, or post-download assessment gate.

`Fits`, `DoesNotFit`, and `Incompatible` are genuine terminal evidence. Transport or operation
failure drops the target rather than fabricating compatibility evidence. Hardware observations
that were not performed are represented as `NotObserved`; zero-valued headroom is never fabricated.

Ranking exists only for reviewed catalog models with `Fits` evidence and the required bounded
performance sample. Intelligence and fidelity come from authored catalog evidence; speed comes
from native assessment. Missing evidence yields absent ranking scores, never zeros. Discovered
models receive no invented intelligence or fidelity score.

Provider selection requires `Fits`, current selectability, profile, and capabilities from the same
assessed state. Package validation establishes only structural artifact validity and presence;
assessment is the sole capability authority. Loading consumes the completed assessment's template,
reasoning, modality, and speculative configuration. Admission recomputes only memory fit against
current resources; it cannot rerun capability inspection or speculative preflight. The inference
worker verifies the loaded artifact's fingerprint and modalities against the assessment rather than
rediscovering them. Cached assessment alone never authorizes admission.
