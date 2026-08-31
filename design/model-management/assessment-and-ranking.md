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

ICN owns material resolution, compatibility, memory, placement, performance, filtering, scheduling,
cache reuse, native concurrency, and deadlines. ACN owns assessment demand, request correlation,
coherent-inventory publication, and catalog ranking. Clients only render the result.

Assessment is addressed without exposing material:

- catalog requests send catalog revision, complete `ModelId`, explicit `Desired` or `Effective`
  selection, and requested profiles;
- discovery requests send discovery revision, complete `ModelId`, and requested profiles.

ICN validates the domain revision, resolves the selected model to exact private material, and emits
request-correlated streamed results. Packages, bundles, target IDs, and serving configurations do
not cross the boundary.

ACN assesses catalog desired material when not installed and effective material when installed. It
assesses only `Ready` discoveries. Each source cycle starts with exactly its current demands, and a
result is published only if both source revisions and the hardware cycle still match. Removed or
superseded models therefore cannot retain stale results.

A hardware assessment change starts a new cycle even when the catalog and discovery revisions are unchanged. ACN
also guards publication by its local cycle generation, so a slower result from the
prior hardware environment cannot overwrite the newer cycle. It accepts a stream only when start,
revision, target count, echoed subject and selection, profiles, uniqueness, environment, and
completion all correlate with the admitted request.

`Fits`, `DoesNotFit`, and `Incompatible` are genuine terminal evidence. Transport or operation
failure is `Failed`, not a compatibility result. Hardware observations that were not performed are
represented as `NotObserved`; zero-valued headroom is never fabricated.

Ranking exists only for reviewed catalog models with `Fits` evidence and the required bounded
performance sample. Intelligence and fidelity come from authored catalog evidence; speed comes
from native assessment. Missing evidence yields absent ranking scores, never zeros. Discovered
models receive no invented intelligence or fidelity score.

Provider selection requires `Fits`, current selectability, profile, and capabilities from the same
assessed state. Loading repeats native planning against current resources; cached assessment never
authorizes admission.
