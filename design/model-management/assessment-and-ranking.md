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

ICN owns material resolution, assessment demand, compatibility, memory, placement, performance,
filtering, scheduling, cache reuse, native concurrency, deadlines, and assessment publication. ACN
owns catalog ranking and projects native assessment state. Clients only render the result.

The automatic ICN pool assesses catalog desired material when not installed, effective material
when installed, and only `Ready` discoveries. It publishes a read-only revisioned snapshot with
independent catalog and discovery source slices. Exact work identity guards publication, so removed
or superseded models cannot retain stale results. Packages, bundles, and serving configurations do
not cross the boundary. Retryable target and source failures are observed as failures and retried
with bounded background backoff while their exact demand remains current.

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
