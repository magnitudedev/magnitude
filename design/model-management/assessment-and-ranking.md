---
applies_to:
  - inference/service/models/**
  - inference/service/server/src/assessment/**
  - inference/catalog/**
  - packages/icn/src/hardware/**
  - packages/icn/src/models/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
---

# Model assessment and ranking

The inference service owns assessment-material resolution, assessment demand, filtering,
scheduling, cache reuse, concurrency, deadlines, and assessment publication. The inference engine
computes template-derived capabilities, compatibility, memory fit and performance. ACN owns catalog
ranking and projects assessment state. Clients only render the result.

Assessment Material is the immutable GGUF header evidence required to assess a model without
installed tensor payloads: each component's exact header, through the aligned tensor-data offset,
with its component roles and content identities. The header carries the tokenizer, chat templates,
metadata and tensor directory the engine reads; a companion projector header lets the engine
establish vision without an authored capability flag. Catalog desired, catalog effective, and
discovered targets resolve this same input shape. A catalog serving profile's context is the
target's supported maximum context, the context the engine serves and assesses; memory fit uses
`min(context, 100_000)`.

Temporary assessment files preserve the artifact's logical size without allocating its absent
tensor payload. Windows explicitly marks these files sparse before extending them; unsupported
filesystem operations fail that assessment rather than consuming model-sized disk space.

The automatic assessment pool assesses catalog desired material when not installed, effective material
when installed, and only `Ready` discoveries. It publishes a read-only revisioned snapshot with
independent catalog and discovery source slices. Exact work identity guards publication, so removed
or superseded models cannot retain stale results. Packages, bundles, and serving configurations do
not cross the boundary. Whole-source failures are observed and retried with bounded background
backoff. Each exact target is attempted once; any target failure settles as `Dropped`, is never
retried, and is omitted by ACN. Catalog drops emit an OpenTelemetry error; discovered drops are
silent.

One assessment reads the target's Assessment Material headers once, in the service process.
Planning (header, family definition and execution plan) yields each known target's measurement
classes; the measurement job then measures only the classes the stored basis lacks (the classes come
from headers, never a hand-written list) while tokenizer, template, reasoning and resource
preparation finish, and times them once that preparation has ended. The engine's own tokenizer,
template and reasoning inspection supplies capabilities and the template fingerprint; its method
resolution decides speculative execution. Only the final support, memory-fit and performance
calculation waits for the basis.
One deadline covers the target, and one flat assessed result
publishes capabilities, template fingerprint, and profile evidence together. There is no planning
worker, template worker, inventory capability state, or post-download assessment gate. Equal
immutable tokenizer and template content may share derived preparation within the process without
becoming a separate capability authority.

`Fits`, `DoesNotFit`, and `Incompatible` are genuine terminal evidence. Transport or operation
failure drops the target rather than fabricating compatibility evidence. Hardware observations
that were not performed are represented as `NotObserved`; zero-valued headroom is never fabricated.

Ranking exists only for reviewed catalog models with `Fits` evidence and the required bounded
performance sample. Intelligence and fidelity come from authored catalog evidence; speed comes
from engine assessment. Missing evidence yields absent ranking scores, never zeros. Discovered
models receive no invented intelligence or fidelity score.

Provider selection requires `Fits`, current selectability, profile, and capabilities from the same
assessed state. Package validation establishes only structural artifact validity and presence;
assessment is the sole published capability authority. Loading resolves the same engine
configuration from the installed package; the service verifies the loaded instance's template
fingerprint and modalities against the host's resolution. Admission plans memory against current
resources. Cached assessment alone never authorizes admission.
