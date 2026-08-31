---
applies_to:
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-models/**
  - packages/icn/src/models/**
  - packages/icn/src/events/**
  - packages/icn/src/instances/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
  - web/src/components/model-center.tsx
---

# Local-model product projection

## Native boundaries

ICN exposes two explicit model resources:

- `/api/v1/catalog/**` publishes reviewed catalog variants and model-addressed managed installation
  operations.
- `/api/v1/discovery/**` publishes non-catalog models observed in external sources and supports
  explicit refresh.

There is no combined native management model list and no package, bundle, or raw-download resource
at the ICN–ACN boundary. The OpenAI-compatible `/v1/models` endpoint is inference advertisement,
not management state.

Each row already carries a valid canonical `ModelId`. Catalog rows exist without local material and
have `NotInstalled` or `Installed` local state. Discovered rows exist because material was observed
and have exactly one truthful state: `Ready` or `Unavailable`; both contain the selected resolved
installation.

## ACN projection

ACN observes catalog, discovery, catalog-installation operations, assessments, and instances
separately, then publishes one `LocalModel` union:

- `Catalog` contains catalog membership and managed acquisition state.
- `Discovered` contains discovery state and never exposes managed acquisition actions.

Shared fields contain presentation and canonical `modelId`. Catalog rows directly contain their
catalog data, acquisition state, and serving state. A discovered row contains one structural state:
`Ready` owns its resolved installation, residency, catalog attribution, and serving state;
`Unavailable` owns its resolved installation and native failure. An unavailable discovery has no
catalog-attribution fact because its selected artifact is not ready.
Presentation may include deduplicated HTTPS source links, but never package or bundle structure.
Where present, serving state is `Assessing`, `Failed`, or `Assessed`. Native catalog/discovery rows
already carry a complete desired, effective, or unavailable resolution, so ACN never fabricates an
intermediate resolving row. A failed ready discovery retains the selected serving profile. A
catalog failure retains a profile only when one was resolved. An assessed state contains metadata,
capabilities, and one assessment that owns its exact profile. A fitting catalog assessment alone
contains ranking scores; discovered and non-fitting states cannot contain them. Provider
selectability is derived from these structural facts and is never stored as a second state machine.
An assessed state also carries the optional speculative method as a direct serving fact; ACN does
not reconstruct it from private packages.

Catalog rows retain the desired configuration's total storage bytes as a direct product fact, so
size remains available before assessment or installation without exposing constituent packages.
Catalog acquisition state is model-level: installation bytes, ownership, progress, failure,
update availability, and residency appear only in variants where they are meaningful. A primary
path exists only on a resolved installation; genuinely ambiguous installed target material is an
unresolved installation and does not fabricate one.
Package lists and native occurrence IDs never enter the product. Installation and update failures
retain their typed native variants, including required and available bytes for insufficient disk
space; they are not flattened into diagnostic strings.

ACN publishes assessment results only while both catalog and discovery revisions and the local
hardware cycle still match the request. A superseded result cannot overwrite a newer source
snapshot. Derived projections may be materialized for cost, but source revisions and native state
remain authoritative.

## Conformance

- Catalog and discovered models share `ModelId` without sharing lifecycle semantics.
- Every callable external Hugging Face artifact appears once under its `hf:` identity.
- Invalid and incompatible discovery outcomes remain visible but never become offerings.
- An externally owned discovery can be selected and loaded but cannot be installed, updated, or
  removed through catalog commands.
- Provider offerings contain no fallback profile, capability, ranking, package, or bundle data.
- Clients never correlate package collections or reconstruct model identity.
