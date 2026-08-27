---
applies_to:
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-models/**
  - packages/icn/src/models/**
  - packages/icn/src/events/**
  - packages/icn/src/downloads/**
  - packages/icn/src/installed/**
  - packages/icn/src/instances/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/inference-projection.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
  - cli/src/features/local-inference/**
  - cli/src/features/model-*/**
---

# Local-model product projection

This document defines the authoritative model read boundary and the client-facing local-model
projection.

## ICN model boundary

ICN publishes one `Models` snapshot. It contains every catalog-defined callable model with one
canonical `id`, its desired configuration, and current local state. Installed artifacts are read
from the separate Packages resource; a package is not promoted to a callable Model merely because
it exists on disk. ACN does not reconstruct catalog products from a second public catalog endpoint.

A catalog model is identified by one `CatalogIdentity`. Its local state is:

- `NotInstalled`; or
- `Installed`, carrying the current installation and its update state. The installation contains
  the effective configuration or an explicit unavailability failure and all present affiliated
  packages. The update state is `Current` or `Available`; `Available` contains missing desired
  package IDs and superseded package IDs.

The filesystem is the presence authority. An attributed target makes its catalog model installed
and visible even when desired dependencies are missing or the exact desired bundle has changed.
An unattributed or attribution-failed target remains visible in Packages and in Magnitude's
product projection, but is not advertised by the standard inference model list until ICN can
resolve it to a canonical callable model.

## Exact desired/current comparison

For one catalog identity:

```text
desired     = package IDs in the current catalog bundle
present     = package IDs observed on the filesystem
affiliated  = present package IDs affiliated with the catalog identity
missing     = desired - present
superseded  = affiliated - desired
```

The model has `updateState: Available` when either `missing` or `superseded` is non-empty. No
persisted version, manifest, configuration, or cached completeness flag participates.

Superseded packages remain present and runnable while `missing` is non-empty. Once an authoritative
installed-set change makes `missing` empty, catalog maintenance may remove the exact `superseded`
IDs. Request handlers and individual package attempts do not own this transition.

The desired configuration is effective when its whole bundle is present. Otherwise ICN keeps the
current desired target runnable as standalone, or one unique prior attributed target runnable as
standalone. Multiple prior targets with no current target produce a visible typed unavailability
failure; they are never hidden or guessed between.

## ACN product projection

ACN enriches the ICN model snapshot with assessment, provider publication, ranking scores, and
live memory observations. Clients consume one `LocalModel` row and do not correlate parallel
collections.

The configuration resolver carries the target's `ModelPackageInspection` with the exact effective
configuration. An installed target uses its package inspection. An uninstalled desired catalog
configuration uses the successful release-bound inspection of that same immutable package.
Capabilities exist only in the inspection's `Inspected` state; `Pending`, `Invalid`, and
`Incompatible` remain explicit states and are never collapsed into optional capability data.
Desired catalog capabilities are never substituted for a different installed fallback package or
for an installed package whose inspection has not succeeded. The local-model projection maps those
inspection states into its serving stages, and the provider publishes only inspected targets.
The resolver likewise carries the assessment state directly: absence before an assessment result is
represented by `Assessing`, not by an optional assessment beside the state machine.

`acquisitionState` is one flat union covering the model's whole materialization lifecycle:
`NotInstalled`, `Installing`, `InstallFailed`, `Installed`, `UpdateAvailable`, `Updating`, and
`UpdateFailed`. Every variant is a reachable product state, and each payload exists only under the
state it belongs to: transfer progress on `Installing`/`Updating`, an unacknowledged transfer
failure on the failed states, and — on every installed-family variant — the exact package
identity, filesystem path, and installation origin for every installed package plus the model's
runtime `residencyState`. An installed model remains selectable while an update is available,
transferring, or failed; the prior version's packages stay on the row until the update lands.
Native occurrence identities (download and instance IDs) never appear in the projection;
cancellation and failure acknowledgement are model-addressed commands that ACN resolves to the
native occurrence.

Catalog membership carries structured branded identity components and the catalog's complete
intelligence assessment, including direct or estimated provenance. ACN does not flatten or
reconstruct that provenance. Only the local provider adapter serializes identity components into a
provider-model identity. Configurations and product rows are derived and never persisted.

## Conformance

- Every active catalog model is represented in Models; every installed artifact is represented in
  Packages.
- One catalog identity produces one row across target, dependency, repository, and drafter changes.
- A prior target stays runnable until the desired bundle is complete.
- Any exact desired/current package difference produces update availability.
- Completed files, not download history or cached state, determine presence.
- Catalog or attribution failure cannot hide an installed target.
- Clients consume acquisition, serving, and ranking state from one row; they do not construct a
  second reconciliation lifecycle from mutation state.
- Every installed row carries the exact current filesystem location of every bundle package.
- Every assessed row and provider offering carries capabilities for its exact effective target.
