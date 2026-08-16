---
applies_to:
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-models/**
  - packages/icn/src/models/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
  - cli/src/features/local-inference/**
  - cli/src/features/model-*/**
---

# Local-model product projection

This document defines the authoritative model read boundary and the client-facing local-model
projection.

## ICN model boundary

ICN publishes one `Models` snapshot. It contains every catalog model with its desired
configuration and current local state, plus every independently servable installed target not
attributed to a catalog model. ACN does not reconstruct catalog products by joining a catalog list
to a separate installed-package list.

A catalog model is identified by one `CatalogIdentity`. Its local state is:

- `NotInstalled`; or
- `Installed`, carrying the current installation and its update state. The installation contains
  the effective configuration or an explicit unavailability failure and all present affiliated
  packages. The update state is `Current` or `Available`; `Available` contains missing desired
  package IDs and superseded package IDs.

The filesystem is the presence authority. An attributed target makes its catalog model installed
and visible even when desired dependencies are missing or the exact desired bundle has changed.
An unattributed or attribution-failed target remains visible as a standalone installed model.

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

ACN enriches the ICN model snapshot with assessment, provider publication, recommendations, and
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

`acquisitionState` describes first acquisition. `upgradeState` is `NotApplicable`, `Current`,
`Available`, `Upgrading`, or `Failed`. An installed model may remain selectable while its upgrade
state is `Available` or `Upgrading`. Installed acquisition carries the exact package identity,
filesystem path, and installation origin for every package in the effective bundle. An installed
product row without a complete, non-empty set of package locations is invalid.

Catalog membership carries structured branded identity components. Only the local provider adapter
serializes them into a provider-model identity. Configurations, product rows, and upgrade state are
derived and never persisted.

## Conformance

- Every active catalog model and independently servable installed target is represented.
- One catalog identity produces one row across target, dependency, repository, and drafter changes.
- A prior target stays runnable until the desired bundle is complete.
- Any exact desired/current package difference produces update availability.
- Completed files, not download history or cached state, determine presence.
- Catalog or attribution failure cannot hide an installed target.
- Clients consume acquisition, reconciliation, serving, and recommendation state from one row.
- Every installed row carries the exact current filesystem location of every bundle package.
- Every assessed row and provider offering carries capabilities for its exact effective target.
