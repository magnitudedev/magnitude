---
applies_to:
  - packages/acn/src/local-models.ts
  - packages/acn/src/local-model-assessor.ts
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - packages/client-common/src/local-models/**
  - cli/src/features/local-inference/**
  - cli/src/features/model-menus/**
  - cli/src/features/model-setup/**
---

# Local-model product projection

This document defines the ACN-owned local-model read model and its client contract. Terms follow
[Model-management terminology](./terminology.md).

## Boundary

Inventory discovery, artifact inspection, configuration assessment, model download, and provider
offering projection are separate backend mechanisms. Clients do not join those mechanisms or infer
lifecycle from absent fields.

```text
installed package inventory --+                         +--> installed models
artifact inspection ----------+--> ACN reconciliation -+--> downloads
configuration assessment -----+                         +--> recommendations
provider offerings -----------+
catalog and retained config --+
```

`LocalModelsState` contains three explicitly different collections:

- `models` contains only fully installed servable bundles;
- `downloads` contains configuration-keyed download operations that have begun or completed;
- `recommendations` contains assessed catalog discovery and recommendation state.

Catalog membership, retained configuration, assessment demand, and download activity cannot create
an entry in `models` without installed package evidence.

## Installed models

Each `LocalModel` is grouped by servable-bundle tag and ordered package identities. It contains:

- exact bundle structure;
- display presentation;
- installed bytes and one or more installation origins;
- explicit readiness.

Both `Magnitude` and `HuggingFaceCache` are installed origins and are equally runnable. Origin
affects presentation and ownership-sensitive operations, not eligibility.

Readiness is exactly one of:

- `Assessing`: the backend has installed package evidence but no publishable terminal
  configuration result;
- `Failed`: package validation or compatibility produced the included exact failure;
- `Assessed`: capabilities, one decided serving configuration, and its terminal result are present.

The configuration result is `Fits`, `DoesNotFit`, `Incompatible`, or `Failed`. Clients never infer
additional product states from absent fields or empty collections.

The backend decides exactly one serving configuration for each installed bundle, in this order:

1. the bundle's retained configuration;
2. otherwise its exact catalog configuration;
3. otherwise its standard configuration derived from inspected package facts.

Retained and catalog configurations may coexist as backend inputs, but they never become parallel
product rows or parallel choices for one bundle. Retaining another configuration for the same
bundle replaces the previous retained decision.

```text
derived configurations --set----+
catalog configurations --replace+--> Map<bundle identity, serving configuration>
retained configurations -replace+
```

The map value is the configuration itself. Source provenance is not retained or published.

Configuration durability and generation are defined in the
[terminology durability boundary](./terminology.md#durability-boundary). In particular, package
origin is irrelevant: catalog matching selects the exact catalog configuration, while only an
installed, inspected, non-catalog standalone bundle receives a generated standard configuration.
Generation is disposable observation; selection/installation materializes the decision durably.

An installed independently servable package always contributes a standalone model. It does not
need catalog membership, a retained configuration, or an existing provider offering. Catalog
metadata may enrich presentation only for the exact bundle.

## Backend lifecycle

Inventory reconciliation frequently establishes current package presence. Artifact inspection is
cached by content and observation evidence. Configuration assessment is separately keyed by exact
configuration, package material, stable hardware environment, native build, and assessment policy.

An unchanged inventory observation reuses inspection and assessment results. It does not publish
`Assessing` again. Assessment runs only when its semantic evidence changes or no terminal result is
available. Reconciliation is serialized, invalidations are coalesced, and completion rechecks the
semantic key before publication.

If installed files disappear, the backend removes the corresponding model. If the files remain but
inspection finds invalid or incompatible content, the model remains visible with the exact failure.
Transient inventory implementation stages never cross the local-model product boundary.

Assessment and provider projection start in scoped background fibers. ACN readiness and the initial
installed-model snapshot do not synchronously wait for native assessment.

## Downloads

A download is keyed by the exact serving configuration and carries its bundle, presentation,
capabilities when known, and `state`. Its state is `Downloading`, `Failed`, `Cancelled`, or `Downloaded`;
absence means no download is projected. `NotDownloaded` belongs to catalog candidate availability
and is not a download-operation state.

Download observation, cancellation, onboarding, and persistent footer status consume `downloads`
directly. They never infer download truth from recommendation readiness or installed
model membership.

## Client behavior

The Models page renders every entry in `models` and no uninstalled catalog entries. A model whose
decided configuration `Fits` is selectable. `Assessing`, `DoesNotFit`, incompatibility, and failures remain visible
as nonselectable installed rows with their precise reason or requirements.

The Catalog page renders eligible catalog candidates. It owns discovery of models that are not
installed. Moving between screens or remounting a query consumer cannot clear the last successful
server snapshot.

Clients render the readiness and download-state unions exhaustively. They do not derive status from
empty arrays, optional capabilities, absent offerings, absent memory, or recommendation lifecycle.

## Conformance

- Every independently servable installed package appears in exactly one installed-model bundle.
- No uninstalled catalog or retained configuration appears in `models`.
- Hugging Face cache packages appear as installed models.
- Catalog removal cannot hide an installed package.
- Invalid installed artifacts remain visible with the exact failure.
- Each installed bundle produces exactly one `LocalModel` and one decided serving configuration.
- `Assessed` always contains capabilities, that configuration, and exactly one terminal result.
- Inventory refresh does not reopen assessment for unchanged semantic evidence.
- Download workflows observe `downloads`, never recommendations or model absence.
- Clients render only backend-owned readiness, results, and exact failures.
