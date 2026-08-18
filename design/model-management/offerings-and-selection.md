---
applies_to:
  - packages/acn/src/local-provider-**
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/model-selection.ts
  - packages/acn/src/model-slot-**
  - packages/acn/src/local-model-**
  - packages/acn/src/handlers.ts
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/storage/src/types/model-state.ts
  - packages/client-common/src/**/local-model*
---

# Model offerings and selection

## Provider boundary

Generic provider and agent code sees `(ProviderId, ProviderModelId)` and a bound model. It does not
see catalog packages, downloads, assessments, or reconciliation state.

The local provider projects each effective `ModelServingConfiguration` as one offering. For a
catalog product, it converts the structured `CatalogIdentity` to a provider-model ID at this
boundary. For a non-catalog model, the provider-model ID is its ICN-issued configuration ID.
Provider offerings and configurations are derived and are not persisted.

An offering is selectable only when its effective configuration is runnable, installed, currently
assessed as fitting, and published by the provider. Update availability does not disable the old
effective offering.

## Catalog reconciliation

Catalog acquisition and update use one idempotent mutation:

```text
ReconcileCatalogModel(CatalogIdentity)
  -> Current(providerModelId)
   | DownloadAdmitted(providerModelId, ModelDownloadId)
```

ICN compares the current filesystem and package affiliations with the current desired catalog
bundle. It acquires missing desired packages, waits until the desired bundle is complete, and then
removes only affiliated superseded packages. Removal passes through runtime ownership checks; a
package used by a live instance is retained and reconciliation remains available for retry.

The old effective target remains runnable throughout acquisition. Failed or cancelled acquisition
does not remove it. Reconciliation writes artifacts and non-derivable package affiliations only; it
does not persist configurations, manifests, versions, or product state.

## Slot selection

A slot persists only provider, provider-model identity, and normalized reasoning effort. Assignment
validates a current offering. Loading resolves the selection again and submits its exact current
configuration to ICN.

Temporary unavailability preserves a selection. Local offerings expose whether ICN's model
inventory has completed its initial authoritative observation; an empty local list before that
point cannot invalidate a selection. Once a successful, ready provider projection proves the
identity absent, ACN clears the selection durably. A slot with an unknown selected model is not a
representable published state.

## Conformance

- Catalog identity is structured and branded outside the provider boundary.
- Reconciliation is the sole catalog install/update mutation.
- Reconciliation is idempotent and exact-package based.
- A runnable prior target remains available during update.
- Superseded cleanup never deletes an unaffiliated or live package.
- Startup ordering cannot clear a local selection before local offerings are authoritative.
- Slot state never publishes a selection that authoritative provider state proves invalid.
