---
applies_to:
  - packages/acn/src/local-provider-**
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/model-selection.ts
  - packages/acn/src/model-slot-**
  - packages/acn/src/local-model-**
  - packages/acn/src/boundary/**
  - packages/sdk/src/inference.ts
  - packages/icn-protocol/src/generated/**
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/storage/src/types/model-state.ts
  - packages/client-common/src/**/local-model*
  - web/src/components/local-model-onboarding.tsx
  - web/src/components/model-center.tsx
---

# Model offerings and selection

Generic provider code sees `(ProviderId, ProviderModelId)` and a bound model. For the local provider,
`ProviderModelId` is the canonical `ModelId` unchanged for both catalog and `hf:` discovery rows.
ACN creates no alias, configuration identity, package identity, or bundle identity.

ACN publishes an offering only when the exact effective model is assessed `Fits` and its availability
is `Selectable`. The offering carries the assessed profile and capabilities. Models without that
evidence may remain visible in the model center but are absent from selectable provider offerings;
ACN never fills required provider fields with zero or false placeholders.

The local offering set is authoritative only after inventory and discovery reconciliation complete
and every current assessment is terminal. Startup and reassessment therefore cannot turn temporary
offering absence into removal of a persisted Slot selection.

Catalog installation and removal address `/api/v1/catalog/models/{modelId}/**` with the complete
catalog-form `ModelId`. Cancellation and failure acknowledgement address the admitted
`CatalogInstallationOperationId` under `/api/v1/catalog/installations/**`. These actions apply only
to catalog models. Discovered models are externally owned observations and expose no managed
lifecycle.

A Slot persists only provider ID, provider-model ID, and normalized reasoning effort. Assignment
validates a current offering. Slots express selection, while ICN owns load admission, replacement,
physical `ModelInstanceId`, and request leases. Temporary offering loss preserves a selection until
authoritative provider state proves it invalid according to Slot policy.

## Conformance

- Catalog and discovered offerings preserve canonical `ModelId` unchanged.
- Only evidence-backed assessed models become offerings.
- Catalog commands never accept discovered identities as managed acquisitions.
- Slots never persist package, bundle, installation-operation, assessment, or instance identity.
- Update availability does not disable a still-runnable installed catalog model.
