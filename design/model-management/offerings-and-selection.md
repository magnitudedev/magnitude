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
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - packages/client-common/src/**/local-model*
  - cli/src/features/model-menus/**
  - web/src/components/local-model-onboarding.tsx
  - web/src/components/model-center.tsx
---

# Model offerings and selection

## Provider boundary

Generic provider and agent code sees `(ProviderId, ProviderModelId)` and a bound model. It does not
see catalog packages, downloads, assessments, or reconciliation state.

The local provider projects each callable ICN Model as one offering. Its provider-model ID is the
Model's existing canonical model ID (for example `gemma-4-26b-a4b-it-qat:gguf:q4`) everywhere:
native Models, standard inference, slot selection, and provider invocation. ACN creates no alias or
configuration identity. Provider offerings and configurations are derived and are not persisted.

An offering is selectable only when its effective configuration is runnable, installed, currently
assessed as fitting, and published by the provider. Update availability does not disable the old
effective offering.

## Model installation

Acquisition and update use one idempotent native inference mutation:

```text
POST /api/v1/models/install { modelId }
  -> Current(modelId)
   | DownloadAdmitted(modelId, downloadId)
```

ICN compares the current filesystem and package affiliations with the desired Model
bundle. It acquires missing desired packages, waits until the desired bundle is complete, and then
removes only affiliated superseded packages. Removal passes through runtime ownership checks; a
package used by a live instance is retained and installation remains available for retry.

The old effective target remains runnable throughout acquisition. Failed or cancelled acquisition
does not remove it. Installation writes artifacts and non-derivable package affiliations only; it
does not persist configurations, manifests, versions, or product state.

## Slot selection

A slot persists only provider, provider-model identity, and normalized reasoning effort. Assignment
validates a current offering. Slots express durable selection, not physical residency. Explicit
warm loading and inference both send the selected canonical model ID to ICN; ICN resolves the
current configuration and owns admission, replacement, and request leases.

Temporary unavailability preserves a selection. Local offerings expose whether ICN's model
inventory has completed its initial authoritative observation; an empty local list before that
point cannot invalidate a selection. Once a successful, ready provider projection proves the
identity absent, ACN clears the selection durably. A slot with an unknown selected model is not a
representable published state.

## Conformance

- Catalog identity is structured and branded outside the provider boundary.
- Native Model installation is the sole install/update mutation.
- Model installation is idempotent and exact-package based.
- A runnable prior target remains available during update.
- Superseded cleanup never deletes an unaffiliated or live package.
- Startup ordering cannot clear a local selection before local offerings are authoritative.
- Slot state never publishes a selection that authoritative provider state proves invalid.
