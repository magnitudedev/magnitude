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

This document defines derived local provider offerings and durable slot selections. Terms follow
[Model-management terminology](./terminology.md).

## Provider boundary

Generic provider and agent code sees an ordinary `(ProviderId, ProviderModelId)` and bound model.
It does not see packages, downloads, assessments, native plans, or residency.

The local provider adapter projects each currently resolved `ModelServingConfiguration` as one
offering. For the local provider, the provider-model ID represents the ICN-issued configuration ID
within the `local` namespace. The brands remain distinct even though their string values match.

Configurations and offerings are not persisted. ACN derives them continuously from:

- issued catalog configurations, including deprecated entries required by durable references; and
- ICN-issued standard configurations from the canonical stable standard-profile rule for inspected
  independently servable packages not described by the catalog.

Capabilities, presentation, installation, and assessment are joined from their current authorities
and are not copied into offering state.

## Offering availability

An offering exists when its exact configuration is currently resolvable. Availability is derived
separately: it is enabled only when every required package is installed and the exact configuration
has a current `Fits` assessment. Missing files, pending inspection, insufficient resources, or
dependency failure disable the offering without deleting a durable slot selection that references
it.

An unresolved provider-model identity has no fabricated offering. Slots retain that identity as
unresolved user intent until its authority becomes available again. ACN never substitutes another
configuration by bundle similarity, filename, array position, or current recommendation.

## Installation

Installation accepts one current `ModelServingConfigurationId`. ACN resolves it from the current
configuration projection, validates exact `Fits` evidence, rechecks the same resolution at
admission, and asks ICN to acquire its bundle. It performs no durable configuration write.

```text
InstallModel(configurationId) -> AlreadyInstalled(providerModelId)
                               | DownloadAdmitted(providerModelId, ModelDownloadId)
```

The returned provider-model identity is deterministic from configuration identity. Download
identity is process-local and exists only for observation and cancellation of that occurrence.
Installation, assignment, and loading remain separate mutations.

## Slot selection

A slot selection is the user's durable choice of provider, provider model, and normalized reasoning
effort for one product role. It copies no configuration, package, source, assessment, or runtime
state. Favorites and recency use the same complete provider-qualified identity.

Assignment validates the exact current offering, normalizes reasoning effort, durably stores the
selection, and atomically publishes slot and agent-model configuration. Later unavailability does
not discard selection. Explicit user deletion may clear affected selections only when that deletion
operation promises to forget the chosen model, not merely because files disappeared.

Loading resolves the selected provider-model identity again through the current provider offering
and submits that exact configuration to ICN. The slot controller rechecks selection before native
admission so concurrent reassignment cannot load a replacement accidentally.

## Composite client workflows

Onboarding owns only process-local causal identities returned by its composed operations. It may
install, await the same configuration becoming selectable, assign, and load. It does not copy model
or configuration state.

Download synchronization uses the admitted process-local download identity or observes that the
same configuration is already installed. Process restart returns onboarding to passive choice;
catalog and artifact reconciliation reconstruct current models and offerings independently.

Cancellation addresses the exact active download or instance identity. It never searches mutation
history, guesses from files, or clears durable selection.

## Conformance

- Provider offerings are derived and have no persistence surface.
- Model state contains no serving configuration or bundle copy.
- Every offering maps one exact current configuration to one provider-model identity.
- Installation performs no durable write beyond artifact acquisition itself.
- Assignment persists only provider-qualified selection and reasoning effort.
- Unavailability preserves selection as unresolved intent and never substitutes another model.
- Loading re-resolves and validates the exact selected configuration.
- Download identities are used only within the owning process lifetime.
- Generic provider code remains independent of local-model management concepts.
