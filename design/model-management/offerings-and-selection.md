---
applies_to:
  - packages/acn/src/local-provider-**
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/model-configuration.ts
  - packages/acn/src/model-slot-**
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/handlers.ts
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/storage/src/types/config.ts
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - cli/src/features/model-menus/**
---

# Model offerings and selection

This document defines how assessed local configurations become provider choices and durable slot
selections. Terms follow [Model-management terminology](./terminology.md).

## Provider boundary

Generic provider and agent code sees an ordinary `(ProviderId, ProviderModelId)` and bound model. It
does not see packages, downloads, assessments, catalog candidates, native plans, or residency.

ACN owns durable local provider offerings. Each offering contains:

```text
ProviderOffering
  provider identity
  provider-model identity
  exact ModelServingConfiguration
```

For the local provider, ACN deterministically derives the provider-model value from the ICN-issued
serving-configuration identity within the `local` provider namespace. The values remain different
branded concepts: configuration identity selects assessed configuration; provider-model identity
addresses an existing offering. Capabilities are resolved from authoritative catalog or
installed-package evidence and are not duplicated in the durable offering.

## Offering availability

Offering existence and availability are different facts. An offering remains durable while its
packages are absent, downloading, being inspected, or temporarily unavailable. Its provider-catalog
projection is enabled only when every required package is installed and its exact configuration has
a current `Fits` assessment.

Assessment publishes configuration candidates independently of offering existence. Offering
creation occurs only through explicit configuration selection. Provider projection observes the
persisted offering and authoritative package and assessment state; it never creates offerings.
Model-selection surfaces join installed targets with their eligible assessed configurations
independently of provider projection, so an installable choice remains visible before explicit
selection creates its offering. This presentation join creates no durable state.

## Selecting a catalog option

A catalog candidate is a presentation row for one exact assessed configuration and introduces no
identity. It carries the target identity used for acquisition and the serving-configuration
identity used for offering creation. The client-owned selection pipeline is:

```text
DownloadModel(targetId)                              [if packages are missing]
  -> CreateLocalModelOffering(configurationId)       -> providerModelId
  -> AssignSlot(slotId, providerModelId)
  -> LoadModel(slotId)                               [if residency is requested]
```

Each command receives the identity of the fact it owns. Download resolves and acquires one exact
offering target. Offering creation persists one exact assessed serving configuration. Assignment
stores the resulting provider offering as durable slot intent. Loading acts only on that slot and
its selected offering. `AssignSlot` also carries the provider ID and reasoning effort; the
shorthand above highlights the identities that determine each stage.

Before offering creation, neither the row nor the client workflow uses `ProviderModelId`. After
creation, provider and slot operations use `(ProviderId, ProviderModelId)` and do not use catalog
membership as authority. Persisted state contains the provider offering and slot selection, never
recommendation membership. Downloading addresses resolved target packages; loading addresses the
assigned serving configuration.

## Slot selection

A slot selection is the user's durable choice of provider, provider model, and normalized reasoning
effort for one product role. It references a provider offering and copies none of its package,
source, assessment, recommendation, or runtime state.

Assignment validates the exact offering before commit. A successful assignment means ACN has:

- confirmed the offering is assignable from current authoritative state;
- normalized reasoning effort against the provider model;
- durably stored the selection; and
- atomically published slot and agent-model configuration.

A rejected assignment leaves the previous selection unchanged. Assignment never creates a blocked
slot. Conditions may degrade after assignment; the slot then projects the authoritative
unavailability without discarding user intent. Authoritative deletion of an authored provider or
model is not temporary unavailability: it clears slots selecting that deleted identity without
substituting another model.

## Composite client workflows

Onboarding may compose target acquisition, offering creation, assignment, loading, completion, and
explicit cancellation as one client-owned workflow. Its transient state retains the submitted
choice through each finite mutation and contains only the exact command identities required to
bridge admitted work. It does not duplicate download, slot, or instance lifecycle.

Interruption or restart never reconstructs onboarding intent from server observations. Confirmed
cancellation invokes ordinary download-cancellation or slot-clear mutations. Successful load closes
setup; an externally stopped load is terminal presentation rather than an invitation to replay the
workflow.

## Favorites

A favorite is a durable preference over `(ProviderId, ProviderModelId)`. Favoriting never installs,
offers, selects, loads, or stops a model. An open selection menu retains its captured ordering;
preference and recency changes affect the next menu entry.

## Conformance

- Provider identity never depends on recommendation membership or package presence.
- Acquisition actions address target identity; offering-creation actions address configuration
  identity; neither uses provider-model identity before the offering exists.
- Provider projection never creates or substitutes an offering.
- Installed targets with an eligible assessed configuration remain visible in model-selection
  surfaces before offering creation.
- Selecting an installed target without an offering creates its exact assessed offering before
  assigning the slot.
- Catalog row identity is not persisted as user intent.
- Assignment commits durable selection and published configuration atomically.
- Selection, acquisition, assignment, and loading remain distinct mutations even when composed by a client.
- Generic provider code remains independent of local-model management concepts.
