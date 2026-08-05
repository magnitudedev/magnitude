---
applies_to:
  - packages/acn/src/local-provider-**
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/model-configuration.ts
  - packages/acn/src/model-slot-**
  - packages/acn/src/local-model-auto-setup.ts
  - packages/acn/src/handlers.ts
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
  creation origin
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

Automatic reconciliation may create an offering from an installed target. While inspection or
assessment is incomplete, ACN retains the previous complete projection or withholds a new one. A
failure publishes typed unavailability with retryability; it is not rewritten as incompatibility.
Reconciliation never changes the offering's configuration to obtain a different assessment result.
Reconciliation runs under a scoped background owner and never delays ACN service publication.

## Selecting a catalog option

A catalog candidate is a presentation row for one exact assessed configuration and introduces no
identity. Catalog actions use its `ModelServingConfigurationId`:

```text
resolve exact assessed configuration
  -> persist exact provider offering
  -> acquire missing target packages, if requested
  -> return ProviderModelId and acquisition identities
  -> validate and assign slot
  -> load assigned configuration, if requested
```

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
unavailability without discarding user intent.

## Composite client workflows

Onboarding may compose selection, optional download, assignment, loading, completion, and explicit
cancellation as one client-owned workflow. Its transient state contains only the submitted choice
and exact command identities required to bridge admission. It does not duplicate download, slot, or
instance lifecycle.

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
- Catalog actions address configuration identity, never provider-model identity.
- Offering reconciliation never silently substitutes a configuration.
- Catalog row identity is not persisted as user intent.
- Assignment commits durable selection and published configuration atomically.
- Selection, acquisition, assignment, and loading remain distinct mutations even when composed by a client.
- Generic provider code remains independent of local-model management concepts.
