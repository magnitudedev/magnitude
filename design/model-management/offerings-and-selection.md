---
applies_to:
  - packages/acn/src/local-provider-**
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/model-selection.ts
  - packages/acn/src/retained-model-configurations.ts
  - packages/acn/src/model-slot-**
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/handlers.ts
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/storage/src/types/model-state.ts
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - cli/src/features/model-menus/**
---

# Model offerings and selection

This document defines how retained local configurations project into provider choices and durable
slot selections. Terms follow [Model-management terminology](./terminology.md).

## Provider boundary

Generic provider and agent code sees an ordinary `(ProviderId, ProviderModelId)` and bound model. It
does not see packages, downloads, assessments, catalog candidates, native plans, or residency.

ACN owns one durable retained local configuration per bundle. The local provider adapter projects
each one as:

```text
ProviderOffering
  provider identity
  provider-model identity
  exact ModelServingConfiguration
```

For the local provider, ACN represents the ICN-issued serving-configuration identity within the
`local` provider namespace. The values remain different branded concepts: configuration identity
addresses retained local state, while provider-model identity addresses its generic provider
projection. Capabilities are resolved from current authoritative catalog or installed-package
evidence and are not persisted as another offering record.

## Offering availability

Configuration retention and offering availability are different facts. A configuration remains
durable while its packages are absent, downloading, being inspected, or temporarily unavailable.
Its provider offering remains present, while its provider-catalog projection is enabled only when
every required package is installed and its exact configuration has a current `Fits` assessment.

Assessment publishes catalog candidates independently of retained configuration existence.
Provider projection observes retained configurations and authoritative package and assessment state;
it never materializes, defaults, substitutes, or rewrites configurations. Catalog-only choices and
unretained standalone packages are not provider offerings. Their visibility belongs to
[Local-model product projection](./local-model-product-projection.md), not to provider availability.

## Selecting a catalog option

A catalog configuration is visible before retention. Only a completed eligible catalog candidate
permits installation; it introduces no identity beyond its exact configuration. The client-owned
selection pipeline is:

```text
InstallModel(configurationId)                        -> providerModelId + exact attempt IDs
  -> AssignSlot(slotId, providerModelId)
  -> LoadModel(slotId)                               [if residency is requested]
```

Installation resolves a retained configuration first or a catalog configuration for first adoption,
durably retains it, and privately admits acquisition for its exact bundle. Retaining a different
configuration for the same bundle replaces the prior retained configuration; it does not create a
second choice for the same installed model. Assignment stores the
resulting provider identity as durable slot intent. Loading acts only on that slot and resolves the
same retained configuration. `AssignSlot` also carries reasoning effort; the shorthand highlights
the identities that determine each stage.

After installation, provider and slot operations use `(ProviderId, ProviderModelId)` and do not use
catalog membership as authority. Persisted state contains exact serving configurations and slot
selection, never recommendation membership or provider-offering records.

## Slot selection

A slot selection is the user's durable choice of provider, provider model, and normalized reasoning
effort for one product role. It references a provider offering and copies none of its package,
source, assessment, recommendation, or runtime state.

Selection comparison requires both concrete provider and provider-model identities. An unassigned
slot, catalog-only configuration, unretained model, or unavailable observation has no substitute
selection key; absence never compares as identity.

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

Onboarding may compose configuration installation, assignment, loading, completion, and
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
- Product installation addresses configuration identity and returns its local provider-model
  representation; bundle structure remains private to package acquisition.
- Provider projection is exactly one-to-one with retained configurations and never creates durable
  model state.
- At most one retained configuration and provider offering exist for one bundle.
- Catalog configurations and unretained standalone packages remain visible according to the product projection
  without becoming provider offerings.
- Installing a configuration retains it before assignment and acquisition admission returns.
- Catalog row identity is not persisted as user intent.
- Assignment commits durable selection and published configuration atomically.
- Selection matching always uses one concrete provider-qualified offering identity.
- Selection, acquisition, assignment, and loading remain distinct mutations even when composed by a client.
- Generic provider code remains independent of local-model management concepts.
