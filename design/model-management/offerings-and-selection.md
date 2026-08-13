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
  - packages/client-common/src/local-models/**
  - packages/client-common/src/model-slots/**
  - packages/client-common/src/onboarding/**
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - cli/src/features/model-setup/**
  - cli/src/features/model-menus/**
---

# Model offerings and selection

This document defines how retained local configurations project into provider choices and durable
slot selections. Terms follow [Model-management terminology](./terminology.md).

## Provider boundary

Generic provider and agent code sees an ordinary `(ProviderId, ProviderModelId)` and bound model. It
does not see packages, downloads, assessments, recommendation-policy inputs, native plans, or residency.

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
Provider-catalog entries also carry an optional presentation-only variant label. Local offerings
publish it from the shared ACN local-model presentation resolver; remote and custom offerings omit
it unless their provider owns equivalent structured metadata. Slot descriptors copy this field so
status surfaces never recover it from a display string.

## Offering availability

Configuration retention and offering availability are different facts. A configuration remains
durable while its packages are absent, downloading, being inspected, or temporarily unavailable.
Its provider offering remains present, while its provider-catalog projection is enabled only when
every required package is installed and its exact configuration has a current `Fits` assessment.

Assessment publishes per-configuration results independently of retained configuration existence.
Provider projection observes retained configurations and authoritative package and assessment state;
it never materializes, defaults, substitutes, or rewrites configurations. Catalog-only choices and
unretained standalone packages are not provider offerings. Their visibility belongs to
[Local-model product projection](./local-model-product-projection.md), not to provider availability.

## Selecting a catalog option

A catalog configuration is visible before retention on its unified local-model row. Only a
completed eligible assessed row permits acquisition; it introduces no identity beyond its exact
configuration. The client-owned
selection pipeline is:

```text
InstallModel(configurationId)                         -> AlreadyInstalled(providerModelId)
                                                     | DownloadAdmitted(providerModelId, ModelDownloadId)
  -> AssignSlot(slotId, providerModelId)
  -> LoadModel(slotId, exact selection)              [if residency is requested]
```

Installation resolves a retained configuration first or a catalog configuration for first adoption,
durably retains it, and privately admits acquisition for its exact bundle. Retaining a different
configuration for the same bundle replaces the prior retained configuration; it does not create a
second choice for the same installed model. Assignment stores the
resulting provider identity as durable slot intent. Loading acts only on that slot and resolves the
same retained configuration. Load admission atomically verifies that the slot still contains the
requested selection, so concurrent reassignment is rejected instead of loading its replacement.
`AssignSlot` also carries reasoning effort; the shorthand highlights
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

Assignment acknowledges after the durable selection and published configuration commit. When that
commit displaces the final slot using a resident local instance, the model-slot owner starts exact
instance replacement cleanup in its own service scope. That cleanup does not delay assignment
acknowledgement and cannot change the committed selection; its lifecycle remains observable through
the slot and instance authorities.

## Composite client workflows

Onboarding model setup is owned by one client-side service in client-common. The service owns
only its in-memory causal state: the explicit configuration choice and exact admission, provider,
selection, and instance identities returned by its composed operations. It exposes one passive
derived state atom and explicit start, cancel, and skip Effects. Those commands execute in the
connection's existing Effect runtime and compose the lower domain services' semantic Effects.

The start command follows exactly one of these paths:

```text
exact chosen instance already ready -> complete onboarding
installed choice -> assign -> load -> await exact instance ready -> complete
uninstalled choice -> install -> await exact selectable installation -> assign -> load
                   -> await exact instance ready -> complete
```

Mutation synchronization validates that each exact admission, selection, or instance is visible in
the affected canonical query before dependent work begins. Background completion is observed by
the same model-download or instance identity through `LocalModels` or `ModelSlots`. The service state joins
those identities with canonical model and slot snapshots, so progress and lifecycle remain
backend-owned and are never copied into service state.

Download admission synchronization requires the chosen configuration to expose either the exact
admitted model-download identity or installed acquisition. Provider publication is a later readiness
condition: physical installation with `Preparing` availability remains a waiting state, and
assignment begins only when that same configuration is `Selectable` with the provider-model
identity returned by admission. Publication order therefore cannot become model replacement or a
signal to restart setup.

For an admitted multi-package download, package components may complete independently before the
first synchronized snapshot. The aggregate model row continues to expose the one `ModelDownloadId`
returned by admission; synchronization never depends on internal package-attempt membership.

Opening or remounting onboarding only reads the service state. It cannot select or load a model.
Only an explicit configuration-ID submission starts setup; current slot state and mutation history
are never choice inputs. Component unmount does not interrupt active work. Process restart returns
to passive choosing and does not reconstruct intent from surviving backend facts.

Cancellation signals the one active continuation. After an admission it invokes the ordinary
exact-model-download cancel or exact-instance stop mutation and waits for that mutation's exact
query synchronization. It never searches mutation history, targets a current-but-unowned slot
instance, or clears durable slot selection.

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
- Installing a configuration retains it before assignment and installation admission returns.
- Catalog row identity is not persisted as user intent.
- Assignment commits durable selection and published configuration atomically.
- Selection matching always uses one concrete provider-qualified offering identity.
- Selection, acquisition, assignment, and loading remain distinct mutations even when composed by a client.
- Composite client interactions carry dependencies through function outputs; service state retains
  only causal identity and never a parallel model lifecycle.
- Generic provider code remains independent of local-model management concepts.
