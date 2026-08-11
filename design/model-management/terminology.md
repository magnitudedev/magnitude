---
applies_to:
  - inference/crates/icn-contracts/src/**
  - inference/crates/icn-models/**
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/icn/src/downloads/**
  - packages/acn/src/local-model-**
  - packages/acn/src/local-provider-**
  - packages/acn/src/model-slot-**
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
---

# Model-management terminology

This document defines the canonical local-model vocabulary. An unqualified `model` is presentation
language, not an identity-bearing domain type.

## Package and serving terms

| Term | Meaning |
|---|---|
| **Model file** | One immutable content-identified file with a role such as weights, shard, projector, draft, or MTP. |
| **Model package** | One immutable set of exact files, roles, relationships, inspected properties, and one source. |
| **Servable model bundle** | The complete artifact structure that can be served: one standalone package or an ordered speculative-decoding pair. |
| **Serving profile** | Provider intent for serving a bundle, currently its context length. |
| **Model serving configuration** | One exact servable bundle plus one serving profile, constructed and canonically identified by ICN. |
| **Download attempt** | One admitted attempt to install one exact package. |

The bundle is structural data, not an independently identified entity:

```text
ServableModelBundle
  +-- Standalone
  |     package: ModelPackage
  |
  +-- SpeculativeDecodingPair
        target: ModelPackage
        draft: ModelPackage
```

`target` is the established speculative-decoding term for the primary model in a pair. It is not
the name or identity of the enclosing bundle.

Two bundles are the same when their tag and ordered package identities are the same. Components may
derive private canonical keys for maps and caches, but those keys are not configuration identity.
ICN alone derives canonical serving-configuration identity from the exact ordered bundle and
profile. Private bundle keys are not serialized product data and never cross the protocol as bundle
identity.

## Assessment terms

| Term | Meaning |
|---|---|
| **Hardware calibration** | Serializable model-free performance evidence for one native hardware/backend environment. |
| **Model assessment** | Compatibility, capacity, memory, and performance evidence for one exact serving configuration. |
| **Semantic assessment key** | The configuration, immutable package evidence, stable hardware environment, native build, and assessment policy that determine reuse. |
| **Assessing** | Ephemeral state for semantic assessment work currently owned by the assessor. |
| **Standard profile decision** | ACN's disposable decision to apply its standard serving profile to an inspected standalone package with no retained or catalog configuration. |
| **Eligible assessed configuration** | A configuration whose assessment is `Fits`. |
| **Resolved execution plan** | Load-time native allocation evidence; it is not serving intent or durable identity. |

Assessment predicts whether a configuration normally fits. Load admission decides whether it may
run now. Cached assessment never authorizes a load.

## Catalog and recommendation terms

| Term | Meaning |
|---|---|
| **Recommendable model** | One curated configuration plus presentation, capabilities, license, and ranking evidence. |
| **Recommendable model catalog** | Release-bound curated configurations eligible for assessment and recommendation. |
| **Catalog candidate** | A product projection of one catalog configuration with completed assessment and current acquisition state. |
| **Recommendation** | A policy-selected catalog candidate labeled with an intent and explanation. |

Catalog membership contributes metadata and a configuration. It implies no installation, retained
state, offering, selection, or residency. Candidate identity is the configuration identity.

## Offering and runtime terms

| Term | Meaning |
|---|---|
| **Retained configuration** | The exact serving configuration durably chosen for one bundle through installation or bounded recovery. At most one is retained per bundle. |
| **Configuration recovery** | One bounded initialization epoch that retains exact catalog configurations for already-installed bundles after model state defaults. |
| **Provider offering** | A provider-facing projection of one retained configuration. |
| **Slot selection** | The user's durable provider-qualified choice and reasoning effort for one product role. |
| **Model slot** | Durable role intent joined with availability, actions, and optional instance state. |
| **Model instance** | One physical admitted occurrence of a serving configuration in ICN. |

A retained configuration and offering may remain while packages are absent or unavailable. A slot
selection does not imply residency, and an instance does not own durable user intent.

## Durability boundary

Magnitude model state stores retained configurations, slot selections, recency, favorites, and the
configuration-recovery completion marker. Onboarding state stores only completion. Bundle keys,
catalog association, assessment lifecycle and results, provider offerings, package inventory,
download progress, presentation, and model instances are derived or externally authoritative and
are not stored in those documents.

For a standard standalone model, ACN decides whether its standard profile applies and supplies that
bundle/profile demand to ICN. ICN constructs and canonically identifies the corresponding serving
configuration. That ICN-issued configuration becomes durable only when selection or installation
materializes it. Catalog removal therefore cannot erase an installed package, retained
configuration, or user selection.

Installation origin does not determine durability. Magnitude-managed and Hugging Face cache
packages follow the same rules:

| Bundle case | Configuration used by the product | When it is stored in model state |
|---|---|---|
| Catalog bundle, not installed | Exact catalog configuration; no `LocalModel` exists | When installation is admitted |
| Installed bundle with a retained configuration | Exact retained configuration | Already stored |
| Installed bundle matching the catalog, with no retained configuration | Exact catalog configuration | On selection/installation, or during the single bounded recovery epoch |
| Installed non-catalog standalone bundle | ICN-issued configuration for ACN's standard profile decision after package inspection | Only when selected/installed |
| Speculative-decoding bundle | Exact retained configuration, otherwise exact catalog configuration | On selection/installation, or bounded recovery |

ACN makes a standard profile decision only for an installed, inspected standalone bundle having
neither a retained nor catalog configuration. ICN constructs the corresponding configuration; ACN
does not derive its identity or construct it independently. The result is disposable until
materialized. Magnitude does not generate speculative-decoding pairs or replacement catalog
configurations from package inventory.

Recovery does not generate a configuration. It copies an exact catalog configuration into model
state only when the exact bundle is already installed, no configuration is retained for that
bundle, and the one recovery epoch is still incomplete. Later discovery alone never writes model
state.

## Product projection

`LocalModel` is one disposable product row grouped by exact servable-bundle structure:

```text
LocalModel
  bundle: ServableModelBundle
  presentation
  installation
  readiness:
    Assessing
    | Failed(failure)
    | Assessed(
        capabilities,
        configuration: ModelServingConfiguration,
        offering?,
        assessment: Fits | DoesNotFit | Incompatible | Failed
      )

LocalModelDownload
  configuration: ModelServingConfiguration
  presentation
  capabilities?
  state: Downloading | Failed | Cancelled | Downloaded
```

Every independently servable installed package is represented as a `Standalone` bundle even when
it has no catalog entry or retained configuration. Each installed bundle has one product
configuration decision: retained first, otherwise exact catalog configuration, otherwise the
ICN-issued configuration for ACN's standard profile decision. Catalog association enriches the same
row with curated metadata; it does not decide whether the row exists.

## Identity map

| Identity | Identifies | Owner |
|---|---|---|
| `ModelPackageId` | One immutable package | ICN |
| `DownloadAttemptId` | One package-install attempt | ICN |
| `ModelServingConfigurationId` | One bundle/profile combination, canonically constructed and validated with its configuration | ICN |
| `ModelInstanceId` | One physical loaded occurrence | ICN |
| `(ProviderId, ProviderModelId)` | One provider offering | Provider boundary |
| `SlotId` | One product role assignment | ACN |

Display names, paths, repositories, filenames, recommendation membership, cache keys, and array
positions are never operational identity. For the local provider, the configuration ID is
represented in the `ProviderModelId` namespace without making the brands interchangeable.

## Canonical relationship

```text
ModelPackage(s) -> ServableModelBundle + ServingProfile
                                      -> ModelServingConfiguration
                                                   |
                         +-------------------------+------------------+
                         |                                            |
                         v                                            v
             LocalModelAssessor                           Install(configurationId)
                         |                                            |
                         v                                            v
                  assessment                               retained configuration
                         |                                            |
                         +-> catalog candidate                         +-> provider offering
                              +-> recommendation                       +-> slot -> instance

installed packages ----------------+--> LocalModel[]
inspection and assessment ---------+
provider offerings ----------------+

known configurations --------------+--> LocalModelDownload[]
download state --------------------+
```
