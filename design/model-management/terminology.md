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

This document defines canonical local-model vocabulary. An unqualified `model` is presentation
language, not an identity-bearing domain type.

## Package and serving terms

| Term | Meaning |
|---|---|
| **Model file** | One immutable content-identified file with a role such as weights, shard, projector, draft, or MTP. |
| **Model package** | One immutable set of exact files, roles, relationships, inspected properties, and one source. |
| **Installed package** | The current observation that every required valid file for one package is present in a configured artifact root. |
| **Servable model bundle** | The complete structure that can be served: one standalone package or one method-identified speculative bundle with an embedded or separate draft. |
| **Serving profile** | Serving intent for a bundle, currently its context length. |
| **Model serving configuration** | One exact bundle plus one serving profile, constructed and canonically identified by ICN. |
| **Model download** | One process-local admitted acquisition occurrence for an exact bundle. Its identity is stable only for the owning ICN process lifetime. |
| **Download attempt** | One process-local attempt to acquire one exact package, private to ICN coordination. |

Installed is not an entity, lifecycle record, or durable state transition. Package and bundle
presence are recomputed from current valid files. Historical acquisition state never contributes
presence.

The bundle is structural data, not an independently identified entity:

```text
ServableModelBundle
  +-- Standalone(package)
  +-- SpeculativeDecoding(target, method, Embedded | Separate(draft))
```

Two bundles are equal when their tag, target package, speculative method, draft-source variant, and
separate draft package are equal. ICN alone constructs configuration identity from exact bundle and
profile. Private bundle or cache keys are not serialized product identities.

## Assessment terms

| Term | Meaning |
|---|---|
| **Hardware calibration** | Recomputable model-free performance evidence for one native hardware/backend environment. |
| **Model assessment** | Recomputable compatibility, capacity, memory, and performance evidence for one exact serving configuration. |
| **Semantic assessment key** | The configuration, immutable package evidence, stable hardware environment, native build, and assessment policy that determine reuse. |
| **Standard profile rule** | ACN's canonical, stable rule for an inspected standalone package without a catalog configuration. |
| **Resolved execution plan** | Load-time native allocation evidence; it is not serving intent or durable identity. |

Assessment predicts whether a configuration normally fits. Load admission decides whether it may
run now. Cached assessment never authorizes loading.

## Catalog, offering, and runtime terms

| Term | Meaning |
|---|---|
| **Recommendable model** | One active catalog configuration plus presentation, capabilities, license, and ranking data. |
| **Deprecated catalog entry** | An issued configuration excluded from recommendation and first-time discovery while remaining resolvable by identity. |
| **Local model** | One client-facing product row for an exact bundle, with acquisition and serving state. |
| **Provider offering** | A provider-facing projection of one currently resolved serving configuration. |
| **Slot selection** | The user's durable provider-qualified choice and reasoning effort for one product role. |
| **Model slot** | Durable role intent joined with current availability, actions, and optional instance state. |
| **Model instance** | One physical admitted occurrence of a serving configuration in ICN. |

Catalog membership contributes configuration and metadata. It implies no package presence,
selection, or residency. A provider offering is derived from catalog configuration, the canonical
standard-profile rule, current artifacts, and assessment. It is not persisted.

## Durability boundary

Magnitude model state stores slot selections, provider-qualified recency, and favorites. Onboarding
state stores its own completion fact. Model files are authoritative artifacts. Catalog
configurations are release data. Package inventory, serving configurations, offerings, assessments,
download occurrences, presentation, and instances are derived or process-local and are not copied
into model state.

A slot persists only `(ProviderId, ProviderModelId, reasoningEffort)`. For the local provider,
`ProviderModelId` represents the canonical configuration identity. The catalog resolves issued
configuration identities, including deprecated entries. Standard configurations are reconstructed
from inspected packages and the canonical standard-profile rule. That rule does not reinterpret an
existing package identity across releases. If an identity cannot be resolved, the
slot remains explicit unresolved user intent; it is never substituted silently.

## Product projection

Every active catalog bundle and every independently servable installed package is represented.
Exact bundle identity coalesces catalog, artifact, acquisition, assessment, recommendation, and
provider facts into one row. Configuration resolution is exact catalog configuration first,
otherwise ICN-issued standard configuration for an inspected standalone package.

## Identity map

| Identity | Identifies | Lifetime and owner |
|---|---|---|
| `ModelPackageId` | One immutable package | Stable, ICN |
| `ModelServingConfigurationId` | One exact bundle/profile combination | Stable, ICN-issued catalog or policy result |
| `ModelDownloadId` | One exact-bundle acquisition occurrence | ICN process lifetime |
| `DownloadAttemptId` | One package acquisition attempt | ICN process lifetime, private |
| `ModelInstanceId` | One physical loaded occurrence | ICN controller lifetime |
| `(ProviderId, ProviderModelId)` | One provider offering identity | Provider boundary; selection may retain it |
| `SlotId` | One product role assignment | Durable, ACN |

Display names, paths, repositories, filenames, recommendation membership, cache keys, and array
positions are never operational identities.

## Canonical relationship

```text
catalog or inspected package + serving policy
                    -> ModelServingConfiguration
                    -> assessment
                    -> provider offering
                    -> slot selection -> model instance

current files -> installed packages ----+
catalog + acquisition + assessment -----+-> LocalModel[]
recommendations + offerings + memory ---+
```
