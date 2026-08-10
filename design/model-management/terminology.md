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
  - packages/storage/src/types/config.ts
---

# Model-management terminology

This document defines the canonical vocabulary for Magnitude's local-model domain. More specific
documents define behavior and lifecycle. An unqualified `model` is presentation language, not an
identity-bearing domain type.

## Artifact and acquisition terms

| Term | Meaning |
|---|---|
| **Model file** | One immutable, content-identified file with a role such as weights, shard, projector, draft, or MTP. |
| **Model package** | One immutable bundle of exact model files, roles, relationships, properties, and one source. |
| **Model package source** | One exact retrieval location. A mutable repository name or revision is not an exact source. |
| **Speculative decoding pair** | An ordered target package and draft package with an explicit speculative method. |
| **Model offering target** | The complete installable unit exposed for inference: one package or one speculative pair. |
| **Download attempt** | One admitted attempt to install one exact package. Failure belongs to the attempt, not permanently to the package. |

Package, target, and download-attempt identity are distinct. Installation changes local presence;
it never changes package or target identity.

## Configuration and assessment terms

| Term | Meaning |
|---|---|
| **Serving profile** | Model-agnostic provider intent, currently the maximum context for one request. |
| **Model serving configuration** | One exact offering target plus serving profile, with an ICN-issued stable identity. |
| **Hardware calibration** | Serializable model-free performance evidence for one native hardware/backend environment. |
| **Model assessment** | Native assessment of one exact target and serving profile, producing compatibility, capacity, memory, and performance evidence. Expensive on an exact cache miss. |
| **Assessing** | Ephemeral, deadline-bounded state owned by an active model-assessment operation. |
| **Eligible assessed configuration** | A configuration whose complete assessment result is `Fits` and contains the evidence required by recommendation policy. |
| **Resolved execution plan** | Load-time native allocation evidence, including acceleration and sequence capacity. It is not provider intent or serving-configuration identity. |

Assessment predicts whether a configuration can normally run. Load admission decides whether it
may run now. A cached assessment never authorizes a load.

## Catalog and recommendation terms

| Term | Meaning |
|---|---|
| **Recommendable model** | One curated offering target and serving profile plus presentation, capability, and recommendation evidence. |
| **Recommendable model catalog** | The release-bound set of targets Magnitude is willing to assess and recommend. Membership implies no assessment result, installation, offering, selection, or residency. |
| **Recommendation candidate** | An algorithm-local eligible assessed configuration enriched with ranking inputs. It has no independent lifecycle or persisted identity. |
| **Catalog candidate** | ACN's presentation projection of one eligible assessed configuration, joined with acquisition and availability state. Its stable identity is the configuration identity. |
| **Recommendation** | A policy-selected configuration labeled with an intent and explanation. |
| **Recommendation portfolio** | The small set of recommendations selected for the current catalog, hardware, native build, backends, and policy identity. |

Candidate records are complete facts, not placeholders. Missing, pending, failed, or incompatible
assessment is not a candidate state. Recommendation membership and intent belong to the portfolio,
not to the candidate.

`Candidate` must not be used unqualified outside recommendation or projection code. In native load
planning, alternatives such as sequence counts one through four are **sequence-capacity options**,
not catalog or recommendation candidates.

A catalog candidate introduces no identity. Acquisition actions use its `ModelOfferingTargetId`;
configuration selection uses its `ModelServingConfigurationId`. ACN resolves both identities
against authoritative catalog state and creates a provider offering only when the configuration is
selected.

## Offering and runtime terms

| Term | Meaning |
|---|---|
| **Provider offering** | One stable provider-facing choice containing provider identity, provider-model identity, and one exact serving configuration. |
| **Slot selection** | The user's durable choice of provider offering and reasoning effort for one product role. |
| **Model slot** | ACN's aggregate for durable product intent, availability, actions, and optional instance projection. |
| **Model instance** | One physical admitted occurrence of a serving configuration in ICN. |

A provider offering may exist while its packages are absent or unavailable. A slot selection does
not imply residency. An instance does not own durable user intent.

## Identity map

| Identity | Identifies | Owner |
|---|---|---|
| `ModelPackageId` | One immutable package | ICN |
| `DownloadAttemptId` | One package-install attempt | ICN |
| `ModelOfferingTargetId` | One package or speculative pair | ICN |
| `ModelServingConfigurationId` | One target/profile combination | ICN |
| `ModelInstanceId` | One physical loaded occurrence | ICN |
| `(ProviderId, ProviderModelId)` | One provider offering | ACN/provider boundary |
| `SlotId` | One product role assignment | ACN |

There is no generic `ModelId`. Display names, paths, repository names, filenames, recommendation
membership, cache keys, and array position are never operational identity.

For the local provider, ACN may derive `ProviderModelId` deterministically from the selected
`ModelServingConfigurationId`, but the branded identities are not interchangeable: configuration
identity exists before an offering; provider-model identity exists only at the provider boundary.

## Canonical relationship

```text
ModelPackage(s)
  -> ModelOfferingTarget
  + ServingProfile
  -> ModelServingConfiguration
  + HardwareCalibration + NativeEnvironment + CapacityPolicy
  -> ModelAssessment
  -> EligibleAssessedConfiguration
  -> CatalogCandidate
       -> optional Recommendation
       -> DownloadModel(targetId), if packages are missing
       -> CreateLocalModelOffering(configurationId) -> ProviderOffering / providerModelId
       -> AssignSlot(slotId, providerModelId) -> SlotSelection
       -> LoadModel(slotId) -> ModelInstance
```

The action pipeline deliberately changes identity at each ownership boundary: target identity for
package acquisition, configuration identity for offering creation, provider-model identity for
durable selection, and slot identity for loading.
