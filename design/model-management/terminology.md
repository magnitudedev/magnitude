---
applies_to:
  - inference/crates/icn-contracts/src/**
  - inference/crates/icn-models/**
  - packages/icn/src/**
  - packages/sdk/src/inference*
  - packages/acn/src/local-model-**
  - packages/acn/src/local-provider-**
  - packages/acn/src/model-slot-**
  - packages/acn-protocol/src/schemas/model-state.ts
---

# Model-management terminology

## Authoritative resources

| Term | Meaning |
|---|---|
| **Model** | A callable model addressed everywhere by its existing canonical model ID, such as `gemma-4-26b-a4b-it-qat:gguf:q4`. It projects release declaration, capabilities, serving configuration, and current installation state. |
| **Model package** | One immutable content-identified set of files, roles, relationships, inspection, and source evidence. |
| **Installed package** | The current observation that every required valid file for a Package is present. It is a state, not another resource. |
| **Download** | One process-local admitted acquisition occurrence created by Model installation. |
| **Instance** | One physical loaded occurrence of a Model, identified by an ICN-created Instance ID. |
| **Hardware** | ICN's singleton current topology, capacity, and calibration profile. |
| **Slot** | ACN-owned durable provider-qualified model selection, reasoning preference, favorites, and recency for a product role. It contains no Instance identity or residency state. |

Catalog is a source of Model declarations, not a parallel collection. A local product view is a
read-only ACN projection adding Magnitude assessment, provider, ranking, update, and warning
semantics to native facts. It is not an inference authority.

## Derived values

| Term | Meaning |
|---|---|
| **Servable model bundle** | Structural set of one standalone Package or a method-identified speculative target and draft relationship. It has no public resource lifecycle. |
| **Serving profile** | Serving intent for a bundle, currently context length. |
| **Model serving configuration** | The exact bundle and profile ICN currently resolves for a Model. Callers do not construct or persist it. |
| **Model assessment** | Recomputable compatibility, capacity, memory, and performance evidence for a Model configuration and Hardware. |
| **Load plan** | Advisory allocation evidence computed from a Model, current Hardware, and resident Instances. |
| **Provider offering** | ACN's provider-facing metadata for a callable Model, keyed by the same canonical model ID. |
| **Intelligence** | Model-level broad capability on the catalog's declared, versioned Artificial Analysis Intelligence Index scale, with direct or explicit-estimate provenance. |
| **Fidelity** | Artifact-variant preservation of its represented model, derived from the catalog fidelity rank and independent of Intelligence. |

Assessment predicts whether a Model normally fits. ICN admission decides whether it may run now.
Cached assessment never authorizes loading.

## Identity and durability

| Identity | Identifies | Lifetime and owner |
|---|---|---|
| Canonical model ID | One callable Model across native APIs, `/v1/models`, inference requests, providers, Slots, and harness configuration | Stable release meaning, ICN |
| `ModelPackageId` | One immutable Package | Content-stable, ICN |
| `ModelDownloadId` | One acquisition occurrence | ICN process lifetime |
| `ModelInstanceId` | One loaded occurrence | ICN controller lifetime |
| `(ProviderId, ProviderModelId)` | One provider offering; for `local`, `ProviderModelId` is the canonical model ID | Provider boundary; Slot selection may retain it |
| `SlotId` | One product role assignment | Durable, ACN |

Magnitude persists Slot selection, recency, favorites, and onboarding completion. Model files are
authoritative artifacts; catalog declarations are release data. Package inventory, configurations,
offerings, assessments, Downloads, and Instances are derived or process-local and are not copied
into durable model state.

`ManagedModelStore` is the exclusive mutation boundary for Magnitude-owned artifacts. Model install
and uninstall address a canonical model ID; ICN resolves exact Packages, preserves shared or active
Packages, and never asks clients to reproduce bundle accounting.

## Runtime relationship

```text
release declaration + current Packages
              -> Model -> serving configuration -> assessment / load plan
                    |
                    +-> install -> Download
                    +-> explicit warm load --------+
                    +-> inference request ----------+-> residency coordinator -> Instance -> lease

ACN Slot selection by canonical model ID
  + native Model and Instance Queries
  -> first-party residency presentation
```

Equivalent explicit and inference demand joins one admitted load. ICN creates Instance identity,
serializes incompatible replacement, and protects active inference with exact request leases.
Display names, paths, repositories, filenames, cache keys, and array positions are never operational
identities.
