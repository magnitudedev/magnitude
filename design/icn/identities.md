---
applies_to:
  - inference/crates/icn-contracts/src/**
  - packages/protocol/src/schemas/model-state.ts
  - packages/acn/src/local-provider-**
  - packages/acn/src/model-slot-**
---

# Identities

| Identity | Meaning |
|---|---|
| `ModelPackageId` | One immutable artifact package. |
| `DownloadAttemptId` | One admitted attempt to install an exact package. |
| `ModelOfferingTargetId` | One installable package set: a package or target/draft pair. |
| `ModelServingConfigurationId` | One target plus serving profile, currently chiefly context length. Multiple configurations may share a target. |
| `ModelInstanceId` | One physical loaded occurrence of a configuration. |
| `ProviderId` | One provider namespace, such as `local`. |
| `ProviderModelId` | One offering inside a provider namespace. Slots address `(ProviderId, ProviderModelId)`. |
| `SlotId` | One ACN product role assignment. |
| `CatalogCandidateId` | Presentation-row identity only; never a domain-operation identity. |

There is no generic `ModelId`.

For the local provider, `ProviderModelId` is the `ModelServingConfigurationId` value within the
`ProviderId = "local"` namespace.

## Layer boundaries

| Layer | IDs it uses |
|---|---|
| **ICN** | `ModelPackageId`, `DownloadAttemptId`, `ModelOfferingTargetId`, `ModelServingConfigurationId`, `ModelInstanceId`. It never knows provider, slot, catalog-candidate, or onboarding identity. |
| **ACN** | ICN identities when observing or commanding ICN; `(ProviderId, ProviderModelId)` for offerings; `SlotId` for assignments. `CatalogCandidateId` may appear only in presentation projections. |
| **Client** | `(ProviderId, ProviderModelId)` and `SlotId` for selection; `ModelOfferingTargetId` for artifact actions; `ModelInstanceId` for exact Stop; `CatalogCandidateId` only as a row key. It does not use package or configuration IDs. |
