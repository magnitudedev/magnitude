---
applies_to:
  - inference/crates/icn-contracts/src/**
  - packages/acn-protocol/src/schemas/model-state.ts
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
| `MemoryDomainId` | One physical memory pool in a captured hardware topology. |
| `HardwareDeviceId` | One stable native device view and its device-local constraints. |

There is no generic `ModelId`.

For the local provider, `ProviderModelId` is the `ModelServingConfigurationId` value within the
`ProviderId = "local"` namespace.

## Hardware topology identity

A native device view has one canonical identity formed from its normalized backend, optional
physical-device identity, and process-global native registry index. Backend spelling aliases are
normalized by that identity abstraction, not independently by discovery, fitting, or runtime
accounting.

Native allocation reports carry a locator rather than claiming a stable device identity. A locator
contains the identity evidence available from that native API: either the registry index alone or
the index plus backend and optional physical identity. Only the captured hardware topology may
match a locator to a canonical device identity and its physical memory domain. Contradictory or
unknown evidence fails closed.

## Layer boundaries

| Layer | IDs it uses |
|---|---|
| **ICN** | `ModelPackageId`, `DownloadAttemptId`, `ModelOfferingTargetId`, `ModelServingConfigurationId`, `ModelInstanceId`. It never knows provider, slot, catalog-candidate, or onboarding identity. |
| **ACN** | ICN identities when observing or commanding ICN; `(ProviderId, ProviderModelId)` for offerings; `SlotId` for assignments. `CatalogCandidateId` may appear only in presentation projections. |
| **Client** | `(ProviderId, ProviderModelId)` and `SlotId` for selection; `ModelOfferingTargetId` for artifact actions; `ModelInstanceId` for exact Stop; `CatalogCandidateId` only as a row key. It does not use package or configuration IDs. |
