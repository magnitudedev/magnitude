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

## Callable identity

`ModelId` identifies one stable callable local-model variant everywhere: ICN catalog and discovery
operations, assessment, runtime admission, ACN products, local-provider offerings, Slots, harnesses,
and inference. The local provider preserves its serialized value unchanged as `ProviderModelId`.

Catalog IDs compose two authored identities:

```text
CatalogBaseId    = qwen3.5-4b
CatalogVariantId = gguf:q4
ModelId          = qwen3.5-4b:gguf:q4
```

External Hugging Face discoveries use:

```text
ModelId = hf:<owner>/<repository>/<repository-relative-GGUF-selector>
```

There is no `ModelVariantId`, `ModelTargetId`, catalog identity object, package-derived provider
alias, or bundle key at the ICN–ACN boundary.

## Different entities

| Term | Meaning and owner |
|---|---|
| Catalog model | Reviewed callable variant that exists whether installed or not; ICN catalog domain |
| Discovered model | Non-catalog callable candidate observed in an external source; ICN discovery domain |
| Package / bundle | Exact files and private servable structure; ICN implementation only |
| Inventory entry | One source-location/content observation; ICN implementation only |
| Catalog installation operation | One model-addressed install/update synchronization occurrence; ICN |
| Assessment | Recomputable compatibility, memory, and performance evidence for a model snapshot; ICN computes, ACN coordinates |
| Instance | One physical loaded occurrence identified by `ModelInstanceId`; ICN |
| Local model product | ACN application projection combining catalog or discovery facts with assessment, acquisition, ranking, and residency |
| Provider offering | Selectable ACN projection keyed by the same canonical `ModelId` |
| Slot | Durable ACN provider-qualified selection; never an instance or material identity |

Packages, content IDs, source revisions, operation IDs, assessment request IDs, and instance IDs
identify genuinely different material or occurrences. None substitutes for `ModelId` in a user
selection.

## Authority

ICN owns catalog, discovery, packages, inventory, material resolution, acquisition, assessment, and
runtime instances. It exposes catalog and discovery separately. ACN alone creates the combined
application product and provider projections. Clients consume those projections through the SDK.

Assessment predicts fit; current runtime admission decides whether a model can load now. Cached
assessment never authorizes loading.
