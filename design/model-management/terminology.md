---
applies_to:
  - inference/service/contracts/src/**
  - inference/service/models/**
  - packages/icn/src/**
  - packages/sdk/src/inference*
  - packages/acn/src/local-model*
  - packages/acn/src/local-provider-**
  - packages/acn/src/model-slot-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/storage/src/types/custom-models*
---

# Model-management terminology

## Callable identity

`ModelId` identifies one stable callable local-model variant everywhere: ICN catalog and local-model
operations, assessment, runtime admission, ACN products, local-provider offerings, Slots, harnesses,
and inference. The local provider preserves its serialized value unchanged as `ProviderModelId`.

Catalog IDs compose two authored identities:

```text
CatalogBaseId    = qwen3.5-4b
CatalogVariantId = gguf:q4
ModelId          = qwen3.5-4b:gguf:q4
```

A custom model with a Hugging Face identity uses:

```text
ModelId = hf:<owner>/<repository>/<repository-relative-GGUF-selector>
```

Every other custom model uses:

```text
ModelId = local:<header-digest>
```

The header digest is the lowercase hex SHA-256 of the primary GGUF's header bytes, through the
aligned tensor-data offset. It is neither a package ID nor a full-file hash; external files are never
hashed in full. The same bytes in two places are one model, and a moved file keeps its ID. A model
with a Hugging Face identity (added, in the Hugging Face cache, in LM Studio's publisher/repository
layout, a llama.cpp file name that parses to a repository, or an Ollama `hf.co/…` pull) uses its
`hf:` ID wherever it is found.

Neither form names a drafter, revision, commit, path, or projector. The drafter of any model is
configuration, as it is for catalog IDs. There is no `ModelVariantId`, `ModelTargetId`, catalog
identity object, package-derived provider alias, or bundle key at the ICN–ACN boundary.

## Different entities

| Term | Meaning and owner |
|---|---|
| Catalog model | Reviewed callable variant that exists whether installed or not; ICN catalog domain |
| Custom model | Any non-catalog model Magnitude can run; ICN local-models domain |
| Origin | Where a custom model's files came from and therefore who owns them: `Added` (from Hugging Face into the managed store, pinned to a repository commit), `Imported` (one file added by path, referenced in place), or `Found` (observed in a model location, with its source) |
| Model location | A directory ICN scans for GGUFs: the Hugging Face cache, LM Studio, the llama.cpp cache, Ollama, or a user folder |
| Candidate | Files not yet added, inspected before adding: a Hugging Face repository at a commit plus a selector, or a local path |
| Declaration | ACN-persisted record of an added or imported model, or of a drafter attached to any custom model |
| Unsupported local model | A complete GGUF in a model location that the engine cannot run; listed with its reasons, never callable |
| Package / bundle | Exact files and private servable structure; ICN implementation only |
| Assessment Material | Compact immutable GGUF and bundle evidence sufficient for native assessment, including effective template inputs but no tensor payloads; ICN implementation only |
| Inventory entry | One source-location/content observation; ICN implementation only |
| Package validation | Structural statement that exact package files are valid and supported; never capability or hardware evidence |
| Catalog installation operation | One model-addressed install/update synchronization occurrence, ending with the model's optimization for this computer; ICN |
| Download | One model-addressed acquisition of a non-catalog servable bundle (an added model, or a drafter alone); ICN |
| Assessment | One recomputable result containing template-derived capabilities, template fingerprint, compatibility, memory, performance evidence, drafter verdict, and shape fingerprint for exact model work; ICN owns computation and coordination |
| Instance | One physical loaded occurrence identified by `ModelInstanceId`; ICN |
| Local model product | ACN application projection combining catalog or custom-model facts with assessment, acquisition, ranking, and residency |
| Provider offering | Selectable ACN projection keyed by the same canonical `ModelId` |
| Slot | Durable ACN provider-qualified selection; never an instance or material identity |

Packages, content IDs, source revisions, assessment IDs, operation IDs, and instance IDs identify
genuinely different material, evidence, or occurrences. None substitutes for `ModelId` in a user
selection.

## Authority

ICN owns the catalog, local-model scanning and identity, packages, inventory, material resolution,
acquisition, Hugging Face access, assessment, and runtime instances. It exposes catalog and custom
models separately. ACN persists every user choice about custom models, alone creates the combined
application product and provider projections, and supplies ICN its resolved configuration. Clients
consume those projections through the SDK.

Assessment predicts fit; current runtime admission decides whether a model can load now. Cached
assessment never authorizes loading.
