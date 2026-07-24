---
applies_to:
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/icn/src/downloads/**
  - packages/acn/src/local-model-**
  - packages/acn/src/local-models.ts
  - packages/acn/src/local-provider-**
  - packages/protocol/src/schemas/model-state.ts
  - cli/src/features/local-inference/**
  - packages/storage/src/types/config.ts
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-models/**
  - inference/crates/icn-server/src/main.rs
---

# Local model domain

This document defines Magnitude's local-model concepts and their relationships.

## Concepts

### Model file

One immutable model-related file, identified by its contents and assigned a role such as weights,
shard, projector, draft, or MTP.

### Model package

One exact bundle of model files and their relationships. Quantization, shard composition, and
auxiliary components are part of the package.

A package is immutable and has one source:

```text
ModelPackage
  package identity
  source
  files and their relationships
```

### Model package source

One exact location from which a package can be obtained.

A Hugging Face source identifies a repository, immutable commit, and exact package files. A
repository alone is not a package source because it may contain multiple packages.

A local filesystem source identifies a root and exact package files. It does not imply Hugging
Face provenance.

### Model package entry

ACN's view of a package and its relationship to this machine:

```text
ModelPackageEntry
  package
  local state
    not installed
    | downloading(attempt, progress)
    | installed(location)
  inspection: pending | inspected | invalid | incompatible
  optional last download failure
```

The package remains unchanged as local state changes. ACN owns the entry and derives installed
state from ICN inventory.

### Download

A managed attempt to install a package:

```text
not installed -> downloading -> installed
                       |
                       └─> not installed + last download failure
```

A terminal failure is an attempt result, not a permanent package state. Retry starts a new attempt
and clears the previous failure. Dismiss clears the failure. Automatic retries are bounded; after
exhaustion the failure is published.

### Speculative decoding pair

An ordered combination of a target model package and a draft model package.

MTP weights form the draft package when they are independently packaged. An MTP or projector file
intrinsic to one package remains a file in that package.

### Model offering target

The complete model target exposed for inference:

```text
ModelOfferingTarget = ModelPackage | SpeculativeDecodingPair
```

The target is explicit. A draft package is never silently attached to a single-package target.
Likewise, installing the components of a catalogued pair does not create duplicate standalone
product models unless those packages are also explicitly offered as standalone targets.

### Serving profile

Model-agnostic serving intent:

```text
ServingProfile
  context length
  parallel sequence count
```

Context length is the total shared KV-pool capacity. Parallel sequence count is the maximum
concurrent occupancy of that pool, not a multiplier or a promise of the full context per sequence.

Whether a profile works well is determined only when it is assessed with a target, runtime, and
hardware.

### Model serving configuration

The complete provider-neutral configuration that can be assessed and run:

```text
ModelServingConfiguration
  configuration identity
  model offering target
  serving profile
```

This is the combination selected by a recommendation, assessed for hardware fit, and exposed
through a provider. ICN owns its stable identity. ACN stores and passes that identity unchanged.

### Offering assessment

The compatibility, fit, and performance result for one exact model serving configuration, runtime,
and hardware environment.

An assessment of a single package is not an assessment of a speculative decoding pair.

### Recommendable model

One catalogued target that Magnitude is willing to assess and recommend.

```text
RecommendableModel
  identity
  target
  eligible serving profiles
  presentation and curation metadata
  capability evidence
```

Package, source, quantization, and speculative-pair facts belong to the target. The recommendable
model adds only recommendation-specific metadata.

Each exact package or speculative pair is a separate recommendable model. Family or checkpoint
grouping is presentation metadata, not operational identity.

### Recommendable model catalog

The complete set of recommendable models. This is the domain concept currently represented by
recipes.

Catalog membership means only that a target may be recommended. It does not mean the target fits,
is recommended, is installed, is offered, or is loaded.

ICN exposes this catalog at `GET /v1/models/catalog`. It is distinct from ACN's provider model
catalog, which contains configured provider offerings available for slot selection.

Repository metadata used to resolve the curated catalog is a disposable snapshot. A fresh snapshot
is reused without a network request. An expired snapshot remains sufficient to serve the last
complete catalog immediately while ICN conditionally revalidates it in the background; a failed
refresh never replaces the last complete result. A machine with no repository snapshot must fetch
the metadata once before it can resolve exact immutable package files. The shared HTTP client is
reused across repository operations. Curated package resolution consumes that one repository
snapshot directly for every format, so preparing each quantization does not request the same
repository metadata again. Missing immutable GGUF header ranges are acquired with bounded
concurrency and remain keyed by published content identity.

### Recommendation

A policy suggestion selecting one model serving configuration for a product intent.

Recommendation changes never change package, configuration, provider-offering, or slot-selection
identity.

ACN persists the last complete recommendation portfolio as disposable derived state. It may reuse
that portfolio only when the complete catalog-target/profile input, native hardware topology and
build, enabled backends, and recommendation-policy identity are unchanged. Missing, malformed,
unreadable, mismatched, or older-than-seven-days portfolio data is a cache miss. It never
suppresses recomputation after one of those inputs changes.

Recommendation calculation publishes an ordered, cumulative lifecycle for hardware, downloaded
model discovery, catalog, metadata preparation, assessment, and selection. Each step is pending,
running, completed, or failed; running and terminal states carry authoritative timing, and bounded
collection work carries completed and total counts. Completed work remains visible while later work
runs. Presentation may animate a running step from the published start time, but it must not invent
server progress.

### Provider offering

One stable provider-facing choice:

```text
ProviderOffering
  provider identity
  provider model identity
  model serving configuration
```

A local offering may exist while its packages are downloading or absent. Its provider-catalog
projection is disabled until every required package is installed and its exact configuration fits.
The offering itself remains durable and unchanged as those observations change.
Target capabilities are resolved from catalog or installed-package inspection evidence and are not
duplicated in the durable offering record.

### Slot selection

The user's durable choice:

```text
SlotSelection
  provider identity
  provider model identity
  reasoning effort
```

It references a provider offering. It does not copy package, source, recommendation, assessment, or
runtime identity.

ACN normalizes its reasoning effort against the referenced provider model at the slot boundary,
using the model default whenever the requested or stored value is unsupported.

A client that assigns a slot must keep the initiating flow alive until the authoritative
`ModelSlots` mirror confirms that exact normalized selection. Starting an assignment mutation is
not confirmation and must not advance onboarding or another configuration flow by itself.

## Relationships

```text
ModelPackage
  ├─ has one ModelPackageSource
  └─ contains ModelFiles

Download
  └─ changes ModelPackageEntry local state

SpeculativeDecodingPair
  ├─ target ModelPackage
  └─ draft ModelPackage

ModelOfferingTarget
  └─ ModelPackage | SpeculativeDecodingPair

ModelServingConfiguration
  └─ ModelOfferingTarget + ServingProfile

RecommendableModelCatalog
  └─ contains RecommendableModels
       └─ each has ModelOfferingTarget

OfferingAssessment
  └─ ModelServingConfiguration + Runtime + Hardware

Recommendation
  └─ RecommendableModel + ModelServingConfiguration + OfferingAssessment

ProviderOffering
  └─ exposes ModelServingConfiguration through a provider

SlotSelection
  └─ ProviderOffering + ReasoningEffort
```

## Identity rule

Packages join across entries, assessments, recommendations, and offerings by package identity.
Repository names, filenames, paths, display names, recommendation membership, and cache keys are
never package identity.

Serving configurations join across assessments, recommendations, offerings, provider resolution,
and runtime residency by ICN-issued configuration identity. ACN never derives a configuration
identity from an assessment ID or reimplements ICN's identity algorithm.

## Type and persistence ownership

Rust ICN contracts are the native API authority and derive their OpenAPI schemas directly.
Generated TypeScript ICN schemas remain private transport contracts.

Protocol owns the authored product schemas, including branded identifiers and product invariants.
Storage composes those schemas directly. In particular, it persists the complete local provider
offering and does not redefine packages, targets, profiles, or configurations.

ACN has one explicit adapter between generated ICN values and protocol values. Structurally equal
values cross that boundary through schema validation rather than another authored representation.
