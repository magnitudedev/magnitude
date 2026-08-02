---
applies_to:
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/icn/src/downloads/**
  - packages/acn/src/local-model-**
  - packages/acn/src/local-models.ts
  - packages/acn/src/local-provider-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/local-inference/**
  - cli/src/features/model-setup/**
  - cli/src/features/model-menus/**
  - packages/storage/src/types/config.ts
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-models/**
  - inference/crates/icn-server/src/main.rs
  - inference/catalog/**
---

# Local model domain

This document defines Magnitude's local-model concepts and their relationships.

## Concepts

### Model file

One immutable model-related file, identified by its contents and assigned a role such as weights,
shard, projector, draft, or MTP. Draft and MTP are distinct roles. A model-backed draft file also
records its speculative method; package construction never collapses an ordinary, Eagle3, or
DFlash draft into MTP.

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

MTP weights may form the draft package when they are independently packaged. An MTP or projector
file intrinsic to one package remains a file in that package. Distribution layout and speculative
method are independent facts: embedded-target MTP, companion MTP, and companion model-backed drafts
remain distinguishable even when their files are distributed in one package.

Typed package relationships preserve that distinction:

```text
ProjectorFor(projector, target)
MtpFor(mtp, target)
DraftFor(draft, target, method)
```

The draft method is declared by the package or retained as explicitly unresolved evidence until
native inspection establishes it. Filenames may propose a relationship during discovery but are
not authoritative method evidence.

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
```

Context length is the maximum context for one request. Native physical context and sequence
capacity are resolved when the configuration is loaded and are not provider intent.

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
Configuration identity depends on target and per-request context, not the load-time native
sequence capacity.

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

Package, source, quantization, and speculative-pair facts belong to the target. ICN resolves GGUF
file-type families through the pinned llama.cpp binding rather than duplicating their numeric
values. Package metadata retains llama.cpp's authoritative family name and a coarse user-facing
bit-width name derived by an exhaustive match over the binding's complete file-type enum. The recommendable
model adds only recommendation-specific metadata.

Each exact package or speculative pair is a separate recommendable model. A speculative target's
method and component relationships are part of that exact target and therefore its identity.
Family or checkpoint grouping is presentation metadata, not operational identity.

### Recommendable model catalog

The complete set of recommendable models. This is the domain concept currently represented by
recipes.

Catalog membership means only that a target may be recommended. It does not mean the target fits,
is recommended, is installed, is offered, or is loaded.

ICN exposes this catalog at `GET /v1/models/catalog`. It is distinct from ACN's provider model
catalog, which contains configured provider offerings available for slot selection.

The catalog shipped to users is release-bound data, not a runtime cache. Human-reviewed source
declarations name the curated checkpoints, formats, profiles, and recommendation evidence. An
explicit developer command advances a minimal ID-to-commit lock. Generation resolves only those
pinned commits through ICN's production Hugging Face resolver, GGUF parser, package builder,
template assessor, and native planner, then writes one self-describing planner-input bundle. The
compact form is standard GGUF: it preserves model metadata and tensor descriptors,
omits lexical tokenizer payloads, and supplies a compressible placeholder vocabulary of the
original cardinality so ordinary native vocabulary loading preserves graph-relevant state.
Generation proves that source and compact sparse models produce identical native hardware
assessments for every curated profile. Release builds package the derived bundle, and local
development constructs the same installation layout. The bundle is not committed to source control.

ICN validates the installed bundle's structure, entry integrity, and exact catalog coverage.
Missing, partial, or malformed release data is an installation defect
and prevents ICN readiness; it is never treated as a user cache miss. User setup performs no remote
catalog reconstruction or model-header fetch. It combines the release inputs with current hardware
topology and local calibration to calculate fit and speed. Installed targets continue to use their
local files. A source-backed target outside the release catalog cannot be assessed by the product
runtime. Adding a model, changing a format, or updating an upstream revision requires regenerating,
reviewing, and shipping a new release.

### Recommendation

A policy suggestion selecting one model serving configuration for a product intent.

Recommendation changes never change package, configuration, provider-offering, or slot-selection
identity.

ACN retains a recommendation portfolio for the current process while its catalog,
hardware, native build, enabled backends, and recommendation-policy identity are unchanged. It does
not persist portfolios across daemon restarts. ICN's finer-grained assessment cache supplies the
expensive reusable work without allowing a stale or operationally incomplete empty portfolio to be
published as current truth. Failed calculations are not cached as recommendation results.

Recommendation calculation publishes an ordered, cumulative four-stage lifecycle for hardware,
downloaded-model discovery, model evaluation for the current machine, and recommendation
preparation. Each step is pending, running, completed, or failed. Bounded evaluation carries
authoritative completed and total counts plus an optional remaining-time estimate derived from
measured work in the current run. Completed work remains visible while later work runs.
Presentation may animate the running marker, but it must not invent progress, phases, or timing.

Local-model onboarding presents that lifecycle as one coherent discovery flow. While the initial
recommendation lifecycle or provider-model catalog is loading, onboarding shows authoritative
progress but does not expose partial target projections, provisional model names, preparation
failures, or blocked slot state. Model choices appear only after recommendations are ready and the
initial provider catalog has settled. A terminal discovery failure appears once in progress, while
failures from a user-initiated download or selection remain immediate and attached to that action.

Choosing an installed or curated model starts one client-owned sequence of ordinary domain
mutations. The target-level download command admits or joins required package work and returns only
after every exact admitted attempt reports successful publication. Generic slot assignment retains
the selected serving configuration and stores the primary selection without using installed
presentation as command authorization. Generic slot load lets ICN validate current package presence
and returns only after the exact selected instance is Ready; the client then marks onboarding
complete. The submitted command and choice are the only onboarding-specific transient state and
exist only in the client that received the explicit action. They bridge command admission but do
not represent download or instance lifecycle; presentation derives those facts from authoritative
mirrors. Confirmed onboarding cancellation clears the submission, an externally stopped load is
terminal presentation, and successful completion closes setup. ACN has no onboarding model command,
activation service, or startup reconciler. Confirmed cancellation uses the ordinary target-download
cancellation or slot-clear mutation. Interruption or restart never reconstructs command intent from
onboarding, download, slot, or instance snapshots.

The compatible recommendable-candidate projection is published beside the small labeled portfolio.
Candidate records contain assessment facts; recommendation membership and intent exist only in the
portfolio. Onboarding presents installed models and selected portfolio entries, not every
compatible candidate. Complete execution assessments always contain measured performance.
Operational failure remains a failed refresh and is never published as an empty ready result.

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

Retryable failures while automatically configuring or assessing an installed offering remain a
preparing state. They are not presented as model incompatibility. Only a terminal reconciliation
failure or an authoritative assessment result may make the target unavailable to the user.

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

A local offering whose exact packages are absent, downloading, incompatible, or not load-admissible
cannot be assigned. Existing assignments may later become unavailable when their dependencies
degrade, but assignment never creates a blocked slot.

A successful slot-assignment mutation means ACN has validated the exact installed configuration,
durably stored the normalized selection, and atomically published the `ModelSlots` mirror and agent
model configuration. An immediately following slot-load command is therefore admissible unless
authoritative conditions changed after assignment. A rejected assignment leaves the slot unchanged.

### Model favorites

A favorite is a durable user preference identified by the provider identity and provider model
identity together. Favoriting never selects, loads, installs, or otherwise changes a model.

The model-selection menu marks favorites immediately after the authoritative preference changes and
places models that were favorites when the menu was entered before non-favorites. Preference,
selection, and recency changes never reorder an open model list; the next entry into that menu
captures a new ordering snapshot.

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
