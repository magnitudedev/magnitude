---
applies_to:
  - inference/catalog/**
  - inference/crates/icn-catalog/**
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/models.rs
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/icn/src/downloads/**
  - packages/acn/src/local-model-packages.ts
  - packages/acn/src/local-models.ts
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/composer/**
  - cli/src/features/local-inference/**
  - cli/src/features/model-menus/**
---

# Model catalog and acquisition

This document defines catalog publication, package resolution, installation, inventory, and
download behavior. Terms follow [Model-management terminology](./terminology.md).

## Ownership

ICN owns the recommendable catalog, exact packages and sources, the managed model store, installed
inventory, model-download occurrences, and their internal package attempts. ACN observes these independent authorities
and produces product projections. Clients initiate mutations through ACN and do not infer command
authorization or completion from cached projections.

Catalog, package presence, download activity, inspection, assessment, provider offering, slot
selection, and runtime residency remain separate facts.

## Release catalog

The recommendable catalog is immutable release data. Human-reviewed declarations identify each
artifact variant by its exact upstream format selector, short variant label, fidelity rank, and
quantization-aware-training fact. Each variant publishes one exact default
`ModelServingConfiguration`, required companion components, base presentation, and recommendation
evidence. A declaration may identify a separately packaged speculative draft from another upstream
repository. Every distinct package source is locked independently, and the same immutable draft
package may be referenced by multiple target-variant bundles without duplicating its installation.
The configuration and its ICN-issued identity are published independently of hardware assessment.
Advancing the source lock is an explicit development operation.

Catalog generation resolves only locked revisions through production package construction,
inspection, template analysis, and native planning. It emits a self-contained planner-input bundle
for every target and separately packaged draft and proves that compact planner inputs yield the
same native assessments as their source metadata at each declaration's catalog profile. Repeated
references to one immutable package reuse one package identity and one content payload. A declared
profile above the exact bundle maximum, a missing or differently typed companion or draft, an
unresolved entry, incomplete coverage, integrity mismatch, or assessment mismatch fails generation
or ICN readiness.

Runtime catalog use performs no upstream discovery and does not follow mutable revisions. Adding a
model, format, or upstream revision requires a new reviewed release catalog.

## Package resolution

The shipped package manifest is authoritative for file paths, sizes, digests, roles, and
relationships. A repository revision is only a retrieval address. Downloaded content is published
only after it matches the shipped package exactly.

An explicitly declared fallback may change the retrieval commit when the primary revision is no
longer available. It must not change package identity, bundle composition, capabilities, planner
inputs, or recommendation evidence.

Fallback is attempted only when the pinned revision or required file is definitively absent. An
authentication, authorization, rate-limit, timeout, or server failure preserves the original
failure. A fallback commit is accepted only when every required relative path, byte size, and
SHA-256 digest matches the shipped manifest. Otherwise acquisition fails without changing the
catalog entry or any installed copy.

## Stores and inventory

The configured managed store is authoritative for Magnitude-owned installations. Explicit
Hugging Face cache roots are read-only installation sources; they remain externally owned and are
never silently moved, deleted, or adopted into the managed store.

The standard Hugging Face cache root is resolved once by ACN from `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, then the user-home default, in that order, and
is passed to ICN explicitly. Magnitude uses only credentials explicitly supplied to it; ambient
host-login credentials are not inherited.

Installed inventory reports packages currently present in the managed store or configured Hugging
Face caches, their generated installation origin (`Magnitude` or `HuggingFaceCache`), and their
inspection results. Inventory discovery is hardware-independent: it does not choose profiles,
calibrate hardware, assess configurations, or recommend models. Failure to inspect one package is
isolated to that package.

Initial Hugging Face cache reconciliation is asynchronous and does not delay ACN readiness. The
observer publishes an initial snapshot, then atomically replaces it with a completed scan; later
scans retain the previous complete snapshot while work is in flight. Reconciliation owns filesystem
discovery, content hashing, GGUF inspection, tensor-storage derivation, and package construction.
Inventory queries only return the current materialized snapshot and perform no filesystem work.

Package inspection is one of `Pending`, `Inspected`, `Invalid`, or `Incompatible`. Only an inspected,
compatible package may participate in loading. Draft, MTP, and projector components retain their
roles and relationships. Any package inventory identifies as independently servable may be
represented and assessed as a standalone bundle without catalog membership.

Discovery prefers authoritative GGUF role evidence. Exact filename tokens may propose a
relationship only when authoritative evidence is unavailable; file size never determines role.
`DraftFor(target, method)` and `MtpFor(target)` remain distinct artifact relationships. A servable
speculative-decoding bundle carries its method and embedded-versus-separate draft source directly;
callers never infer that execution contract by searching package relationships.

## Downloads

A download command carries an exact servable bundle. If every required package is installed, ICN
returns `AlreadyInstalled` without manufacturing a download occurrence. Otherwise ICN creates one
durable `ModelDownload` with a stable `ModelDownloadId`, atomically admits missing package work as
new attempts or joins already admitted equivalent work, and returns that occurrence. The caller
waits for the exact model-download identity; it does not wait for a later inventory-wide refresh.

The model download aggregates bounded progress and exactly one terminal outcome across the bundle.
Package attempts are ICN execution details: each installs one package, and equivalent concurrent
model downloads may share one active package attempt. Caller interruption after admission detaches
that waiter but does not abandon shared work. Cancelling a model download is an explicit command;
shared package work is cancelled only when no other live model download depends on it. A retry
creates a new model-download identity and any required new package attempts. Historical success
never proves current presence, and historical failure never becomes permanent package state.

Download failures whose facts support recovery or presentation are structured domain outcomes.
Insufficient disk space carries the required and currently available byte counts from ICN through
ACN's package and product projections. Interrupted-operation manifests and reconstructed inventory
retain the same typed failure rather than reducing it to a string. No higher layer parses
diagnostic prose or reconstructs the storage check. Expected failures distinguish insufficient
disk space, interruption, unavailable source content, unavailable network, local storage failure,
and corrupt downloaded content. Cancellation is a separate terminal outcome, not a failure.
Everything else is an internal failure representing a violated invariant or implementation defect.
There is no catch-all error code: clients render the variant and never infer behavior from prose.

ICN also owns durable acknowledgement of an exact failed model download. Acknowledgement is an
idempotent mutation accepted only for that terminal occurrence and committed before success is
returned. It does not erase or alter the failure outcome. A retry has a new model-download identity
and begins unacknowledged. ACN retains no parallel acknowledgement or dismissal state; it projects
an acknowledged failure as no longer requiring presentation.

Completed content is verified before atomic publication. Partial and resumable content is not
reported as installed. Deleting or externally removing an installed package changes inventory
independently of attempt history.

Managed installation manifests retain exact package identity even when distinct packages share a
primary weights path. Path identity may suppress duplicate discovery of the same external artifact;
it never coalesces two managed manifests with different package structure. Publication makes the
managed manifest and installed snapshot observable before the corresponding attempt becomes
terminally completed.

A managed transfer incrementally hashes successfully written bytes and checkpoints the
serializable digest state with its exact committed offset. Resume truncates any uncommitted tail and
continues without rereading the downloaded prefix. Missing or invalid checkpoint evidence discards
the untrusted partial artifact. Publication compares the final accumulated digest before the
atomic move.

## Product-projection contribution

The complete join and visibility contract belongs to
[Local-model product projection](./local-model-product-projection.md). Catalog and acquisition
contribute exact catalog configurations, installed-package observations, and model-download state;
they do not decide whether another domain's entity is visible.

Exact bundle structure also joins presentation metadata. A bundle present in the recommendable
catalog retains its curated display name and description regardless of installation or
recommendation status. Repository and filename-derived presentation is used only for bundles with
no curated metadata under that exact structure; similarity of repository, filename, family, or
quantization never transfers presentation between bundles. Every independently servable installed
package contributes a standalone product row even when no retained or catalog configuration
references it.

Product installation addresses the row's `ModelServingConfigurationId`. For an existing retained
configuration ACN resolves that exact value directly; for first installation it resolves the exact
catalog-published configuration. ACN durably retains it before the private package owner admits its
bundle's required packages. The client-facing admission result carries the configuration-backed
provider-model ID and reports that the bundle is already installed or carries the exact stable
`ModelDownloadId` needed for subsequent cancellation and observation. ICN
package-download commands remain bundle/package operations private to installation.

Download state stores no assessment evidence and determines neither configuration, offering, nor
slot identity. ACN may immediately merge an exact admission response into observation to reduce
presentation latency, while ICN model-download and inventory state remain authoritative for completion and
physical presence.

Every CLI download notification is a pure projection of
`LocalModelsState.models[].acquisitionState`. Closing the Models menu leaves active download status
visible in the persistent composer footer, and the client does not retain a second download counter
or lifecycle. The composer footer and Models menu consume the same resolved notification and remove
download activity when no row is actively downloading.

## Concurrency and recovery

Operations that could publish or remove the same package serialize. Reads may share in-flight
resolution and inspection. Removal must reject or wait while a runtime owns package use; it never
invalidates a resident instance silently.

Installation, explicit configuration deletion, and configuration-recovery completion also share one
ACN-scoped configuration-operation coordinator. Its critical section spans package mutations and the
corresponding durable configuration commit, preventing a deletion snapshot from racing a newly
retained configuration that shares those packages.

Derived inspection, source-resolution, assessment, and timing caches are disposable. Corruption or
deletion causes the smallest possible cache miss and never removes installed models, reconstructs
the release catalog, changes identity, or creates a permanent failed state.

Configuration recovery is a bounded durable-state repair epoch. A missing or invalid model-state
document installs the defined default with recovery incomplete. Once catalog and installed
inventory have complete initial snapshots, ACN matches exact verified installed
packages against catalog configurations, adds configurations only for bundles without retained
configurations, and marks recovery complete in the same durable commit. It performs
no filesystem scan, package inspection, model assessment, or recommendation work. Later inventory
changes do not create retained configurations. An installed independently servable package remains
visible regardless of recovery and
receives only the ICN-issued configuration for ACN's standard profile decision after successful
inspection. ACN makes that profile decision only when neither a retained nor catalog configuration
exists for the bundle; ICN constructs and canonically identifies the configuration.

## Conformance

- Catalog membership implies only eligibility for assessment and recommendation.
- Format, variant label, fidelity rank, and quantization-aware-training status are explicit reviewed
  catalog facts; no downstream layer reconstructs one from another.
- Every catalog bundle publishes the exact reviewed serving profile and required companion set used
  to construct its package.
- Every catalog and standard serving configuration carries canonical identity constructed by ICN;
  ACN never recreates that identity.
- Runtime setup requires no network access to reconstruct the release catalog.
- Installed inventory reports presence and inspection, never inferred model assessment.
- Insufficient-disk failures preserve required and available byte counts through every serialized
  and projected download state; diagnostic text is never their data contract.
- Download admission, observation, cancellation, and failure acknowledgement correlate one exact
  `ModelDownloadId`; package-attempt membership is not a client-facing identity.
- On first observing a completed model download, ACN refreshes installed inventory before
  projection. A currently absent package is therefore `NotInstalled`; historical completion never
  proves current presence.
- Shared artifact paths never coalesce distinct managed package identities.
- Failed model-download acknowledgement survives restart and cannot hide a later retry's failure.
- Package identity is independent of paths, display names, and mutable upstream references.
- Exact catalog bundle structure preserves curated presentation across installation and
  recommendation changes; fallback presentation cannot overwrite it.
- Product installation and retained selection begin with serving-configuration identity; only the
  private package owner uses bundle/package structure, and provider projection represents the same
  configuration in the local provider namespace.
- Catalog failure cannot hide retained configurations or installed packages.
- One package failure cannot corrupt unrelated inventory or download state.
