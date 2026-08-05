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
---

# Model catalog and acquisition

This document defines catalog publication, package resolution, installation, inventory, and
download behavior. Terms follow [Model-management terminology](./terminology.md).

## Ownership

ICN owns the recommendable catalog, exact packages and sources, the managed model store, installed
inventory, package inspection, and download attempts. ACN observes these independent authorities
and produces product projections. Clients initiate mutations through ACN and do not infer command
authorization or completion from cached projections.

Catalog, package presence, download activity, inspection, assessment, provider offering, slot
selection, and runtime residency remain separate facts.

## Release catalog

The recommendable catalog is immutable release data. Human-reviewed declarations identify curated
targets and evidence. A separate lock maps each declaration to an immutable upstream revision.
Advancing that lock is an explicit development operation.

Catalog generation resolves only locked revisions through production package construction,
inspection, template analysis, and native planning. It emits a self-contained planner-input bundle
and proves that compact planner inputs yield the same native assessments as their source metadata
at every standard profile. Any unresolved entry, incomplete coverage, integrity mismatch, or
assessment mismatch fails generation or ICN readiness.

Runtime catalog use performs no upstream discovery and does not follow mutable revisions. Adding a
model, format, or upstream revision requires a new reviewed release catalog.

## Package resolution

The shipped package manifest is authoritative for file paths, sizes, digests, roles, and
relationships. A repository revision is only a retrieval address. Downloaded content is published
only after it matches the shipped package exactly.

An explicitly declared fallback may change the retrieval commit when the primary revision is no
longer available. It must not change package identity, target composition, capabilities, planner
inputs, or recommendation evidence.

Fallback is attempted only when the pinned revision or required file is definitively absent. An
authentication, authorization, rate-limit, timeout, or server failure preserves the original
failure. A fallback commit is accepted only when every required relative path, byte size, and
SHA-256 digest matches the shipped manifest. Otherwise acquisition fails without changing the
catalog entry or any installed copy.

## Stores and inventory

The configured managed store is authoritative for Magnitude-owned installations. External model
caches may be supplied as explicit read-only roots; they remain externally owned and are never
silently moved, deleted, or adopted into the managed store.

The standard Hugging Face cache root is resolved once by ACN from `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, then the user-home default, in that order, and
is passed to ICN explicitly. Magnitude uses only credentials explicitly supplied to it; ambient
host-login credentials are not inherited.

Installed inventory reports packages currently present in configured sources and their inspection
results. Inventory discovery is hardware-independent: it does not choose profiles, calibrate
hardware, assess configurations, or recommend models. Failure to inspect one package is isolated to
that package.

Initial external-source reconciliation is asynchronous and does not delay ACN readiness. The
observer publishes an initial snapshot, then atomically replaces it with a completed scan; later
scans retain the previous complete snapshot while work is in flight. Reconciliation owns filesystem
discovery, content hashing, GGUF inspection, tensor-storage derivation, and package construction.
Inventory queries only return the current materialized snapshot and perform no filesystem work.

Package inspection is one of `Pending`, `Inspected`, `Invalid`, or `Incompatible`. Only an inspected,
compatible package may participate in loading. Draft, MTP, and projector components retain their
roles and relationships and are not projected as standalone models unless explicitly catalogued as
standalone targets.

Discovery prefers authoritative GGUF role evidence. Exact filename tokens may propose a
relationship only when authoritative evidence is unavailable; file size never determines role.
`DraftFor(target, method)` and `MtpFor(target)` remain distinct relationships.

## Downloads

A download command addresses an exact offering target and is idempotent over its required packages.
ICN atomically admits missing package work as new attempts or joins already admitted equivalent
work. The caller waits for the exact admitted attempt identities; it does not wait for a later
inventory-wide refresh.

Each attempt publishes bounded progress and exactly one terminal outcome. Caller interruption after
admission detaches that waiter but does not abandon shared work. Cancellation is an explicit domain
command. A retry creates a new attempt. Historical success never proves current presence, and
historical failure never becomes permanent package state.

Completed content is verified before atomic publication. Partial and resumable content is not
reported as installed. Deleting or externally removing an installed package changes inventory
independently of attempt history.

A managed transfer incrementally hashes successfully written bytes and checkpoints the
serializable digest state with its exact committed offset. Resume truncates any uncommitted tail and
continues without rereading the downloaded prefix. Missing or invalid checkpoint evidence discards
the untrusted partial artifact. Publication compares the final accumulated digest before the
atomic move.

## ACN product projection

ACN builds stable target-level models by joining:

- recommendable targets;
- packages referenced by durable provider offerings;
- installed inventory and inspection;
- current download attempts; and
- complete configuration availability.

Installed targets remain visible when catalog loading or recommendation generation fails.
Catalog-only targets remain visible as not downloaded. ACN may immediately merge an exact command response into its
observer to reduce presentation latency, but only ICN observations confirm package state.

Acquisition actions address the row's `ModelOfferingTargetId`. ACN resolves the authoritative exact
target and admits downloads for its required packages. The client-facing admission result reports
that the target is already installed or carries the exact download-attempt identities needed for
subsequent cancellation and observation. ICN package-download commands address targets or packages,
never configuration or provider identity.

Download state stores no assessment evidence and determines neither configuration, offering, nor
slot identity. Exact admission snapshots are merged immediately into ACN observation, while ICN
attempt state remains authoritative for completion.

## Concurrency and recovery

Operations that could publish or remove the same package serialize. Reads may share in-flight
resolution and inspection. Removal must reject or wait while a runtime owns package use; it never
invalidates a resident instance silently.

Derived inspection, source-resolution, assessment, and timing caches are disposable. Corruption or
deletion causes the smallest possible cache miss and never removes installed models, reconstructs
the release catalog, changes identity, or creates a permanent failed state.

## Conformance

- Catalog membership implies only eligibility for assessment and recommendation.
- Runtime setup requires no network access to reconstruct the release catalog.
- Installed inventory reports presence and inspection, never inferred model assessment.
- Download admission and completion depend on exact attempt state, not reconciliation timing.
- Package identity is independent of paths, display names, and mutable upstream references.
- Acquisition uses target identity; configuration selection uses serving-configuration identity;
  provider-model identity begins with the resulting offering.
- Catalog failure cannot hide installed targets.
- One package failure cannot corrupt unrelated inventory or download state.
