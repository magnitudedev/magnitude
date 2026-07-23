---
applies_to:
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/icn/src/downloads/**
  - packages/acn/src/local-model-packages.ts
---

# ICN model management

ICN owns model acquisition, the managed model store, package inspection, the recommendable model
catalog, download attempts, and installed-package inventory. These are separate observations; no
single endpoint joins them with serving configuration, recommendation, provider offering, or
runtime residency.

## Managed store

The configured model store is authoritative for Magnitude-managed local models. ICN does not
implicitly adopt host directories. The product-managed ACN resolves the active standard Hugging
Face Hub cache from `HF_HUB_CACHE`, the deprecated `HUGGINGFACE_HUB_CACHE`, `HF_HOME`,
`XDG_CACHE_HOME`, or the user-home default, in that precedence order. When the resolved directory
exists, ACN supplies it to ICN as an explicit read-only cache root. A nonstandard per-operation
`cache_dir` remains an explicit deployment input because it cannot be inferred from process
configuration.

External cache artifacts are locally available but remain externally owned. Inventory may inspect,
fit, offer, and load them without copying them into the managed store. Magnitude must not delete or
mutate them.

Downloaded model artifacts are authoritative data. Derived indexes, inspections, resolutions, and
assessments are disposable caches.

## Model packages

A model package is an immutable manifest of exact files, roles, relationships, properties, and one
source. Package identity covers the canonical file identities and relationships. Mutable repository
refs, filenames, local paths, display names, and download attempts are not package identity.

ICN resolves Hugging Face sources to immutable commits and exact selected files before publishing a
package. A local package does not claim Hugging Face provenance unless it has an established source.

Package inspection produces `Pending`, `Inspected`, `Invalid`, or `Incompatible`. Invalid and
incompatible packages retain specific diagnostics and are never loadable.

## Recommendable model catalog

`GET /v1/models/catalog` returns the recommendable model catalog. Every entry contains an exact
package target or speculative pair, eligible serving profiles, capabilities, and curation evidence.

Catalog membership means only that Magnitude may assess and recommend the target. It does not mean
the target fits, is recommended, is installed, is offered, or is resident.

Catalog resolution pins immutable Hugging Face commits and exact files. One invalid entry produces
a catalog diagnostic without suppressing valid siblings.

Source-backed assessment and fitting keep any temporary sparse materialization alive for the entire
request. Temporary paths are never returned as durable package locations or retained in assessment
caches.

The catalog is ICN-owned. ACN owns assessment batching and recommendation policy.

## Installed packages

`GET /v1/models/installed` returns only packages currently installed in configured local sources,
including their local path and inspection result.

It does not return:

- catalog-only packages;
- active download attempts;
- serving profiles or fit conclusions;
- provider offerings or selections; or
- runtime residency.

`DELETE /v1/models/installed/{packageId}` removes one installed package when safe. Removal does not
delete catalog membership or durable ACN offerings; those offerings become unavailable until the
package is installed again.

## Downloads

A download is one managed attempt to install an exact package:

```text
Pending -> Downloading -> Completed
                      ├-> Failed
                      └-> Cancelled
```

`POST /v1/models/downloads` starts an attempt. List, detail, and cancel operations return
authoritative attempt snapshots.

Failure and cancellation are terminal attempt results, not package states. Retry creates a new
attempt. ACN projects only the latest relevant attempt into package UI state and stores dismissal
of a surfaced failure as product acknowledgement. Failed attempts retain completed and total byte
counts so resumable progress remains observable.

Successful publication is atomic: incomplete staging is never reported as installed. ICN validates
the complete package before publication. Interrupted attempts recover as terminal failures or are
cleaned without leaving a false installed record.

## ACN package projection

ACN builds `ModelPackageEntry` values by joining:

- targets present in the recommendable catalog;
- packages referenced by durable local offerings;
- installed packages; and
- current download attempts.

This join changes only product presentation. The immutable `ModelPackage` value is reused unchanged.
An entry's local state is `NotInstalled`, `Downloading`, or `Installed`; its last surfaced download
failure and retained progress are separate.

Installed packages appear even when catalog resolution or assessment is unavailable. Catalog-only
packages appear as not installed. Download progress does not require inventory-wide reconciliation.

## Concurrency

Operations addressing the same package serialize where publication or removal could conflict.
Concurrent reads may share in-flight source resolution and inspection. A completed attempt cannot
overwrite a newer authoritative installation or a user removal.

Model loading holds package use through the runtime boundary. Removal of an actively required
package is rejected or waits according to the runtime's explicit ownership contract; it never
silently invalidates a resident model.

## Cache behavior

ICN uses the shared model-derived cache for source resolution, inspection, and assessment evidence.
Keys include immutable package identity and every behavior-changing runtime or hardware input.

Malformed, missing, stale, or unreadable entries are misses at the smallest independent unit.
Deleting the cache may repeat work but cannot remove installed models, change identity, or produce a
permanent failed state.

## Acceptance criteria

- Catalog, installed packages, downloads, offerings, and residency remain distinct.
- Installed listing returns installed packages only.
- Download failure belongs to one attempt and can be retried with a new attempt.
- Package identity is independent of paths and mutable repository refs.
- Catalog failure does not hide installed packages.
- Cache corruption cannot make a valid package permanently unloadable.
- ICN stores no durable product serving configuration or slot selection.
