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
  - inference/catalog/**
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

Hub access uses only credentials explicitly supplied to Magnitude. Ambient credentials from a
host Hugging Face login are not inherited, so an expired or unrelated cached login cannot break a
public curated-model download. An explicit `HF_TOKEN` remains available for authenticated sources.

## Model packages

A model package is an immutable manifest of exact files, roles, relationships, properties, and one
source. Package identity covers the canonical file identities and relationships. Mutable repository
refs, filenames, local paths, display names, and download attempts are not package identity.

ICN resolves Hugging Face sources to immutable commits and exact selected files before publishing a
package. A local package does not claim Hugging Face provenance unless it has an established source.

Package inspection produces `Pending`, `Inspected`, `Invalid`, or `Incompatible`. Inspected GGUF
packages carry both llama.cpp's authoritative file-type family name and a coarse user-facing
bit-width name. Invalid and incompatible packages retain specific diagnostics and are never
loadable.

## Recommendable model catalog

`GET /v1/models/catalog` returns the recommendable model catalog. Every entry contains an exact
package target or speculative pair, eligible serving profiles, capabilities, and curation evidence.
An eligible serving profile identifies the per-request context capacity only. Parallel execution
capacity is resolved from current hardware when the installed configuration is loaded and is not
part of catalog or configuration identity.

Catalog membership means only that Magnitude may assess and recommend the target. It does not mean
the target fits, is recommended, is installed, is offered, or is resident.

Catalog publication is an explicit release-development operation. It resolves the human source
catalog through production ICN code, pins immutable Hugging Face commits and exact files, and emits
a deterministic manifest with the exact byte ranges and content digests of the GGUF metadata
needed for native planning. Generation fails if any source entry produces a diagnostic.
Development and release builds materialize the deterministic compressed bundle in build output,
retrieving and verifying those ranges from the pinned immutable revisions when a matching build
artifact is absent, and embed it in the binary. The bundle is not committed to source control, and
a build never embeds a partial catalog.
Normal ICN startup validates the embedded artifacts against the source declarations, native
template and planner identities, bundle digest, and exact catalog coverage. It performs no remote
catalog refresh.

At runtime, a release-catalog target is assessed from its embedded hardware-independent properties
and planner inputs, combined with current machine topology and local calibration. ICN materializes
the planner inputs as temporary sparse files off the asynchronous runtime and keeps them alive for
the entire assessment. Temporary paths are never returned as durable package locations or retained
in assessment caches. Installed targets use their authoritative local files. Arbitrary remote
source targets outside the release catalog are not resolved or assessed by the product runtime.
Consequently a complete setup makes no Hugging Face request and creates no remote-header cache.

The catalog is ICN-owned. ACN owns runtime assessment batching and recommendation policy.

## Installed packages

`GET /v1/models/installed` returns only packages currently installed in configured local sources,
including their local path and inspection result.

Installed-package reconciliation discovers artifacts and obtains hardware-independent inspection
facts only. It never calibrates hardware, chooses a serving profile, or runs fit assessment.
Failure to inspect one artifact is reported on that package and does not fail or restart the
inventory-wide scan. Recognized draft and MTP companion GGUFs remain available in their source
cache for explicit paired configurations but are not projected as independently loadable models.
Discovery prefers authoritative GGUF role evidence (`eagle3` architecture or DFlash decoder
metadata), then exact role tokens in filenames. The `mtp`, `eagle3`, and `dflash` tokens follow
the GGUF naming convention or llama.cpp's own companion discovery; `draft` is a compatibility
fallback for older or vendor-specific conversions. File size is never used to infer a model's
execution role. Package materialization preserves `DraftFor(target, method)` separately from
`MtpFor(target)`; neither relationship is rewritten as the other.

Product startup does not wait for external-source reconciliation. The installed-package observer
publishes an empty initial snapshot, starts reconciliation immediately in the background, and
publishes the complete result when ICN returns it. Later refreshes retain the previous complete
snapshot while reconciliation runs. Large external caches therefore cannot prevent ACN from
becoming healthy.

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
authoritative attempt snapshots. Active snapshots retain transfer stage, completed and total bytes,
and the measured current-attempt transfer rate when sufficient evidence exists. Resumed bytes never
count as bytes transferred during the current attempt. Product clients may derive a remaining-time
estimate from those authoritative values but never measure transfer speed themselves.

Failure and cancellation are terminal attempt results, not package states. Retry creates a new
attempt. ACN projects only the latest relevant attempt into package UI state and stores dismissal
of a surfaced failure as product acknowledgement. Failed attempts retain completed and total byte
counts so resumable progress remains observable.

Successful publication is atomic: incomplete staging is never reported as installed. ICN validates
the complete package before publication. Interrupted attempts recover as terminal failures or are
cleaned without leaving a false installed record.

ICN owns an accepted attempt independently of the HTTP caller. ACN's download observer periodically
refreshes even when its last snapshot is idle, then uses the faster active interval until every
attempt is terminal. Observation therefore cannot depend on the initiating request completing a
post-acceptance refresh.

## ACN package projection

ACN builds `ModelPackageEntry` values by joining:

- targets present in the recommendable catalog;
- packages referenced by durable local offerings;
- installed packages; and
- current download attempts.

This join changes only product presentation. The immutable `ModelPackage` value is reused unchanged.
An entry's local state is `NotInstalled`, `Downloading`, or `Installed`; its last surfaced download
failure and retained progress are separate. The target-level product projection aggregates progress
and concurrent transfer rates across every package required by that target.

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

ICN uses the shared `.magnitude/cache` root for source resolution, inspection, and assessment
evidence. The authoritative `.magnitude/models` store contains installed artifacts and the managed
Hugging Face hub, never disposable derived-cache namespaces.
Successful model loads also retain bounded, disposable phase-duration evidence for adaptive loading
progress. Exact observations are keyed by content, context-only serving profile, resolved physical
context and parallel allocation, native build, path-independent resolved plan identity, selected
acceleration, phase shape, and process-residency class; cross-model observations from the same
native environment are workload-scaled fallbacks. Failed or canceled loads never train this cache.
Keys include immutable package identity and every behavior-changing runtime or hardware input.

Malformed, missing, stale, or unreadable cache entries are misses at the smallest independent unit.
Deleting the cache may repeat machine-specific inspection and assessment, but cannot remove
installed models, reconstruct the release catalog, trigger model-header downloads, change identity,
or produce a permanent failed state. The embedded release catalog and planner-input bundle are
intentionally outside this cache contract.

## Acceptance criteria

- Catalog, installed packages, downloads, offerings, and residency remain distinct.
- Installed listing returns installed packages only.
- Installed listing performs no hardware assessment and isolates artifact inspection failures.
- Draft and MTP execution companions are never listed as standalone installed models.
- Download failure belongs to one attempt and can be retried with a new attempt.
- Package identity is independent of paths and mutable repository refs.
- Catalog failure does not hide installed packages.
- User setup never requires network access to reconstruct the curated release catalog.
- A release catalog with unresolved entries or mismatched generation evidence fails release
  validation and ICN readiness.
- Cache corruption cannot make a valid package permanently unloadable.
- ICN stores no durable product serving configuration or slot selection.
