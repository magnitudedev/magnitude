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
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/local-models.ts
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - packages/acn-protocol/src/schemas/model-state.ts
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
a deterministic manifest with both the exact source GGUF-header ranges and the identities of their
compact planner stubs. A stub is standard GGUF: it retains all model metadata and tensor
descriptors and supplies a compressible placeholder vocabulary of the original cardinality so
retained graph metadata is loaded through llama.cpp's ordinary vocabulary path. Auxiliary split
components receive no primary overrides. A stub contains no tensor payload and is not a runnable
model. The transformation is generic across architectures and has no curated architecture-key
table.

Generation fails if any source entry produces a diagnostic or if the native planner gives a
different hardware assessment for a source-header sparse model and its compact-stub sparse model
at any curated serving profile. An explicit hydration command retrieves and verifies the source
ranges from pinned immutable revisions, reruns the same deterministic compactor, verifies every
stub identity, and materializes the compressed stub bundle. Release builds package the lock file
and bundle as ICN catalog sidecars; local development constructs the same installation layout.
The bundle is not committed to source control, and release validation rejects a partial catalog.
Normal ICN startup validates the installed sidecars against the source declarations, native
template and planner identities, bundle digest, and exact catalog coverage. It performs no remote
catalog refresh.

At runtime, a release-catalog target is assessed from its installed hardware-independent properties
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

`POST /v1/models/downloads` starts or joins the exact package attempt and returns that attempt.
List, detail, and cancel operations return authoritative attempt snapshots. Active snapshots retain
transfer stage, completed and total bytes, and the measured current-attempt transfer rate when
sufficient evidence exists. Resumed bytes never count as bytes transferred during the current
attempt. Product clients may derive a remaining-time estimate from those authoritative values but
never measure transfer speed themselves.

Failure and cancellation are terminal attempt results, not package states. Retry creates a new
attempt. ACN projects only the latest relevant attempt into package UI state and stores dismissal
of a surfaced failure as product acknowledgement. Failed attempts retain completed and total byte
counts so resumable progress remains observable.

Successful publication is atomic: incomplete staging is never reported as installed. ICN validates
the complete package before publication. Interrupted attempts recover as terminal failures or are
cleaned without leaving a false installed record. Retrying reuses resumable partial data or
recognizes a package that was already published.

Source-integrity hashing is part of the download write path. A fresh managed transfer feeds each
successfully written source chunk into its whole-file SHA-256 accumulator and never rereads the
completed file merely to reconstruct the digest. The downloader durably checkpoints the serializable
digest state and its exact byte offset alongside each partial component. Recovery restores that
constant-size state, truncates any uncommitted tail, and resumes the range transfer without reading
the downloaded prefix. Controlled interruption checkpoints the current offset; unclean interruption
may redownload only the bounded interval after the latest periodic checkpoint. A completed blob has
a content-bound final checkpoint, so reuse also requires no integrity reread. Missing or invalid
checkpoint state is never repaired by hashing a large artifact; the untrusted artifact is discarded
and downloaded again. Final verification compares the accumulated digest before atomic publication.

`Completed` records that the attempt successfully published its exact package. Installed inventory,
not attempt history, owns
whether the package is currently present; a user may remove a package after a successful attempt.
Package presentation then becomes `NotInstalled`, never 100% active progress or a fabricated failed
attempt. This later current-state change does not rewrite or delay the completed command.

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
One package-acquisition derivation joins installed inventory with the latest relevant attempt and
produces `NotInstalled`, `Downloading`, `DownloadFailed`, or `Installed`. It owns presentation only.
An admitted target-acquisition waiter instead retains its exact attempt identities and reads those
attempts directly until each is terminal.
The target-level product projection aggregates progress and concurrent transfer rates across every
package required by that target.

Catalog assessment stores only candidate metadata. The final ACN `LocalModels` projection attaches
the already-derived target download and preparation values to client-ready catalog candidates.
There are no placeholder candidate lifecycles and no fallback source of target state.

`DownloadModel` is one idempotent target-acquisition operation: it admits missing package attempts
and waits for their exact authoritative terminal outcomes. It performs no installed-inventory
refresh. Clients do not precompute whether a download is required. Observed inventory and attempt
snapshots never gate admission; ACN sends the command directly to ICN, whose package operation is
idempotent. Every exact attempt snapshot returned by admission or command polling is immediately
merged into the download observer so presentation does not wait for a periodic polling interval;
this local publication is not external confirmation and cannot determine command completion.
Reconciliation therefore cannot delay either the start or completion of a download.
Local assignment records the exact provider offering without using cached installed presence as
command authorization. ICN load admission validates the current package files and remains the final
load-time authority. A frontend flow that composes download, slot
assignment, loading, completion, and cancellation represents the command and its lifecycle as one
composite Effect Atom; React does not coordinate those mutation results or copy mirror state.

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
or produce a permanent failed state. The installed release catalog and planner-input bundle are
intentionally outside this cache contract.

## Acceptance criteria

- Catalog, installed packages, downloads, offerings, and residency remain distinct.
- Installed listing returns installed packages only.
- Installed listing performs no hardware assessment and isolates artifact inspection failures.
- Draft and MTP execution companions are never listed as standalone installed models.
- Download failure belongs to one attempt and can be retried with a new attempt.
- Managed download integrity never rereads completed or resumable model content.
- A completed attempt records successful historical publication; installed inventory independently
  owns current presence.
- `Completed` with an absent package projects as `NotInstalled`, never active progress or a failed
  historical attempt.
- Target acquisition waits only on its exact admitted attempts and performs no installed refresh.
- Package and target presentation have one package-outcome derivation that does not authorize or
  complete commands.
- Assignment does not use cached installed presence as command authorization; load validation
  belongs to ICN.
- Composite client workflows are atom-owned and render authoritative mirror state.
- Package identity is independent of paths and mutable repository refs.
- Catalog failure does not hide installed packages.
- User setup never requires network access to reconstruct the curated release catalog.
- A release catalog with unresolved entries or mismatched generation evidence fails release
  validation and ICN readiness.
- Every generated compact planner stub is semantically identical to its source-header sparse model
  for native hardware assessment at every curated serving profile.
- Cache corruption cannot make a valid package permanently unloadable.
- ICN stores no durable product serving configuration or slot selection.
