---
applies_to:
  - inference/catalog/**
  - inference/crates/icn-catalog/**
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-contracts/src/models.rs
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/icn/src/downloads/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
---

# Model catalog and acquisition

This document defines catalog publication, package resolution, artifact presence, inventory,
download, and deletion behavior. Terms follow [Model-management terminology](./terminology.md).

## Ownership

ICN owns the recommendable catalog, exact packages and sources, the managed model store, filesystem
inventory, and process-local download coordination. ACN observes those authorities and produces
product projections. Clients initiate mutations through ACN and do not infer command authorization
or completion from cached projections.

Catalog membership, artifact presence, download activity, inspection, assessment, provider
offering, slot selection, and runtime residency remain separate facts.

Catalog attribution across exact artifact or drafter changes follows
[Intrinsic catalog target mapping](./intrinsic-target-mapping.md).

## Release catalog

The release catalog is immutable release data. Human-reviewed declarations identify each artifact
variant by its stable format-qualified variant ID, exact upstream format selector, short variant label, fidelity rank, and
quantization-aware-training fact. Each entry publishes one exact default
`ModelServingConfiguration`, required package components, presentation, and recommendation
evidence. Reviewed model parameterization states whether the architecture is dense or
mixture-of-experts, its positive total parameter count, and, only for mixture-of-experts, a
positive active parameter count smaller than the total. Parameter counts are factual catalog data;
clients own their human-readable rounding and formatting. Input modality remains an inspected
capability and is not duplicated in catalog declarations. Speculative decoding is explicit as
either capability embedded in the target GGUF or an
exact draft file in the target or another repository. Every package source is locked.

Catalog generation resolves only locked revisions through production package construction,
inspection, template analysis, and native planning. It emits a self-contained planner-input bundle
and proves that compact planner inputs yield the same native assessments as their source metadata.
Repeated references to one immutable package reuse one package identity and content payload.
Incomplete coverage, integrity mismatch, invalid relationships, or assessment mismatch fails
generation or ICN readiness.

An issued configuration remains resolvable by its canonical identity across later releases.
Deprecation excludes an entry from recommendation and first-time discovery without deleting its
configuration or package declaration. Existing selections, installed artifacts, and explicit
reacquisition can therefore resolve the same bundle and profile without a user-state copy of the
catalog entry. Runtime catalog use performs no upstream discovery and does not follow mutable
revisions.

## Package resolution

The shipped package declaration is authoritative for catalog file paths, sizes, digests, roles,
and relationships. A repository revision is only a retrieval address. Downloaded content is
accepted only after it matches the exact package.

An explicitly declared fallback may change the retrieval commit when the primary revision is no
longer available. It must not change package identity, bundle composition, capabilities, planner
inputs, or recommendation evidence. Authentication, authorization, rate-limit, timeout, and server
failures preserve their original classification. A fallback is accepted only when every required
path, size, and digest matches.

## Artifact stores and inventory

Completed model files are the sole authority for physical presence. A package is installed when all
files required by that package are currently present and valid. A bundle is installed when all of
its required packages are installed. No registry, configuration, download record, cache entry, or
historical success can make files present or absent.

Magnitude-owned artifacts live in the configured `ManagedModelStore`. Explicit Hugging Face roots are
read-only external sources with origin `HuggingFaceCache`; they are never moved, deleted, or adopted.
Origin affects ownership-sensitive operations, not package identity or runtime eligibility.

Each repository in the Magnitude-owned store may contain any number of complete revision snapshots.
Each snapshot directory names its immutable revision and contains links to installed
content-addressed blobs. Publishing a package is additive: it adds that package's exact components
to its revision snapshot and does not infer that other components or revisions are obsolete.

ICN derives managed inventory by bounded, containment-safe enumeration of every complete snapshot.
The observation path never creates, repairs, or removes artifact links. Component paths, sizes, content
identities, roles, shards, and relationships come from existing validated files, exact catalog
matches, and artifact inspection. Unsafe links, special files, escaping paths, and invalid entries
make only the affected package unavailable. Store mutations reconcile owned path topology by
quarantining conflicting nodes before writing; they never follow a conflicting link or overwrite an
unclassified node. Explicit external Hugging Face roots retain their ordinary
multi-revision layout and remain read-only.

Temporary and `.incomplete` files never contribute presence. A partial multi-file package is not
installed. A complete independently servable weights package may remain installed while an
optional companion or a larger bundle is incomplete.

Inventory reconciliation owns filesystem discovery, hashing, GGUF inspection, tensor-storage
derivation, and package construction. One cached GGUF inspection supplies every derived property
for an immutable component. Reconciliation publishes complete inventory snapshots and retains the
previous complete inventory snapshot while a refresh is in flight. Queries return the current materialized snapshot
and perform no filesystem work. Discovery is hardware-independent and performs no network access,
assessment, calibration, profile choice, or recommendation.

All inventory indexes, content hashes, inspections, and derived package evidence are optimistic
caches. Missing, stale, malformed, unreadable, or unwritable cache state causes scoped recomputation
and cannot remove a valid artifact from inventory. Catalog failure cannot hide independently
servable artifacts.

## Downloads

A download command carries an exact servable bundle. If every required package is installed, ICN
returns `AlreadyInstalled`. Otherwise ICN creates one process-local `ModelDownload` with a stable
`ModelDownloadId` and admits missing package work or joins equivalent work already owned by the
same ICN process.

The model download aggregates bounded progress and one terminal outcome across its bundle. Package
attempts are process-local ICN details. Equivalent model downloads may share active package work.
Caller interruption detaches that waiter without abandoning admitted work. Cancellation stops
shared package work only when no other live occurrence depends on it. A retry creates a new
occurrence. Restart ends all occurrences, attempt history, cancellation state, and failure
dismissal.

Progress is relative to the work admitted by that download occurrence. Its total is the byte size
of missing packages whose attempts it owns or joins, and completed bytes come only from those
attempts. Packages already installed at admission contribute neither baseline progress nor total.
Consequently a first installation measures the whole missing bundle, while an update measures only
its actual artifact delta.

Expected failures distinguish insufficient disk space, interruption, unavailable source content,
unavailable network, local storage failure, and corrupt content. Cancellation is a separate
terminal result. Structured facts, including required and available byte counts, cross boundaries
without parsing diagnostic prose.

Failure dismissal is process-local presentation state correlated by the exact download identity.
It does not alter the terminal result and does not survive restart.

Downloads write only to temporary `.incomplete` paths until a component matches its expected size
and digest. A completed component is then exposed as an ordinary content-addressed blob and snapshot
entry. Download and deletion operations for one managed repository serialize through one
repository-scoped lock. The first package for a revision is published from a staged directory;
later packages for that revision add only their missing exact links. Publishing one revision never
removes another revision. A failed or cancelled acquisition therefore leaves every previously
complete package available.
No metadata commit follows file publication, and failure to write derived state cannot fail a
successfully published component.

A partial component may carry a narrowly scoped integrity checkpoint containing only the expected
content identity and size, committed offset, and serializable digest state. The retry command
supplies acquisition intent. Missing or invalid checkpoint evidence discards the reusable prefix;
it never hides completed files or fails startup. Checkpoints have no format-version gate.

## Product projection

Catalog and acquisition contribute catalog configurations, current package observations, and
process-local download state to the [local-model product projection](./local-model-product-projection.md).
They do not persist provider offerings or serving configurations.

Every independently servable installed non-catalog package contributes a standalone row. Installed
catalog targets are attributed to `CatalogIdentity` by the exact mapping defined in
[Intrinsic catalog target mapping](./intrinsic-target-mapping.md). One catalog product row retains
curated presentation, including its reviewed variant label, across target, repository, and drafter
changes. A non-catalog row derives its variant label from inspected target-package quantization.
Artifact format and inspection evidence are not copied into parallel product-presentation fields.

Catalog reconciliation addresses a `CatalogIdentity`. ICN compares desired package IDs with current
filesystem presence and exact package affiliations, acquires missing desired packages, and removes
affiliated superseded packages after the desired bundle is complete. It does not materialize or
persist configuration state. The response carries the deterministic local provider-model identity
and, when work was admitted, the process-local download identity used for observation and
cancellation.

Superseded cleanup is an invariant of the authoritative installed set, not a task owned by the
request that admitted a download. After an installed-set change is committed, one coalesced catalog
maintenance pass removes exact affiliated superseded package IDs only for catalog models whose
entire desired package set is present. Cleanup failure is logged and may be retried after a later
authoritative change. There is no request poller and no package-attempt completion trigger.

Download state determines neither configuration, offering, slot identity, nor physical presence.
On completion ACN refreshes inventory before presenting the occurrence as complete. The filesystem
observation remains authoritative.

## Deletion and garbage collection

Deletion plans derive exclusively from current managed files and the addressed package or bundle.
ICN validates containment, scans every repository snapshot, removes only the addressed package's
entries, and removes a blob only when no remaining snapshot entry references its proven content identity.
Bytes reclaimed are measured from blobs actually removed. Inventory cannot recreate deleted links
from catalog declarations or retained blobs.

If safe containment or complete reference accounting cannot be proven, deletion fails without
guessing. Interrupted deletion is represented by the files that remain and may be retried
idempotently. Conservative garbage collection may remove blobs proven unreferenced. Runtime
ownership rejects or waits for removal of files used by a live model instance.

Deleting artifacts does not delete slot selection, favorites, or recency. Those are user intent and
may remain unresolved until the exact catalog configuration and required files are available again.

## Concurrency and recovery

Operations that publish, replace, or remove files in the same managed repository serialize. Reads
may share in-flight hashing and inspection. Package download sharing and cancellation exist only
within the owning ICN process.

Startup derives inventory from current files and configurations from catalog plus serving policy.
There is no model-state recovery epoch. Later filesystem or catalog observations reconcile through
the same continuous derivation.

The standard-profile construction rule is canonical for a package identity. A release must not
reinterpret an existing package identity as a different standard configuration. A new standard
profile requires a distinct explicit configuration authority rather than changing resolution of an
already issued provider-model identity.

## Conformance

- The same valid filesystem produces the same package inventory independent of upgrade history.
- Removing every derived cache changes cost only, never package presence or user intent.
- Catalog and standard configurations are not copied into durable model state.
- Issued catalog configurations remain resolvable after deprecation.
- Standard configuration identity remains stable for a given package identity.
- Installed inventory is derived without network access or hardware assessment.
- Partial, unsafe, invalid, or digest-mismatched content is not installed.
- Complete packages from multiple repository revisions may coexist until exact catalog affiliation
  state identifies a package as superseded and eligible for removal.
- Inventory reconciliation never mutates managed artifacts.
- One artifact failure cannot hide unrelated valid artifacts.
- Download identities and terminal history do not survive ICN restart.
- Valid partial bytes may be reused only with matching integrity evidence.
- Historical download completion never proves current presence.
- Shared blobs are deleted only after current filesystem references prove them unreferenced.
- Catalog failure cannot hide independently servable installed packages.
