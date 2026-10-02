---
applies_to:
  - inference/catalog/**
  - inference/service/catalog-tool/**
  - inference/service/models/**
  - inference/service/server/src/main.rs
  - inference/service/contracts/src/inventory.rs
  - inference/service/contracts/src/models.rs
  - packages/icn/src/models/**
  - packages/icn/src/events/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/acn-protocol/src/schemas/local-model-projection.ts
  - packages/ai/src/provider/model.ts
  - packages/client-common/src/local-models/**
  - packages/client-common/src/utils/model-presentation.ts
  - cli/src/commands/inference-runtime.ts
  - desktop/src/renderer.tsx
---

# Model catalog and acquisition

This document defines catalog publication, package resolution, artifact presence, inventory,
download, and deletion behavior. Terms follow [Model-management terminology](./terminology.md).

## Ownership

ICN owns the recommendable catalog, exact packages and sources, the managed model store, filesystem
inventory, and process-local download coordination. ACN observes those authorities and produces
product projections. Clients initiate native mutations through ACN's transparent inference proxy
and do not infer command authorization or completion from cached projections.

Catalog membership, artifact presence, download activity, package validation, assessment, provider
offering, slot selection, and runtime residency remain separate facts.

Live Hugging Face search and repository resolution are service-owned discovery queries. Search
returns GGUF repositories with immutable commit identities; resolution returns the requested
repository's immutable snapshot and GGUF file evidence. These queries do not change the release
catalog or managed inventory, and catalog use does not depend on them.

Catalog attribution across exact artifact or drafter changes follows
[Intrinsic catalog target mapping](./intrinsic-target-mapping.md).

## Release catalog

The release catalog is immutable release data. Human-reviewed declarations identify each artifact
variant by its stable format-qualified variant ID, exact upstream format selector, short variant label, fidelity rank, and
quantization-aware-training fact. Each model declaration records the calendar date on which the
represented upstream model or material named revision was first publicly released. Artifact
conversions, quantizations, quantization-aware packaging, and provider publication do not change
that date; every artifact variant inherits the model declaration's date. Each entry publishes one exact default
`ModelServingConfiguration`, required package components, presentation, and ranking evidence.
Published catalog rows reduce package sources to deduplicated HTTPS repository links for product
presentation; package coordinates and bundle structure remain private.
A catalog declaration names no context length. The serving profile's context is the target's
supported maximum context from its GGUF metadata, the context the engine resolves and serves;
memory is elastic, so a long supported context no longer reserves memory up front.
Every active model carries one model-level Artificial Analysis score with its declared Artificial
Analysis Intelligence Index methodology version. A direct score records the observation date and
canonical Artificial Analysis model URL. When no direct result exists, or Artificial Analysis marks
its own result as estimated, an estimate is a structurally distinct value that records its target
scale, methodology, observation date, and non-empty primary evidence URLs. The catalog declares one
Artificial Analysis frontier: the top Intelligence Index model, its score, methodology version,
observation date, and URL. Every model score shares the frontier's methodology version, because
scores from different versions are not on one scale, and none exceeds the frontier.

A model's intelligence is its Artificial Analysis score as a percentage of the frontier score,
rounded half-up to a whole number. Intelligence is the canonical score that clients display and
ranking uses; Artificial Analysis scores and their provenance are catalog-internal and never leave
the service. Updating the frontier rescales every model. Intelligence is shared by every artifact
variant; variant-specific preservation remains represented only by fidelity rank. Catalog
Artificial Analysis data is reviewed immutable release data and is never refreshed from the network
at runtime.

Reviewed model parameterization states whether the architecture is dense or
mixture-of-experts, its positive total parameter count, and, only for mixture-of-experts, a
positive active parameter count smaller than the total. Parameter counts are factual catalog data;
clients own their human-readable rounding and formatting. Input modality is assessment evidence
and is not duplicated in catalog declarations. An image-capable target requires one
projector component in its exact target package. Catalog generation selects the projector
automatically only when the locked target repository contains one candidate; otherwise the reviewed
declaration names the exact projector path in that repository. Package construction and package
validation are one operation: an image target requires an exact projector component and a typed
relationship to the package weights. Native assessment establishes whether MTMD recognizes image
input. Primary-weight metadata and projector presence alone never establish runnable image
capability. A missing, ambiguous, incompatible, or MTP-combined projector fails generation.
Speculative decoding is explicit as either capability embedded in the target GGUF or an exact draft
file in the target or another repository. Every package source is locked.

Catalog generation resolves only locked revisions through production package construction,
validation, and unified native assessment. It emits self-contained Assessment Material and proves
that compact material yields the same template-derived capabilities and hardware profiles as the
source artifact.
Repeated references to one immutable package reuse one package identity and content payload.
Incomplete coverage, integrity mismatch, invalid relationships, or assessment mismatch fails
generation or ICN readiness.

Removing a model from the release catalog removes all its variants, revision pins, and planner
inputs. Its former catalog IDs no longer resolve; removal does not alias them to a replacement
model or delete downloaded artifacts. Catalog membership and physical artifact presence remain
separate. Runtime catalog use performs no upstream discovery and does not follow mutable revisions.
Deprecation, not removal, is how a release withdraws a model that users may have installed.

### Support lifecycle

Every model declaration states one reviewed support level, shared by all its variants:

- **Supported**: intended to be available as a supported catalog model. Qualification evidence and
  optimization status are tracked separately from this declaration.
- **Disabled** with a non-empty reason: temporarily unavailable or not yet supported. A disabled
  model is never recommended, offered, installed, or loaded. Its installed files remain identifiable
  and removable.
- **Deprecated** with a calendar date, a non-empty reason, and one required exact replacement
  catalog configuration of a supported model. A deprecated model is never
  recommended, offered as a provider model, admitted for installation, or loaded; ICN rejects
  installation and serving with a typed deprecation failure that names the replacement. Its
  declaration, planner inputs, and locked revisions stay in the release catalog, so an existing
  installation remains identifiable. Clients report an installed copy as unsupported, name the
  replacement, offer a one-step switch to it, and remove its files through ordinary catalog
  removal. A lock update never advances a deprecated model's revisions.

Support level and assessment are independent facts. Support is what the release promises for a
model; assessment is what this device can do with it. A supported model can be `DoesNotFit`. A
catalog model is never `Unsupported`: a catalog model the engine cannot execute is a release
defect, whose assessment is dropped with an error. A declaration whose family the engine does not
implement is disabled until support is implemented and qualified.

## Package resolution

The shipped package declaration is authoritative for catalog file paths, sizes, digests, roles,
and relationships. A repository revision is only a retrieval address. Downloaded content is
accepted only after it matches the exact package.

An explicitly declared fallback may change the retrieval commit when the primary revision is no
longer available. It must not change package identity, bundle composition, capabilities, planner
inputs, or ranking scores. Authentication, authorization, rate-limit, timeout, and server
failures preserve their original classification. A fallback is accepted only when every required
path, size, and digest matches.

## Artifact stores and inventory

Completed model files are the sole authority for physical presence. A package is installed when all
files required by that package are currently present and valid. A bundle is installed when all of
its required packages are installed. No registry, configuration, download record, cache entry, or
historical success can make files present or absent.

Magnitude-owned artifacts live in the configured `ManagedModelStore`. Explicit Hugging Face roots are
read-only external sources with origin `HuggingFaceCache`; they are never moved, deleted, or adopted.
The standard environment-derived Hugging Face hub root remains configured even when absent at ICN
startup, so an explicit later discovery refresh can observe a cache created during the same daemon
lifetime. Origin affects ownership-sensitive operations, not package identity or runtime eligibility.

Each repository in the Magnitude-owned store may contain any number of complete revision snapshots.
Each snapshot directory names its immutable revision and contains links to installed
content-addressed blobs. Publishing a package is additive: it adds that package's exact components
to its revision snapshot and does not infer that other components or revisions are obsolete.
Snapshot links must identify the same underlying file as the verified blob: Unix symbolic links
and Windows hard links satisfy this requirement. Equal paths or equal byte sizes alone do not.

ICN derives managed inventory by bounded, containment-safe enumeration of every complete snapshot.
The observation path never creates, repairs, or removes artifact links. Component paths, sizes, content
identities, roles, shards, and relationships come from existing validated files, exact catalog
matches, and package validation. Unsafe links, special files, escaping paths, and invalid entries
make only the affected package unavailable. Store mutations reconcile owned path topology by
quarantining conflicting nodes before writing; they never follow a conflicting link or overwrite an
unclassified node. Explicit external Hugging Face roots retain their ordinary
multi-revision layout and remain read-only.

Each complete independently callable GGUF group in an external Hugging Face root is a discovered
model candidate. A standalone file uses its repository-relative path as its artifact selector; a
complete shard set uses its lexical first shard. Projectors and execution companions belong to the
selected package and do not receive independent model identities. Metadata-only repositories,
Safetensors-only repositories, and incomplete shard sets produce no discovered model.

The canonical discovered identity is
`hf:<owner>/<repository>/<artifact-selector>`. Its three semantic parts are parsed and validated;
the identity never contains an inventory hash, absolute cache path, package ID, revision, or
projector. Multiple quantizations therefore remain distinct when they occupy distinct repository
paths. Quantization presentation comes from bounded GGUF parsing rather than filename parsing.

When several cached revisions or configured roots provide the same repository and selector, exact
identical packages collapse. Candidates referenced by a cached `main` ref outrank other candidates.
Within the same ref priority, the most recently modified cached revision wins; its immutable commit
provides the deterministic final tie-break. This is local cache recency, not a claim about upstream
commit chronology. Selection precedes validation, so an invalid selected revision never falls back
to a stale revision. An exact catalog target that is successfully attributed publishes only under
its catalog identity; failed attribution remains visible through discovery so the installed
artifact does not disappear from inventory.

Inventory retains source occurrences by `InventoryEntryId` until this selection is complete.
Package-content deduplication therefore never erases the repository, selector, root, revision, or
current-ref evidence needed to form and select distinct `hf:` identities. Catalog package presence
may collapse identical package content afterward, while ownership still reflects every retained
physical occurrence.

Temporary and `.incomplete` files never contribute presence. A partial multi-file package is not
installed. A complete independently servable weights package may remain installed while an
optional companion or a larger bundle is incomplete.

Inventory reconciliation owns filesystem discovery, hashing, bounded GGUF parsing, tensor-storage
derivation, package construction, package validation, and local Assessment Material derivation.
Package construction produces the exact package and its one authoritative structural validation
atomically; catalog generation and installed inventory use that same operation. One cached GGUF
parse supplies every structural property for an immutable component. Template-derived capabilities
are produced only by assessment. Reconciliation publishes complete inventory snapshots and retains the
previous complete inventory snapshot while a refresh is in flight. Queries return the current materialized snapshot
and perform no filesystem work. Discovery is hardware-independent and performs no network access,
assessment, calibration, profile choice, or ranking. ICN reconciles once at startup. Product-owned
artifact mutations publish their exact inventory change before completing and do not trigger a
global scan. A later global reconciliation is performed only when an operation explicitly requires
fresh discovery or validation; it is never driven by an unconditional timer.

All inventory indexes, content hashes, GGUF parses, validations, and derived package evidence are optimistic
caches. Missing, stale, malformed, unreadable, or unwritable cache state causes scoped recomputation
and cannot remove a valid artifact from inventory. Catalog failure cannot hide independently
servable artifacts.

## Downloads

A catalog installation command carries the complete catalog-form `ModelId` and converges that
model to present-and-current. ICN admits one `CatalogInstallationOperation` or reports that the
model is already current. The admitted occurrence retains the exact resolved private bundle and
uses process-local download/package work internally; none of those internal identities crosses the
ICN–ACN boundary.

Admission validates each package in the requested bundle against the current materialized package
record and that package's current exact file evidence. It does not await inventory-wide filesystem
reconciliation. Global reconciliation remains the discovery path for external filesystem changes;
it is never a prerequisite for acknowledging a known catalog download.

The catalog installation occurrence retains `ModelId`, aggregates bounded progress, and has one
terminal outcome. ACN observes and addresses that occurrence by `CatalogInstallationOperationId`.
Equivalent operations may share private package work, but sharing never changes their public
model identity or operation state.
Caller interruption detaches that waiter without abandoning admitted work. Cancellation stops
shared package work only when no other live occurrence depends on it. A retry creates a new
occurrence. Restart ends all occurrences, attempt history, cancellation state, and failure
dismissal.

Progress is relative to the work admitted by that catalog installation occurrence. Its total is the byte size
of missing packages whose attempts it owns or joins, and completed bytes come only from those
attempts. Packages already installed at admission contribute neither baseline progress nor total.
Consequently a first installation measures the whole missing bundle, while an update measures only
its actual artifact delta.

An admitted installation ends by optimizing the model for this computer, so a new model's first
load does not pay for kernel tuning. Once its download has verified and published the complete
bundle, the occurrence enters `Optimizing` and runs one preparation of the model on the device a
load would select (see [instance lifecycle](./instance-lifecycle.md)); it then ends `Completed`
however that preparation ends. Optimization progress is its own measure: `preparing` while the
device opens and tuning units are counted, then `tuning` with completed and total units, naming the
previewed device once known. When every unit is already stored it ends straight from `preparing`.
An installation whose admission downloads nothing does not optimize. `Optimizing` is active for
duplicate-installation admission, and the occurrence's observation covers download and
optimization changes alike.

Optimization is an ahead-of-time cost move, never a gate on the model:

- cancellation while optimizing stops the preparation and the occurrence still ends `Completed`;
- a load of the same model stops the preparation and tunes only what is not yet stored;
- removal while optimizing stops the preparation and then proceeds;
- a failed preparation is logged and never becomes an installation failure; and
- restart loses the process-local occurrence, and the next load tunes what is missing.

An engine update that changes tuning keys leaves models to retune on their next load; installations
are not re-optimized after an update.

Expected failures distinguish insufficient disk space, interruption, unavailable source content,
unavailable network, local storage failure, and corrupt content. Cancellation is a separate
terminal result. Structured facts, including required and available byte counts, cross boundaries
without parsing diagnostic prose.

Failure acknowledgement is process-local presentation state on the exact catalog installation
occurrence. ACN resolves a model-addressed user command to the current occurrence; raw download
identity remains private.
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
The component prefix is flushed before its checkpoint is atomically replaced. Unix also flushes
the containing directory; Windows uses file flushing and atomic rename without attempting Unix
directory fsync. Lost checkpoint or snapshot entries after interruption cause verified recovery or
reacquisition, never acceptance of incomplete bytes.

## Product projection

Catalog and acquisition contribute catalog configurations, current package observations, and
process-local download state to the [local-model product projection](./local-model-product-projection.md).
They do not persist provider offerings or serving configurations.

Every independently servable, unambiguous external Hugging Face artifact contributes a discovery
row under its canonical `hf:` `ModelId`. Installed catalog targets are attributed to their complete
catalog `ModelId` by the exact mapping defined in
[Intrinsic catalog target mapping](./intrinsic-target-mapping.md). One catalog product row retains
curated presentation across target, repository, and drafter changes.

Model installation addresses the canonical catalog `ModelId`; ICN parses its base and variant
components, compares desired package IDs with current filesystem presence and exact package
affiliations, acquires missing desired packages, and removes affiliated superseded packages after
the desired bundle is complete. The response exposes only the catalog installation occurrence and
model-level progress/state.

Superseded cleanup is an invariant of the authoritative installed set, not a task owned by the
request that admitted a download. After an installed-set change is committed, one coalesced catalog
maintenance pass removes exact affiliated superseded package IDs only for catalog models whose
entire desired package set is present. Cleanup failure is logged and may be retried after a later
authoritative change. There is no request poller and no package-attempt completion trigger.

Download state determines neither configuration, offering, slot identity, nor physical presence.
ICN publishes the completed package into its materialized inventory before emitting the download's
ready outcome. The filesystem observation remains authoritative.

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

Catalog removal is serialized with installation admission. An installation that is still
downloading is rejected; one that is optimizing is stopped and removal proceeds.
Externally owned or shared dependencies are retained while a removable target and other exclusively
owned material are removed; removal is retained as a whole only when a target itself is external or
shared, because deleting it would either violate ownership or break another catalog model. Removing
an eligible package removes every Magnitude-owned inventory occurrence of that package and preserves
every external Hugging Face cache occurrence.

Deleting artifacts does not delete slot selection, favorites, or recency. Those are user intent and
may remain unresolved until the exact catalog configuration and required files are available again.

## Concurrency and recovery

Operations that publish, replace, or remove files in the same managed repository serialize. Reads
may share in-flight hashing and inspection. Package download sharing and cancellation exist only
within the owning ICN process.

Startup derives inventory from current files and callable models from the catalog plus unambiguous,
successfully inspected external Hugging Face artifacts.
There is no model-state recovery epoch. Later explicit filesystem observations and product-owned
artifact mutations update the same materialized derivation.

## Conformance

- The same valid filesystem produces the same package inventory independent of upgrade history.
- Removing every derived cache changes cost only, never package presence or user intent.
- Catalog configurations are not copied into durable model state.
- Issued catalog configurations remain resolvable after deprecation.
- Every catalog variant publishes the valid ISO calendar date inherited from its model declaration.
- Every active catalog model declares exactly one finite, non-negative Artificial Analysis score with
  valid direct or estimated provenance, the frontier's methodology version, and a score no greater
  than the frontier's.
- Published intelligence is exactly the model's Artificial Analysis score as a percentage of the
  frontier, rounded half-up to a whole number, in the range 0 to 100.
- Model intelligence and artifact-variant fidelity remain separate catalog authorities.
- Every catalog model declares one support level; a deprecation's required replacement is an exact
  configuration of a supported catalog model.
- A deprecated model is never recommended, offered, installed, or loaded, keeps its locked
  revisions, and its installed files remain removable.
- Independently callable external Hugging Face packages without catalog attribution publish under
  their canonical `hf:` identity; other unattributed packages remain inventory only.
- Installed inventory is derived without network access or hardware assessment.
- Partial, unsafe, invalid, or digest-mismatched content is not installed.
- Complete packages from multiple repository revisions may coexist until exact catalog affiliation
  state identifies a package as superseded and eligible for removal.
- Inventory reconciliation never mutates managed artifacts.
- One artifact failure cannot hide unrelated valid artifacts.
- Download identities and terminal history do not survive ICN restart.
- Optimization never changes an installation's outcome: cancellation, load, removal, failure, or
  restart during it leaves the installed model intact and loadable.
- Valid partial bytes may be reused only with matching integrity evidence.
- Historical download completion never proves current presence.
- Shared blobs are deleted only after current filesystem references prove them unreferenced.
- Catalog failure cannot hide independently servable installed packages.
- External Hugging Face artifacts cannot be installed, updated, or removed through Magnitude-owned
  acquisition operations.
