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
  - packages/acn/src/local-model*
  - packages/acn-protocol/src/schemas/**
  - packages/ai/src/provider/model.ts
  - packages/client-common/src/local-models/**
  - packages/client-common/src/utils/model-presentation.ts
  - packages/storage/src/types/custom-models*
  - cli/src/commands/inference*.ts
  - web/src/app.tsx
---

# Model catalog and acquisition

This document defines catalog publication, package resolution, model locations, artifact presence,
inventory, download, and deletion behavior. Terms follow
[Model-management terminology](./terminology.md).

## Ownership

ICN owns the recommendable catalog, exact packages and sources, the managed model store, scanning
of model locations, filesystem inventory, Hugging Face access, and process-local download
coordination. ACN persists the user's custom-model choices, resolves location paths, observes ICN's
authorities and produces product projections. Clients initiate native mutations through ACN and do
not infer command authorization or completion from cached projections.

Catalog membership, artifact presence, download activity, package validation, assessment, provider
offering, slot selection, and runtime residency remain separate facts.

Hugging Face search, repository resolution, and candidate inspection are service-owned
observational queries. Search returns GGUF repositories with immutable commit identities, lineage
(`baseModels`, and the catalog model a result is a variant of), and a count of results hidden because
the engine recognizes none of their architectures; the Hub's repository-level architecture is only
a prefilter against the engine's recognized-architecture list. Repositories the catalog uses are
never returned as new models, and drafter repositories (a drafter architecture, such as DFlash,
EAGLE, a Gemma assistant or MTP-only heads, or the Hub's speculative-decoding tags) are neither
results nor counted as hidden: they are offered only in drafter lists. "Variant of" follows the
Hub `baseModels` (two hops); a repository whose lineage reaches no catalog model is a variant of
none, since an architecture and a size don't say which model a file is. Results are ordered with those of unsupported architectures last, then repositories whose
name is exactly the query (ignoring case, separators, a trailing `GGUF` and an uploader's `Org_`
prefix), then by downloads. Resolution returns the requested repository's immutable snapshot and
GGUF file evidence. Candidate inspection groups a repository's or path's GGUFs into versions (shards
together) and assesses each through the engine from headers fetched by range request; it never
reads file names for compatibility or quantization. Headers and their classification are cached by
the file's content identity, so a repository is read once. A header is read by walking its layout
from the bytes in hand and requesting about as many more as the layout still declares (exactly for
fixed-width values, extrapolated within string lists, at most 4 MiB at a time), so a header costs
its own length plus little; a header read stops as soon as the bytes in hand can't be a GGUF
header, and whether the engine accepts one (its tensor types) is decided on the complete bytes.
Quantizations of one model share a tokenizer that is nearly all of a header, as a few large
metadata entries (token list, scores, token types, merges). One header is read whole first: an
already cached one, else the largest file's (a projector, an imatrix or a drafter may carry no
tokenizer; one that carries nothing to share leaves the next largest to be read). Every other file
is read with it as a reference: when the walk reaches an entry whose key, type and element count
match a reference's large entry, the run of large entries starting there is checked in one round of
concurrent requests, windows at the start, middle and end of each entry compared byte for byte; the
leading entries that match are taken from the reference and the file's own bytes continue the read.
Small entries may differ in order, length and value (names, special token ids, quantization keys),
an entry that differs (an MTP drafter's token types) is read from the file, and entries after it are
checked again. Equality outside the windows is assumed, so the windows sit where tokenizers of one
model differ: special tokens first, added tokens last. A cold inspection reads within twice the
distinct header bytes its files need (a split version's further shards only as far as their tensor
directories); at the 2026-10 QA's repositories that is 26 MB for `unsloth/gemma-4-E2B-it-GGUF`,
21 MB for `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` and 15 MB for
`Arki05/Qwen3.5-4B-GGUF-shards`, where each read 0.45 to 1.4 GB before. That a file's header is unreadable is cached like a header, by repository, commit, path and
content identity, so a repeated inspection fetches nothing for it. A local file is read at most
once, in one pass: its header (walked the same way) and the first and last MiB of its tensor data
for its `local:` identity, never a byte twice. What that read establishes (header digest, content
sample, header facts, whether the file is truncated) is cached by the file's path, size and
modification time, and its header goes into the content-addressed header cache, so an unchanged
file is never read again: its identity, classification, lineage and every engine assessment of it
(with any drafter) come from those caches, and the engine reads a header the service already holds
through the header source it is given rather than from the file. A header is every byte through the aligned tensor-data offset, or the whole file when it holds no
tensor data and ends before that offset's padding, as a split's metadata-only first shard may. A file whose header is invalid is a version assessed `Unsupported` with the `Unreadable`
reason and the parser's detail; it never fails the repository's inspection, and an inspection
failure is never reported as a network failure unless the inference service was unreachable. Each version also reports the bytes adding it downloads before any drafter: its
files and the repository's projector, or nothing when it is already on this computer or is a local
file used in place. Each version carries its canonical model ID,
so clients never form identities, and a label: the header's `general.file_type` label, spelled as
the file or folder name spells it when that names the same weight type more precisely (`UD-Q4_K_XL`
or `Q4_K_P` for a `Q4_K_M` header); a name naming another type is ignored, since files are renamed
freely (`renamed-q8.gguf` with a `Q4_K_M` header is `Q4_K_M`). With nothing in the header, the
quantization the name spells (a name such as `q40` is not one), else the file name. The ID is a repository file's `hf:` ID, and for a path the ID of the custom
model listed at that path (a Hugging Face identity when its location gives one), else `local:` with
the file's local digest (see terminology), a copy of a listed model's bytes elsewhere included. A
version whose bytes are already listed also names that model as on this computer, whichever location
lists it: the same local digest, the same `hf:` ID and weights size at the same commit when the
listed model's commit is known (a projector found beside it, as with an Ollama pull, is not part of
the version), the same files, or files of the same sizes whose header is the version's. A Hugging Face version that is exactly a current catalog target (the same repository, commit and
file) names that catalog model whether or not it is installed; adding it installs the catalog model
with the catalog's drafter and never declares a custom model, and an installed one is reported as
already on this computer. A version or file the engine can't run is refused before anything is
declared or downloaded, with the engine's reasons, an architecture without kernels included. For a chosen version and drafter the service
also returns the exact servable bundle to download, with the same package identities a post-download
scan computes; clients never construct bundles. These queries do not change the release catalog,
managed inventory, or configuration, and catalog use does not depend on them.

Catalog attribution across exact artifact or drafter changes follows
[Intrinsic catalog target mapping](./intrinsic-target-mapping.md).

## Release catalog

The release catalog is immutable release data. Human-reviewed declarations identify each artifact
variant by its stable format-qualified variant ID, exact upstream format selector, short variant label, fidelity rank, and
quantization-aware-training fact. The format selector must match exactly one primary GGUF; a variant whose repository
holds other files carrying the same format declares its exact primary path instead. Each model declaration records the calendar date on which the
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
capability. A missing, ambiguous, or incompatible projector fails generation. A projector
never decides a drafter's verdict. An MTP head that embeds every prompt token (built-in heads, or a
separate head file of that kind) cannot enter media rows, so the engine generates each request with
media spans without drafting and drafts every text request; the drafter stays bound and the model
keeps its text speedup. A drafter reading the target's history or features (a Gemma QAT assistant,
DFlash, DSpark) sees media rows through the target and drafts image requests too. A separately
packaged Gemma QAT assistant is admitted only with its matching target and validated composed
capability. A request without drafting never retains or resumes a prefix with a drafting request,
whose drafter history it lacks.
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

Acquisition presents only an explicit `HF_TOKEN`; implicit Hugging Face credentials (the token
file under `HF_HOME`) are never discovered. When the Hub rejects the presented credential itself
(HTTP 401, as for an expired or revoked token, which Hugging Face rejects even on public
repositories), the package is resolved and downloaded anonymously instead. A rejected credential
therefore never makes public packages unavailable; content that still requires authorization fails
with its anonymous classification. Header reads for inspection that the Hub refuses (401 or 403, a
gated or private repository) fail with the model operation code `hugging_face_access_required`,
whose message asks for accepting the repository's terms and setting `HF_TOKEN` (or, with a token,
for granting that account access); clients present it as such, never as a connection failure.
Repository metadata the Hub refuses (401 or 404) fails with the model operation code
`hugging_face_repository_not_found`: the Hub answers an anonymous request for a repository that
doesn't exist with 401, so the repository may be missing, private or gated. Clients present it as
a typed not-found error whose message asks to check the name or set `HF_TOKEN`. A gated repository
shows its metadata, so only its refused files get `hugging_face_access_required`.

## Artifact stores and inventory

Completed model files are the sole authority for physical presence. A package is installed when all
files required by that package are currently present and valid. A bundle is installed when all of
its required packages are installed. No registry, configuration, download record, cache entry, or
historical success can make files present or absent.

Magnitude-owned artifacts live in the configured `ManagedModelStore`: catalog models and added
custom models. Every other location is external and read-only (see Model locations). Origin
affects ownership-sensitive operations, not package identity or runtime eligibility.

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
unclassified node.

Temporary and `.incomplete` files never contribute presence. A partial multi-file package is not
installed. A complete independently servable weights package may remain installed while an
optional companion or a larger bundle is incomplete.

Inventory reconciliation owns location scanning, managed-store hashing, bounded GGUF parsing,
tensor-storage derivation, package construction, package validation, and local Assessment Material derivation.
Package construction produces the exact package and its one authoritative structural validation
atomically; catalog generation and installed inventory use that same operation. One cached GGUF
parse supplies every structural property for an immutable component. Template-derived capabilities
are produced only by assessment. Reconciliation publishes complete inventory snapshots and retains the
previous complete inventory snapshot while a refresh is in flight. Queries return the current materialized snapshot
and perform no filesystem work. Scanning is hardware-independent and performs no network access,
assessment, calibration, profile choice, or ranking. ICN reconciles once at startup. Product-owned
artifact mutations publish their exact inventory change before completing and do not trigger a
global scan. Later reconciliation follows location watches, configuration changes, or an explicit
refresh; it is never driven by an unconditional timer.

All inventory indexes, content hashes, GGUF parses, validations, and derived package evidence are optimistic
caches. Missing, stale, malformed, unreadable, or unwritable cache state causes scoped recomputation
and cannot remove a valid artifact from inventory. Catalog failure cannot hide independently
servable artifacts.

### Header cache

The GGUF headers fetched by range request for inspection and previews are kept in a header cache
(`blobs/gguf-headers` under the model cache), by the sha256 of the header bytes. A header is stored
as a recipe of segments: every metadata entry of 64 KiB or more (a tokenizer's lists) is a chunk
file named by its own sha256 (`blobs/gguf-header-chunks`), stored once however many headers hold
it, and the rest of the header is inline in the recipe; the headers of one model's quantizations
share their tokenizer chunks, across repositories too. The cache is bounded to 512 MiB of recipes
and the chunks they reference. Reading a header marks it used, and once the cache has grown by an
eighth of its capacity since it last checked (and on its first write in a process) it evicts the
least recently used headers until it fits, then removes chunks no remaining header references.
Eviction never removes a header the release catalog lists, a header of a file of an installed
Hugging Face package, or a header used in the last ten minutes (an inspection or install in
flight). Eviction is always safe: every read assembles the header and verifies its sha256, so an
evicted, partially removed or damaged header, and any entry not in the recipe format, is a miss
that its reader fetches again by range request; the classification and assessment indexes keyed by
header digest stay valid, so an evicted header costs a fetch only when its bytes are needed again.

### Model locations

ICN scans a list of model roots that ACN resolves and supplies (see
[ICN lifecycle](../icn/lifecycle.md)):

| Location | Layout and identity |
|---|---|
| Hugging Face cache | Hub cache layout; repository and selector give the `hf:` ID. A snapshot entry links to `<repository>/blobs/<hash>`, whose name is the file's Hub content hash; classic caches keep the bytes there, while current `huggingface_hub` makes that entry a link into a store every repository of the hub shares (`<hub>/blobs/<xx>/<hash>`, named by another hash). Both layouts are read, following link chains only within the hub (a link's `..` steps are applied to its canonical directory, since Windows doesn't resolve them in canonical verbatim paths); the repository blob's name is the content identity, and the cache stays read-only. Without symlink support (Windows without Developer Mode, or `HF_HUB_DISABLE_SYMLINKS`) `huggingface_hub` keeps plain copies in the snapshots and no blobs; those files are found like any other, and one at a catalog file's repository, commit and path with that file's size and header digest takes the catalog file's Hub SHA-256 |
| LM Studio | Publisher/repository layout gives the `hf:` ID |
| llama.cpp cache | Current llama.cpp writes the Hugging Face hub layout (in the location itself or its `hub` folder): repository and commit, like the Hugging Face cache, but owned by llama.cpp (without `LLAMA_CACHE`, llama.cpp downloads into the shared Hugging Face cache, and those files stay the Hugging Face cache's). The older flat layout: `manifest=<owner>=<repository>=<tag>.json` names each flat `<owner>_<repository>_<file>` and its projector; a name without a manifest parses only when its repository segment ends in `GGUF`; others are `local:` |
| Ollama | Manifests name each model and its GGUF blob; both `manifests/` and newer releases' `manifests-v2/` (whose entries may be symlinks to the manifest stored as a blob, followed only within the root, and which name Ollama's registry `ollama.com` rather than `registry.ollama.ai`; since Ollama 0.40 an entry is a manifest list whose `gguf` entry names the model's manifest blob by digest) are read, and a model in both is one model, named as Ollama names it (`qwen3.5:0.8b`). A GGUF blob's `sha256-<hex>` name is its content identity; `hf.co/…` pulls take the `hf:` ID of the repository file with the same SHA-256 (resolved once from the repository's file list and cached, matches and misses alike), so the same bytes have one ID whichever locations are enabled; until resolved, and for library models, the ID is `local:` |
| User folders | Any GGUFs beneath the folder; `local:` |

Each automatic location resolves to every directory its tool may use on this host, most
authoritative first: its environment overrides and the owning tool's own configuration, then each
platform default, because one machine often has several (a user's `ollama serve` beside a Linux
system service under `/usr/share/ollama` or `/var/lib/ollama`, `HF_HUB_CACHE` beside the default
Hugging Face cache, LM Studio's current home beside its legacy `~/.cache/lm-studio`). Candidates
that are the same directory, by spelling or through a symlink, are one directory under the first
spelling. Every directory is a root ICN scans, and each remains configured while absent, so one
created later is observed. Each directory reports its own state: found, not found, or can't read
(it exists, but the directory or a folder above it denies Magnitude permission, such as a system
Ollama service's folder owned by the `ollama` user). The location is found when any directory is
found, else can't read when any is unreadable, else not found. The user may turn off an automatic
location, which turns off all of its directories, and add or remove folders, each one directory;
the change reaches ICN as a new configuration and reconciles only the affected roots. Imported files
are scanned at their declared paths, together with the drafter files in the imported file's own folder
(never a subfolder's) and any drafter attached by local path, all referenced in place and watched with
the directories that hold them. An imported file is one more location of its bytes: it names
the model (its `local:` ID and the Imported origin) only while no scanned location holds the same
bytes; otherwise the model keeps the identity and origin its found locations give it, and the import
keeps it available when those files disappear. ACN reports such an import as already on this
computer, records the path under the file's own `local:` ID, and leaves the model's drafter choice
unchanged unless the import names a drafter or turns it off. ICN watches every enabled root, so models appear and disappear
without a restart; a watch event triggers scoped reconciliation of that root, never a global scan.

Magnitude never writes to, moves, adopts, or deletes from an external location. Scans read GGUF
headers only (and, for identity, the first and last 1 MiB of each file's tensor data), follow no
link out of a root, and skip partial downloads and temporary files. External models are identified
by their local digest and never hashed in full; a package file whose SHA-256
is not known from a blob name takes its identity from that digest and its size, unless it is a
catalog file's exact copy in a Hugging Face cache (same repository, commit, path, size and header
digest), which takes the catalog file's SHA-256 and so is that catalog file. A location that
cannot be read is skipped for that reconciliation and never hides another location's models.

Each complete independently callable GGUF group is a custom-model candidate. A standalone file is its
own selector; a complete shard set uses its lexical first shard. Projectors and drafters are
classified from their tensors, never from file names, and belong to the selected package rather
than receiving independent model identities. An importance matrix (`general.type = imatrix`) is
quantization data, never a candidate version. Metadata-only directories, Safetensors-only
repositories, and incomplete shard sets produce no model. The `hf:` and `local:` forms follow
[terminology](./terminology.md); quantization presentation comes from bounded GGUF parsing rather
than filename parsing.

The same model found in several places is one model. When several cached revisions or roots provide
the same identity, exact identical packages collapse. Candidates referenced by a cached `main` ref
outrank other candidates. Within the same ref priority, the most recently modified cached revision
wins; its immutable commit provides the deterministic final tie-break. This is local cache recency,
not a claim about upstream commit chronology. Selection precedes validation, so an invalid selected
revision never falls back to a stale revision. A found file that is exactly a catalog target (the
same repository, commit, and file) is that catalog model and publishes only under its catalog
identity; failed attribution remains visible as a custom model so the installed artifact does not
disappear from inventory. A found file that matches an added model's declaration is that added
model; its files remain externally owned.

A complete GGUF group whose header cannot be read, or whose file is shorter than the tensor data its
header declares, is published as an unsupported local model with the `Unreadable` reason and its
location; a local candidate inspected before adding gets the same verdict; a symlinked file is
read through its link and named by the path the user sees, and an imported file is served through
its declared path wherever its link points; it receives no callable identity and is never offered.
Every readable group is published as a custom model and assessed, so the engine decides support:
ACN lists a found or imported custom model whose assessment is `Unsupported` among the unsupported
local models with the assessment's reasons, never as an offering; an added one keeps its row so it
stays removable (see
[local-model product projection](./local-model-product-projection.md)). Hidden models, named by ID in the
local-models configuration, remain in inventory and are published separately as hidden, so a client
can unhide them; they are neither assessed nor offered.

Inventory retains source occurrences by `InventoryEntryId` until this selection is complete.
Package-content deduplication therefore never erases the repository, selector, root, revision, or
current-ref evidence needed to form and select distinct identities. Catalog package presence
may collapse identical package content afterward, while ownership still reflects every retained
physical occurrence.

## Downloads

A catalog installation command carries the complete catalog-form `ModelId` and converges that
model to present-and-current. ICN admits one `CatalogInstallationOperation` or reports that the
model is already current. The admitted occurrence retains the exact resolved private bundle and
uses process-local download/package work internally; none of those internal identities crosses the
ICN–ACN boundary.

An added custom model is a Magnitude-managed download of a non-catalog `ServableModelBundle` built
from its declaration: the target selector (shards included), the projector, and the drafter, each
pinned to the repository commit resolved when it was added. ICN admits one model-addressed download
for it, and a separate drafter-only download when a drafter is added to any custom model whose
target is already present; a target already on this computer is never downloaded again. Each
such download names the model it serves (`forModel`), and ICN echoes it in the download's state, so
ACN attributes progress and failure to that model without matching package files: a download that
carries an added model's declared target acquires that target, any other fetches its drafter alone,
whatever repository (the catalog's included) the drafter comes from. A catalog installation names its
catalog model the same way. These
downloads share the catalog path's locking, resume, LFS SHA-256 verification, disk-space checks,
failure classification, publication, acknowledgement and post-download optimization.

Update for an added model means its repository has a newer commit where its target is still present
with other content; it never follows a mutable revision implicitly. ICN checks each added declaration
in the background: it resolves the repository's `main` with the `HF_TOKEN` credential when one is set,
compares the target files' LFS content at `main` and at the pinned commit, and reports the newer commit
on the custom model (`updateCommit`) only when a target file differs and none is missing. A result is
kept for about six hours; a check that cannot reach the Hub keeps the last result and reports nothing.
A newer commit that changes only other files, or no longer has the target, is no update. ACN presents
the model as having an update. Like a catalog update, the model keeps serving its pinned copy while the
update downloads, and reads as updating with progress. Sync re-resolves the same version (its model
ID names the target) at that commit, with its shards and projector and a drafter from the same
repository moved to that commit when the commit still has it, downloads that bundle and records the
commit as the declaration's `pendingCommit`; the commit ICN receives stays the pinned one. When the
download completes, ACN pins the newer commit, target, projector and moved drafter and clears
`pendingCommit`; a cancelled or failed update only clears it, so the model stays on its pinned commit.
After a restart, a pending update whose download is gone is fetched again, keeping its bytes, and one
that completed meanwhile is pinned. Once the copy at the pinned commit is installed and valid and the
Hub has been reached since the service started, ICN removes Magnitude's copies of the target at other
commits, except one at the newer commit the Hub reports, which is an update waiting to be pinned.

Admission validates each package in the requested bundle against the current materialized package
record and that package's current exact file evidence. It does not await inventory-wide filesystem
reconciliation. Location reconciliation remains the observation path for external filesystem
changes; it is never a prerequisite for acknowledging a known download.

Development tooling acquires one catalog model outside ICN through the catalog tool, into an explicit model store. It uses the same per-variant resolution at locked commits and the same download and verification path as the service:
- it fetches only missing or invalid files;
- it reports the resolved target, the catalog's drafter (embedded or separate) and the projector, each with its hash.

It creates no installation occurrence and never touches a product store implicitly.

The catalog installation occurrence retains `ModelId`, aggregates bounded progress, and has one
terminal outcome. ACN observes and addresses that occurrence by `CatalogInstallationOperationId`.
Equivalent operations may share private package work, but sharing never changes their public
model identity or operation state.
Caller interruption detaches that waiter without abandoning admitted work. Cancellation stops
shared package work only when no other live occurrence depends on it. Package work whose
cancellation was requested is never joined: a later occurrence admits new work, which waits for the
cancelled work to release the repository and resumes from its checkpoint. A user's cancellation
discards the partial files and checkpoints of the package work it stops, under the repository lock,
keeping any complete blob. An occurrence fails when any of its packages fails; its other package
work then stops under the same rule as cancellation but keeps its partial files, so a retry resumes
and never joins work that belongs to the failed occurrence. A retry creates a new
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
unavailable network, local storage failure, and corrupt content. A source that delivers nothing for
a bounded interval is unavailable network, never an indefinite wait: the attempt keeps its
checkpoint, bounded retries resume from it, and exhausting them fails the occurrence as resumable. Cancellation is a separate
terminal result. Structured facts, including required and available byte counts, cross boundaries
without parsing diagnostic prose. Packages admitted together, such as a target and its separate draft,
check disk space against their combined outstanding bytes plus the reserve, so each reports the same
shortfall and freeing it is enough for the whole download.

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

Every independently servable, unambiguous custom model contributes one row under its canonical
`hf:` or `local:` `ModelId` with its origin. Installed catalog targets are attributed to their complete
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
A managed entry whose blob links into a store shared beyond its repository is refused as unsafe
rather than deleted.
Bytes reclaimed are measured from blobs actually removed. Inventory cannot recreate deleted links
from catalog declarations or retained blobs.

If safe containment or complete reference accounting cannot be proven, deletion fails without
guessing. Interrupted deletion is represented by the files that remain and may be retried
idempotently. Conservative garbage collection may remove blobs proven unreferenced. Runtime
ownership rejects or waits for removal of files used by a live model instance.

Catalog removal is serialized with installation admission. An installation that is still
downloading is rejected; one that is optimizing is stopped and removal proceeds.
Removal deletes Magnitude's own copies and never another app's files. Externally owned or shared
dependencies are retained while a removable target and other exclusively owned material are
removed; a dependency is shared only with another catalog model that is installed or affiliated with
it, never merely declared by an uninstalled one. Removal is retained as a whole only when no target
has a Magnitude-owned copy (the model is a found copy, which is hidden instead) or a Magnitude-owned
target is shared, because deleting it would break another catalog model. Removing an eligible
package removes every Magnitude-owned inventory occurrence of that package and preserves every
external occurrence; when an external copy of the target remains, as after upgrading with the same
file in a Hugging Face cache, it keeps serving the catalog model from where it is. A hidden found
copy is identified by the `hf:` ID it has as a found model, so hiding it ends its attribution to
the catalog model, even when it is exactly a catalog package the catalog remembers: no catalog model
counts it as installed, and it is listed as a hidden found model. Hiding changes only what is
shown: installing the catalog model takes the hidden copy's verified bytes rather than downloading
them, and leaves the copy's files where they are.

A package file whose exact bytes are already in an external location (a found or hidden exact copy,
as when updating a catalog model whose target is in a Hugging Face cache) is never downloaded:
Magnitude's blob is its own file made from that one and verified in full like a download, so only
files present nowhere else are fetched. The blob is never a hard link or any other link to the
external file, which is only ever read. Where the file system supports copy-on-write clones (APFS,
btrfs, XFS), the blob is a clone sharing the external file's extents until either changes, so the
bytes stay on disk once; elsewhere, or across volumes, it is a streamed copy hashed as it is
written. Either way the blob is exclusively Magnitude's: removing it reports its size as freed and
never changes the external file.

Removing an added custom model deletes its Magnitude-owned files under the same rules; any of its
files that are also present in an external location stay there, and the model remains as found.
Removing an imported model forgets its declaration and never touches the file. A found model is
hidden, never deleted; so is a catalog model whose files are all in external locations, which is a
found copy. After a model is removed or stops using its drafter, and after catalog removal, the
Magnitude-owned drafter packages that no installed catalog model and no custom model uses are
removed; while any custom model's automatic drafter is not yet worked out, catalog drafters are kept.
Catalog removal itself keeps the catalog model's drafter while a custom model uses it (or may, while
an automatic drafter is being worked out) and removes the target; pruning removes the drafter once no
model uses it. An added model whose target is still downloading uses the drafter it downloads with
it, so a drafter that finished first is never pruned before its target lands.

Removing a catalog model that is not installed changes nothing: it is reported as not installed,
never as a failed removal, and a new installation supersedes any earlier removal failure. A removal
takes one exclusive residency admission covering every package it deletes (target and drafter
together); it waits behind a load, a release or another removal for at most a minute, then is refused
with a retryable error, so a removal request never hangs.

Deleting artifacts does not delete slot selection, favorites, or recency. Those are user intent and
may remain unresolved until the exact catalog configuration and required files are available again.

## Concurrency and recovery

Operations that publish, replace, or remove files in the same managed repository serialize. Reads
may share in-flight hashing and inspection. Package download sharing and cancellation exist only
within the owning ICN process.

Startup derives inventory from current files and callable models from the catalog plus unambiguous,
successfully inspected custom models in the configured locations and declarations.
There is no model-state recovery epoch. Later filesystem observations, configuration changes, and
product-owned artifact mutations update the same materialized derivation.

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
- Independently callable packages without catalog attribution publish once as custom models under
  their canonical `hf:` or `local:` identity; other unattributed packages remain inventory only.
- The same bytes in two locations are one model; external files are never hashed in full.
- Every complete GGUF group the engine cannot run is never callable; an unreadable, found, or
  imported one appears as an unsupported local model with all its reasons.
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
- Magnitude never writes to or deletes from an external location; found and imported files cannot be
  installed, updated, or deleted through Magnitude-owned acquisition operations.
- Added models are pinned to a repository commit and are downloaded and verified like catalog
  models.
