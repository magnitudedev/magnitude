---
applies_to:
  - inference/service/contracts/src/models.rs
  - inference/service/models/**
  - packages/icn/src/models/**
  - packages/icn/src/events/**
  - packages/icn/src/instances/**
  - packages/acn/src/local-model*
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
---

# Local-model product projection

## Native boundaries

ICN exposes two explicit model resources:

- `/api/v1/catalog/**` publishes reviewed catalog variants and model-addressed managed installation
  operations.
- `/api/v1/local-models/**` publishes custom models and unsupported local models found in the
  configured locations and declarations, accepts the whole local-models configuration, and supports
  explicit refresh.

Hugging Face search, candidate inspection, drafter candidates, and non-catalog downloads are
separate ICN resources. There is no combined native management model list and no package, bundle,
or raw-download identity at the ICN–ACN boundary. The OpenAI-compatible `/v1/models` endpoint is
inference advertisement, not management state: it lists only ready models whose current assessment
is `Fits`, so models that do not fit or that the engine cannot run are never advertised.

Each row already carries a valid canonical `ModelId`. Catalog rows exist without local material and
have `NotInstalled` or `Installed` local state. Custom-model rows exist because material was
observed or declared. Observed material has exactly one truthful state, `Ready` or `Unavailable`,
both containing the selected resolved installation. An added model whose target has not landed in
the store is a `NotInstalled` row from its declaration, so ACN projects its download as installation
progress; it cannot be served until the target lands. Downloads do not outlive the service, so ACN
restarts every part a restart interrupted when it starts: the download of every such row (its target
with its drafter), and the drafter of any model whose chosen drafter has not arrived. Each resumes
from its durable checkpoint.

## ACN projection

ACN observes catalog, custom models, catalog-installation operations, downloads, assessments, and
instances separately, then publishes one `LocalModel` union:

- `Catalog` contains catalog membership and managed acquisition state.
- `Custom` carries its `origin` (`Added`, `Imported`, or `Found` with its source), the catalog model
  it is a variant of, its attached and recommended drafters, serving
  state, and, for `Added` only, managed acquisition state. ICN names it: the repository's name when
  the file has a Hub identity (as the repository step does), else the GGUF's `general.name` unless
  that is only a hexadecimal digest (a source commit some converters write there), else the file
  name, each without the quantization and with separators as spaces, and as the variant label its
  header's `general.file_type` label (the weights' type when its tensors contradict it), kept as its
  file name spells it only when that names
  the same weight type more precisely (`UD-Q4_K_XL` for a Q4_K_M header), as a candidate version is
  labelled, so a renamed file is never labelled from its name; else the quantization its file name
  spells; only without either does ACN label it by bit width. ICN also gives its license when known: the Hub
  card's when the repository is known, else the GGUF's `general.license` (its
  `general.license.name` when the license is `other`).

`LocalModelsState` separately lists unsupported local models: display name, origin, location, the
complete list of structured reasons, and developer detail. It combines the groups ICN could not
read with the found and imported custom models whose assessment is `Unsupported`. They are never
`LocalModel` rows, never selectable, and have no actions. A reason case a client does not know still
renders with its detail. An added model whose assessment is `Unsupported` stays a row with that
assessment, so its Magnitude-owned files remain removable.

`LocalModelsState` also lists hidden models apart. ACN sends the hidden model IDs to ICN in the
local-models configuration; ICN publishes those models separately and never assesses them, and ACN
projects them as `Custom` rows in `hidden`, never in `models`, so they are never offered and a client
can unhide them.

Actions follow who owns the files:

| Origin | Remove | Update | Drafter |
|---|---|---|---|
| Catalog | Deletes the files | New catalog revision | The catalog's; changes only through Update |
| Added | Deletes Magnitude-owned files | New repository commit | Add, Change, Turn off |
| Imported | Forgets the declaration; never touches the file | None | Add, Change, Turn off |
| Found | Hide from Magnitude; never touches the files | None | Add, Change, Turn off |
| Catalog, files only in an external location | Deletes only the parts Magnitude stores; with none, hides it like a found model | New catalog revision | The catalog's |

`UpdateAvailable` carries the update's `parts`: `Model` when the new revision's weights differ,
`ImageInput` when only its projector does, and `Drafter` when its separate drafter does. The engine
derives them from the catalog packages a newer revision needs fetched, so an update always names at least
one part, and clients say which (for example in the Update tooltip). An added model's update, a new
repository commit for its target, is always `Model`; a newer revision of its attached drafter is shown on
the drafter, not as `UpdateAvailable`.

A catalog model whose exact target is found externally serves with the catalog components that are on
this computer. Each component it lacks is listed in `missingComponents` (`Projector` or `Drafter`, with
its download bytes), never as `UpdateAvailable`; clients offer each one while the model is `Installed`
("Image input · Download", "Faster with a drafter · Add"), and `DownloadCatalogComponent` fetches only
that component. While its catalog drafter is missing, a target with built-in heads the engine binds,
and that keep it fitting, serves with them; the drafter is still listed as missing. A catalog model that
fits this computer only without its catalog drafter is served
without it and carries `drafterWithheld`; it lists no `Drafter` component, and clients say the drafter
is off because the model fits only without it (see [drafter selection](./drafter-selection.md)).

Adding a drafter to a model whose target is present downloads only the drafter. When a catalog
drafter gets a new revision, every model using it shows Update. Hide applies only to found models
(including a catalog model whose files are only in an external location); ACN refuses to hide an
added or imported model (`managed_model_not_hideable`) and points to Remove. Importing a file
refuses a folder. Importing a file whose bytes are an installed catalog target's (a copy, or a link to
one) is that catalog model: the import reports it as already on this computer and records nothing.

The same projection publishes a small preparation summary derived directly from the current ICN
source and assessment snapshots: local-model scan completion and the custom-model count (catalog
models are excluded from it), plus assessment completion,
settled count, and total count. Assessment totals describe the current target set and may grow when
authoritative scanning contributes additional targets. This summary is the readiness authority for
onboarding and local provider offerings; clients do not recover global readiness by scanning
individual product rows.

Shared fields contain presentation and canonical `modelId`. Catalog rows directly contain their
catalog data, acquisition state, and serving state. A custom row contains one structural state:
`Ready` owns its resolved installation, residency, catalog attribution, and serving state;
`Unavailable` owns its resolved installation and native failure. An unavailable custom model has no
catalog-attribution fact because its selected artifact is not ready.
Presentation may include deduplicated HTTPS source links, but never package or bundle structure.
Where present, serving state is `Assessing`, `Failed`, or `Assessed`. Native catalog and custom-model rows
already carry a complete desired, effective, or unavailable resolution, so ACN never fabricates an
intermediate resolving row. `Failed` represents source or serving unavailability, not a failed
assessment: ICN marks a failed assessment target `Dropped`, and ACN omits that target's row. An
assessed state contains serving metadata plus capabilities and profile evidence from one atomic ICN
assessment result. Native catalog, custom-model, package-validation, and ready-model state do not
carry a parallel capability copy. A fitting catalog assessment alone
contains ranking scores; custom-model and non-fitting states cannot contain them, and a custom model
never inherits its parent's scores. Provider
selectability is derived from these structural facts and is never stored as a second state machine.
An assessed state also carries the optional speculative method as a direct serving fact; ACN does
not reconstruct it from private packages.

Catalog rows retain the desired configuration's total storage bytes as a direct product fact, so
size remains available before assessment or installation without exposing constituent packages.
Catalog acquisition state is model-level: installation bytes, ownership, progress, failure,
update availability, and residency appear only in variants where they are meaningful. A primary
path exists only on a resolved installation; genuinely ambiguous installed target material is an
unresolved installation and does not fabricate one.
Ownership is of the model's own weights: where the selected target's copies are. `Magnitude` is a
copy only in Magnitude's store, `External` weights found only in another location, and `Mixed` both.
A catalog drafter or projector in Magnitude's store beside external weights leaves the installation
`External`; those parts are listed as `storedComponents`. Remove deletes only what Magnitude stores:
for a `Mixed` model its copy of the weights (it stays listed from its external copy), for an
`External` model with stored components only those components (it keeps running from the found
file), and an `External` model with none is hidden instead, as Remove hides any found model, and
says so ("Hid … from Magnitude"). A component Magnitude is still fetching for an `External` model (a
queued or running installation) is Magnitude's own part too: Remove cancels that installation,
deletes any stored components, and never hides the model, which keeps running from the found file.
The removal gate checks for stored or in-flight components, not ownership alone, and Hide applies to
every `External` model.
An installation's optimization projects as `Optimizing`, an installed-family state that carries
residency and optimization progress (stage, completed and total units, and the device once known).
Installing or Updating moves to `Optimizing` and then to `Installed`; cancellation, a load, or a
failed optimization ends it in `Installed`, and removal moves it to `Removing`. If the operation
reports `Optimizing` before the catalog observes the model as installed, ACN keeps projecting
`Installing` at its publishing stage with full bytes, so the model never appears uninstalled in
between. ACN's cancel command applies to an optimizing operation as it does to a download.
Package lists and native occurrence IDs never enter the product. Installation and update failures
retain their typed native variants, including required and available bytes for insufficient disk
space; they are not flattened into diagnostic strings.

ACN publishes assessment results only while both catalog and local-model revisions and the local
hardware cycle still match the request. A superseded result cannot overwrite a newer source
snapshot. Derived projections may be materialized for cost, but source revisions and native state
remain authoritative.

ACN rereads ICN state and reprojects the whole list on every relevant change, on the one event loop
that also answers health and RPC. Those passes stay cheap: validators are compiled once, never per
model, and ICN observed states and the projection detect change by value equality of the decoded
data, not `Schema.equivalence`, which rebuilds parsers for every union member it compares.

## Conformance

- Catalog and custom models share `ModelId` without sharing lifecycle semantics.
- Every callable custom model appears once under its `hf:` or `local:` identity, wherever its files
  are found.
- `Unsupported` assessment outcomes (custom models only) never become offerings; for found and
  imported models they appear only in the unsupported list, with every reason; targets whose
  assessment attempt fails are omitted.
- Hidden models appear only in the hidden list, are never assessed, and are never offered. A catalog
  model whose installation is entirely externally owned is a found copy: hiding it adds the catalog
  model's `aliases` (the `hf:` IDs its external files have as found models) to the hidden IDs, so ICN
  stops attributing that copy to the catalog model and lists it as a hidden found model; unhiding the
  catalog ID or the alias removes them. A catalog model with a Magnitude-owned copy can't be hidden.
- Catalog rows carry `aliases`, except a hidden external copy's ID, which names that hidden found
  model and is never resolved to the catalog model. A request naming an alias (load, sync, remove, status) means that
  catalog model, and ACN rewrites persisted references to an alias (slot selections, recents,
  favorites, and connected harnesses' selections, automatically) to the catalog ID.
- Catalog rows carry their catalog support level unchanged; support and assessment stay separate
  facts in the projection.
- Found and imported models can be selected and loaded but cannot be installed, updated, or deleted;
  only catalog and added models expose managed acquisition.
- Catalog models never offer a drafter change; every custom model can add, change, or turn off its
  drafter.
- Provider offerings contain no fallback profile, capability, ranking, package, or bundle data.
- Clients never correlate package collections or reconstruct model identity.
