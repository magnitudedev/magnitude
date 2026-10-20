---
applies_to:
  - inference/service/models/src/catalog.rs
  - inference/service/models/src/preview.rs
  - inference/service/models/src/custom_models.rs
  - inference/service/models/src/catalog_models.rs
  - inference/service/server/src/assessment/**
  - packages/acn/src/local-model*
  - packages/acn/src/drafter-references*
  - packages/client-common/src/local-models/drafter-presentation*
  - packages/client-common/src/local-models/catalog-components*
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/storage/src/types/custom-models*
---

# Drafter selection

Every model that runs in Magnitude gets the best drafter known to work for it, whatever its source:
the catalog, added from Hugging Face, imported as a file, or found in any model location. A drafter
is configuration of a model, never part of its `ModelId`.

"Known to work" means both:

- the engine binds it to the target (`DrafterVerdict::Bound`, see
  [assessment](./assessment-and-ranking.md)); and
- its lineage matches the target.

## Ownership

ICN computes catalog targets' shape fingerprints from the release header bundle, enumerates drafter
candidates for a target (a candidate or a model ID), assesses each candidate's binding, and orders
them for automatic choice, marking each candidate automatic choice may take (`automatic`); the
recommended drafter of a custom model is the first automatic candidate in ICN's order. ACN keeps ICN's
order, persists attached drafters as declarations, and exposes the choice. Clients present the list,
preselect the first automatic candidate, and never judge compatibility or fit.

## Matching a model to its catalog parent

Signals in order of strength; any one is enough and later signals break ties:

1. **Exact catalog target:** the same repository, commit, and file. It is the catalog model and uses
   the catalog's drafter.
2. **Lineage:** the Hub's `baseModels`, following up to two hops (GGUF → finetune → base), when the
   repository is known; the GGUF's `general.base_model.N.*` for every file; and Ollama's model name
   when it was pulled as `hf.co/<owner>/<repository>:<quant>`.
3. **Shape fingerprint:** the catalog target whose engine-computed `ModelFingerprint` equals the
   target's, else the catalog model the target is a variant of by shape (below). Requantizations,
   finetunes, and abliterations keep every fingerprinted dimension, and converters may rewrite a
   token table's unused entries (Ollama writes them as `[PAD<n>]`), so the token table only breaks
   ties. This covers files with no Hugging Face identity or metadata, but it is not lineage:
   successive generations of one architecture and size share every dimension and often the token
   table (Qwen3.6 27B and Qwen3.8 27B do), and a drafter trained on another generation's hidden
   states binds yet slows decoding. A catalog drafter matched only by shape is therefore offered,
   labelled "Untested · Trained for <model>", but never chosen automatically; the user may choose
   it. Header-only evidence cannot tell a same-weights copy without lineage (Ollama's blobs) from
   another generation, so making such a drafter automatic needs evidence beyond the header.

A catalog model's upstream lineage is the Hub's `baseModels` of its target repository at the locked
commit, read once and cached with that commit's snapshot, and those base models' own `baseModels`
(a second hop, as for a model's lineage): a catalog target is often a quantization of a derivative
(Gemma 4 E2B's is of `google/gemma-4-E2B-it-qat-q4_0-unquantized`) whose parent
(`google/gemma-4-E2B-it`) community quantizations name instead. Lineage compares repositories,
never model names. Every Hub lineage answer is persisted, so a model whose repository is known (an
added model, a Hugging Face cache or LM Studio file, an Ollama `hf.co` pull) has its declared lineage
from what is on this computer once it has been read, offline included. Catalog fingerprints come
from the release header bundle.

The product side matches only on engine-supplied fingerprints and declared lineage; it never parses
headers or file names. "Variant of" shown to users (a search result, an inspected candidate, a
custom model) is a display hint decided by shape alone, by one rule: the exact catalog target is
its catalog model, and anything else is a variant of the first catalog model, in catalog order,
whose fingerprint has its shape (every fingerprinted dimension but the token table). Declared
lineage never decides it. Models of one architecture and size can share every dimension and the
tokenizer (Qwen3.5 27B and Qwen3.8 27B do), so the first of them in catalog order is named. Drafter
selection still ranks lineage above the fingerprint, as above, and a model shown as a variant is
always offered that catalog model's drafter, automatic only with lineage. The engine verdict alone
decides compatibility.

## Candidates

Only these sources are offered automatically:

- **Separate drafters in the model's own repository (any path in it) or folder**: GGUFs whose header declares a
  drafter role ("In this repository", or "In this folder" for local files).
- **The catalog drafter of the matched catalog model** ("Trained for <model>"; "Untested · Trained for
  <model>", and never automatic, when matched by shape alone).
- **Built-in MTP heads**, detected from the selected file's tensors, never its name ("Built into this model").

Magnitude never searches Hugging Face for drafters and never offers loose drafter files from other locations. A
user may enter **a different drafter** (a Hugging Face file or a local path); it is checked like every candidate,
cannot be chosen when rejected, is labelled "Untested" when its lineage does not match, and is never chosen
automatically. ACN passes a drafter the user entered to ICN as `owner/repo[@revision]:path`, whether it was typed as
a link, `owner/repo/path.gguf`, or `hf:owner/repo/path.gguf`, and a local path as an absolute path.
A different drafter that is the same file as a listed candidate (the catalog drafter's package, or
the same repository, commit and path) is that candidate: ICN reports the listed reference the entry
resolves to, ACN returns that candidate as `entered` with the list, no second row is added, and
choosing it attaches (and saves) that candidate (the CLI resolves a typed reference to `entered`
first). Once
chosen, any other different drafter stays in the model's list (after the candidates above, as a
different drafter with its verdict) and is marked as the one in use, in every client.
**No drafter** is always available. The preselected candidate's label reads "Recommended · ",
followed by its source label.

## Automatic choice

Method order matches the speculation axis of the model radar: DFlash2, DSpark, DFlash, a separate MTP file, then
built-in MTP heads. The first candidate that binds, in this order, is preselected:

1. a separate drafter in the model's own repository or folder, any method, by method order, because it was
   built for those exact weights;
2. the catalog drafter, whatever its method, when the target matched it by exact target or lineage;
3. built-in MTP heads;
4. no drafter.

Ties within one method go to the smaller download (nothing, when its files are already here), then ICN's
enumeration order. Candidates are listed in this same order: by source, then method, then download. A
rejected candidate is listed but never preselected.

A drafter never makes a model that fits stop fitting. A candidate that binds is automatic only when the model fits
with it; one it doesn't fit with is listed with its verdict, and the next automatic one, or no drafter, is
preselected. "Without a drafter" is the model served plain (`ModelMethod::Plain`), never the engine's automatic
method, which takes built-in heads; the heads are a candidate like any other, judged with them. When the model
doesn't fit even without a drafter, no drafter is automatic: every candidate stays listed and choosable, and the
model is served and assessed plain, so a refusal reports the model's own shortfall rather than a larger one for a
drafter the user can't use (a drafter only adds memory, so it can never make such a model fit). A catalog model
whose bound catalog drafter is therefore not automatic is withheld and served plain too, so its refusal reports
its own shortfall. The rule holds for every origin and every path: the
candidate list, the recommended drafter, and the attachment, which takes only automatic candidates. A user-entered
drafter is never automatic. Inspection composes only the recommended version with its preselected drafter (or
serves it plain when none is preselected), exactly what adding it installs; every other version carries its plain
verdict, whose fit equals the installed one's (an automatic drafter never breaks a fit). A version the user selects
gets its drafter list, with each candidate's composed verdict, from the per-version drafter candidates (the dialog
loads them for the selected version; `catalog inspect owner/repo:QUANT` lists them for the named version).
Composing every version would assess each with every drafter candidate.
Catalog models always use their catalog drafter unless it doesn't fit, including one whose exact target is found in
another app's location: it serves with the catalog drafter as soon as that drafter is on this computer (a
projector it lacks never blocks it), and otherwise offers the drafter-only download, serving meanwhile with its
target's built-in heads when they are automatic: a catalog model's candidates are its catalog drafter, then its
built-in heads, and ICN records per catalog model whether those heads are automatic, alongside whether the catalog
drafter is withheld. A catalog model that fits
served plain and not with its catalog drafter is served plain instead of being refused: ICN works this out once
per catalog model (it depends on the release material and this computer, not on the files here), the catalog
projection then serves and assesses the model's target alone, reports `drafterWithheld`, and offers no drafter
download; clients say the drafter is off and, from the model's plain verdict, why: it fits only without it, or it
doesn't fit this computer at all.

## Controls

- Catalog models keep the catalog's drafter; it changes only through Update, and is off only while
  the model fits only without it. They never offer a drafter change.
- Every custom model can add, change, or turn off its drafter. The choice is a persisted
  declaration keyed by `ModelId` and is delivered to ICN with the local-models configuration. A
  chosen reference is persisted with the method the engine binds it with, so ICN receives it right
  after a restart, offline included; ACN only validates that method against ICN's candidates later.
  Choosing the recommended drafter, or adding a model with the automatic drafter (from any repository,
  a catalog model's own included), is saved as automatic, never as a fixed reference, so a later
  catalog drafter still applies. A model whose drafter is turned off is offered no recommended
  drafter.
- Magnitude's downloaded drafters are pruned only after a model is removed, never on a drafter
  change, and only when no model still on this computer references them: attached through its choice, or one it would use (the drafter of every catalog model it
  matches, even while a drafter beside it is recommended instead, a drafter from its own repository,
  or its recommended drafter even while its drafter is turned off), whatever its origin.
  Turned off, the model is a standalone bundle, which the engine serves plain: built-in MTP heads
  draft only through a bundle that declares them, never implicitly.
- A model whose choice is automatic attaches the first of its automatic candidates, in the order
  above, that is on this computer, checked against the files present at every projection: a
  recommended drafter that goes missing gives way at once to the next one here, built-in heads
  included. ICN therefore hands the projection every automatic candidate, never only the first, and
  while a lesser one serves, clients still offer the recommended one ("Faster with a drafter · Add"). ICN's candidate enumeration is the single source of that order and of the recommended
  drafter; it is recomputed whenever its inputs change (a file appearing, moving or disappearing in
  any location, a model's folder or repository included; a drafter or catalog package installed
  or removed; a configuration change), never kept until a restart. While a model's recompute is
  pending (its target package changed, say, when a projector appeared beside it), its previous answer
  stands, so an attached drafter is never shown as none in the meantime; a model with no answer yet
  is computed before the others are recomputed. A model whose choice is automatic and that has no
  drafter while its automatic drafter isn't worked out yet (right after a restart, or as an added
  model's target lands, until the lookup answers for it) is reported as `drafterPending`, which ICN,
  ACN and every client keep apart from having no drafter: clients show "Finding drafter…", never
  none. A drafter already worked out keeps answering instead, and a turned-off or chosen drafter is
  never pending. What is on this computer decides first and never
  waits for the Hub: a model with no Hub-reaching answer gets one from its own files, the release
  catalog and Hub answers already cached (its shape's "Variant of", drafters in its folder, its
  built-in heads, the catalog drafter it matches by shape or by lineage its header or its
  repository's cached Hub answers declare, automatic on that lineage as on the Hub's), which the list
  shows and a load uses at once, while the Hub's part (snapshots, lineage not yet read, drafter
  headers) is worked out alongside and then replaces it; a local answer never replaces a Hub one,
  and drafter pruning waits for the Hub's. A found or imported model with
  none of its automatic candidates on this computer has no attached drafter: ACN projects it
  without one, and clients offer the recommended drafter ("Faster with a drafter · Add (size)").
- "In this folder" means the folder of each place this model's own bytes are found, never another
  model's: files sharing a header but not their weights (their content samples differ) are
  different files and different packages. An imported model's "In this folder" drafters are on this
  computer wherever that folder is: ICN observes and watches them with the imported file, so the
  automatic choice attaches them in place even outside every model location, as it does a drafter
  chosen by local path. A drafter the user chose that is
  missing with no download running is fetched again by Sync, keeping its bytes; a drafter download
  a restart interrupted is resumed by ICN as a requested download, and a cancelled one is not. A
  catalog drafter whose current revision is not on this computer keeps serving with the earlier
  revision of that catalog model's drafter (matched by catalog model and drafter role, never package ID), and ICN reports `updateAvailable` on the
  attached drafter; ACN uses that flag as-is, whether or not the catalog model itself is installed.
  A catalog model's update of other parts (its target or projector) is not its drafter's. An attached
  drafter's verdict is its own: the assessment's when it names the attached method, else ICN's
  verdict for the matching recommended candidate, never another drafter's.
- Adding a drafter to a model whose target is present downloads only the drafter; one already on
  this computer downloads nothing. A drafter is on this computer when any location, another app's
  included, holds its bytes: a file with its header digest and size, or its Hub SHA-256 when known.
  ICN learns a drafter reference's bytes when it reads the drafter's header (listing candidates, or
  enriching a model whose choice names it), reports such a candidate as on this computer, and the
  projection attaches that file as-is, never copying it; when it disappears, the drafter is missing
  again and can be downloaded. Automatic sources are unchanged: this reuses bytes of the drafters
  listed above and never discovers new drafters. The download names the model it is for, so the attached drafter
  shows its progress and failure whichever repository, the catalog's included, it comes from. A
  drafter in an external location remains externally owned.
- When a catalog drafter gets a new revision, every model using it shows Update, catalog or not,
  and whether or not its catalog model is installed.

## Conformance

- No drafter is preselected unless the engine binds it.
- No drafter is preselected or attached automatically that makes a model fitting served plain not
  fit, whatever the model's origin; a catalog model in that case is served plain.
- A candidate version's verdict equals the installed model's verdict for the same composition.
- Found, imported, and added models receive the same recommendation as an equivalent catalog
  quantization, including models without a Hugging Face identity.
- Candidate order and tie-breaks are deterministic for equal inputs.
- Catalog models never expose a drafter change; every custom model does.
- Lineage never overrides a `Rejected` verdict.
- A drafter choice the engine doesn't bind is refused with its reasons as data
  (`DrafterNotUsable`), never with the engine's detail; clients word the reasons.
- A custom model's projection carries its persisted drafter choice, so a drafter the user turned off
  is never offered again as "Faster with a drafter · Add" and Change opens on No drafter.
