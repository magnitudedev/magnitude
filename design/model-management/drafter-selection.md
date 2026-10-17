---
applies_to:
  - inference/service/models/src/catalog.rs
  - inference/service/models/src/preview.rs
  - inference/service/models/src/custom_models.rs
  - inference/service/server/src/assessment/**
  - packages/acn/src/local-model*
  - packages/acn/src/drafter-references*
  - packages/client-common/src/local-models/drafter-presentation*
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
them for automatic choice; the recommended drafter of a custom model is the first candidate ICN's
order binds. ACN keeps ICN's order, persists attached drafters as declarations, and exposes the
choice. Clients present the list, preselect the first bound candidate that the user did not enter,
and never judge compatibility.

## Matching a model to its catalog parent

Signals in order of strength; any one is enough and later signals break ties:

1. **Exact catalog target:** the same repository, commit, and file. It is the catalog model and uses
   the catalog's drafter.
2. **Lineage:** the Hub's `baseModels`, following up to two hops (GGUF → finetune → base), when the
   repository is known; the GGUF's `general.base_model.N.*` for every file; and Ollama's model name
   when it was pulled as `hf.co/<owner>/<repository>:<quant>`.
3. **Shape fingerprint:** the target's engine-computed `ModelFingerprint` equals a catalog target's.
   Requantizations, finetunes, and abliterations keep every fingerprinted dimension, and drafters
   bind to exactly those dimensions. This covers files with no Hugging Face identity or metadata.

A catalog model's upstream lineage is the Hub's `baseModels` of its target repository at the locked
commit, read once and cached with that commit's snapshot; catalog fingerprints come from the release
header bundle.

The product side matches only on engine-supplied fingerprints and declared lineage; it never parses
headers or file names. "Variant of" shown to users is a display hint from lineage; the engine
verdict alone decides compatibility.

## Candidates

Only these sources are offered automatically:

- **Separate drafters in the model's own repository (any path in it) or folder**: GGUFs whose header declares a
  drafter role ("In this repository", or "In this folder" for local files).
- **The catalog drafter of the matched catalog model** ("Trained for <model>").
- **Built-in MTP heads**, detected from the selected file's tensors, never its name ("Built into this model").

Magnitude never searches Hugging Face for drafters and never offers loose drafter files from other locations. A
user may enter **a different drafter** (a Hugging Face file or a local path); it is checked like every candidate,
cannot be chosen when rejected, is labelled "Untested" when its lineage does not match, and is never chosen
automatically. ACN passes a drafter the user entered to ICN as `owner/repo[@revision]:path`, whether it was typed as
a link, `owner/repo/path.gguf`, or `hf:owner/repo/path.gguf`, and a local path as an absolute path.
**No drafter** is always available. The preselected candidate's label reads "Recommended · ",
followed by its source label.

## Automatic choice

Method order matches the speculation axis of the model radar: DFlash2, DSpark, DFlash, a separate MTP file, then
built-in MTP heads. The first candidate that binds, in this order, is preselected:

1. a separate drafter in the model's own repository or folder, any method, by method order, because it was
   built for those exact weights;
2. the catalog drafter, whatever its method;
3. built-in MTP heads;
4. no drafter.

Ties within one method go to the smaller download (nothing, when its files are already here), then ICN's
enumeration order. Candidates are listed in this same order: by source, then method, then download. A
rejected candidate is listed but never preselected.
Catalog models always use their catalog drafter, including one whose exact target is found in another
app's location: it serves with the catalog drafter as soon as that drafter is on this computer (a
projector it lacks never blocks it), and otherwise offers the drafter-only download.

## Controls

- Catalog models keep the catalog's drafter; it changes only through Update. They never offer a
  drafter change.
- Every custom model can add, change, or turn off its drafter. The choice is a persisted
  declaration keyed by `ModelId` and is delivered to ICN with the local-models configuration. A
  chosen reference is persisted with the method the engine binds it with, so ICN receives it right
  after a restart, offline included; ACN only validates that method against ICN's candidates later.
  Choosing the recommended drafter is saved as automatic, never as a fixed reference, so a later
  catalog drafter still applies. A model whose drafter is turned off is offered no recommended
  drafter.
- Magnitude's downloaded drafters are pruned only when no model still on this computer references
  them: attached through its choice, or one it would use (its catalog drafter, a drafter from its
  own repository, or its recommended drafter even while its drafter is turned off).
  Turned off, the model is a standalone bundle, which the engine serves plain: built-in MTP heads
  draft only through a bundle that declares them, never implicitly.
- A model whose choice is automatic attaches the first of its automatic candidates, in the order
  above, that is on this computer, checked against the files present at every projection: a
  recommended drafter that goes missing gives way at once to the next one here, built-in heads
  included. ICN's candidate enumeration is the single source of that order and of the recommended
  drafter; it is recomputed whenever its inputs change (a file appearing, moving or disappearing in
  any location, a model's folder or repository included; a drafter or catalog package installed
  or removed; a configuration change), never kept until a restart. A found or imported model with
  none of its automatic candidates on this computer has no attached drafter: ACN projects it
  without one, and clients offer the recommended drafter ("Faster with a drafter · Add (size)").
- "In this folder" means the folder of each place this model's own bytes are found, never another
  model's: files sharing a header but not their weights (their content samples differ) are
  different files and different packages. An imported model's "In this folder" drafters are on this
  computer wherever that folder is: ICN observes and watches them with the imported file, so the
  automatic choice attaches them in place even outside every model location, as it does a drafter
  chosen by local path. A drafter the user chose that is
  missing with no download running is an interrupted download that Sync resumes, and that ACN
  resumes by itself when the service starts. A catalog drafter whose current revision is not on
  this computer keeps serving with the earlier revision of that catalog model's drafter (matched by
  catalog model and drafter role, never package ID), and ICN reports `updateAvailable` on the
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
- Found, imported, and added models receive the same recommendation as an equivalent catalog
  quantization, including models without a Hugging Face identity.
- Candidate order and tie-breaks are deterministic for equal inputs.
- Catalog models never expose a drafter change; every custom model does.
- Lineage never overrides a `Rejected` verdict.
- A drafter choice the engine doesn't bind is refused with its reasons as data
  (`DrafterNotUsable`), never with the engine's detail; clients word the reasons.
- A custom model's projection carries its persisted drafter choice, so a drafter the user turned off
  is never offered again as "Faster with a drafter · Add" and Change opens on No drafter.
