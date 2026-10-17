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
Catalog models always use their catalog drafter.

## Controls

- Catalog models keep the catalog's drafter; it changes only through Update. They never offer a
  drafter change.
- Every custom model can add, change, or turn off its drafter. The choice is a persisted
  declaration keyed by `ModelId` and is delivered to ICN with the local-models configuration.
  Turned off, the model is a standalone bundle, which the engine serves plain: built-in MTP heads
  draft only through a bundle that declares them, never implicitly.
- A found or imported model whose choice is automatic has no attached drafter until the
  recommended one is on this computer: ACN projects it without a drafter, and clients offer the
  recommended drafter ("Faster with a drafter · Add (size)"). A drafter the user chose that is
  missing with no download running is an interrupted download that Sync resumes. An attached
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
- When a catalog drafter gets a new revision, every model using it, catalog or custom, shows
  Update.

## Conformance

- No drafter is preselected unless the engine binds it.
- Found, imported, and added models receive the same recommendation as an equivalent catalog
  quantization, including models without a Hugging Face identity.
- Candidate order and tie-breaks are deterministic for equal inputs.
- Catalog models never expose a drafter change; every custom model does.
- Lineage never overrides a `Rejected` verdict.
