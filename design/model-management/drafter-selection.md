---
applies_to:
  - inference/service/models/src/catalog.rs
  - inference/service/models/src/preview.rs
  - inference/service/models/src/custom_models.rs
  - inference/service/server/src/assessment/**
  - packages/acn/src/local-model*
  - packages/acn/src/serving-usage*
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
candidates for a target (a candidate or a model ID), and assesses each candidate's binding. ACN
orders candidates by measured acceptance, persists attached drafters as declarations, and exposes
the choice. Clients present the list and never judge compatibility.

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

Candidates are listed in this order, and the first that binds is preselected (`auto`):

1. **The catalog drafter of the matched catalog model** ("Recommended · Trained for <model>").
2. **Built-in MTP heads**, detected from the selected file's tensors, never its name ("Built into
   this model").
3. **Drafter files next to the model** in the same repository or folder ("In this repository").
4. **Drafters already on this computer** in any model location whose lineage matches and which bind
   ("On this computer"); no download.
5. **Third-party drafters on Hugging Face** for the matched base, found by `base_model` and the
   `speculative-decoding` tag, labelled by uploader.
6. **A different drafter**: any Hugging Face reference or local file. It is shown with the engine's
   verdict, cannot be chosen when rejected, and is warned as untested when its lineage does not
   match.
7. **No drafter.**

Within one category, a lineage match precedes a fingerprint-only match, then higher measured
acceptance, then the smaller download. A rejected candidate is never preselected or offered as a
choice.

## Controls

- Catalog models keep the catalog's drafter; it changes only through Update. They never offer a
  drafter change.
- Every custom model can add, change, or turn off its drafter. The choice is a persisted
  declaration keyed by `ModelId` and is delivered to ICN with the local-models configuration.
- Adding a drafter to a model whose target is present downloads only the drafter; one already on
  this computer downloads nothing. The download names the model it is for, so the attached drafter
  shows its progress and failure whichever repository, the catalog's included, it comes from. A
  drafter in an external location remains externally owned.
- When a catalog drafter gets a new revision, every model using it, catalog or custom, shows
  Update.

## Measured acceptance

Once a model with a drafter has served requests, ACN records the acceptance rate and decode speed for
that model and drafter pair with [local serving usage](../acn/local-serving-usage.md). Details show
the measured rate. When a drafter makes decoding slower than plain decoding, the model reports that
the drafter is slowing responses and offers Change and Turn off. Ordering uses these measurements
where they exist; assessment never models acceptance.

## Conformance

- No drafter is preselected unless the engine binds it.
- Found, imported, and added models receive the same recommendation as an equivalent catalog
  quantization, including models without a Hugging Face identity.
- Candidate order and tie-breaks are deterministic for equal inputs.
- Catalog models never expose a drafter change; every custom model does.
- Lineage never overrides a `Rejected` verdict.
