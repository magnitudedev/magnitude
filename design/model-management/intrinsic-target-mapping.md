---
applies_to:
  - inference/catalog/**
  - inference/service/catalog-tool/**
  - inference/service/models/**
  - inference/service/contracts/src/models.rs
---

# Intrinsic catalog target mapping

This document defines exact private attribution of installed packages to catalog `ModelId` values.

A catalog `ModelId` is composed from `CatalogBaseId` and `CatalogVariantId`. Repository, revision,
filename, package identity, serving profile, speculative method, drafter, and dependencies are
release facts and do not contribute to callable identity.

Catalog generation derives exact reverse indexes from current target package identity and intrinsic
model/quality evidence to the complete catalog `ModelId`. Attribution resolves in order:

1. exact current target package;
2. exact persisted package affiliation; or
3. for Magnitude-managed material, one exact intrinsic model/quality candidate; for external material
   (a Hugging Face cache, or llama.cpp's Hugging Face layout), one current target whose weights are
   exactly the material's: the same repository and commit, and the same weight files by path and
   content. A projector or drafter the external material lacks does not change what its weights
   are.

An external exact target is the catalog model: it is never also a custom model, and the catalog
reports the model installed with external ownership, serving the target alone until the rest of the
bundle is present. Installing the catalog model then fetches only what is missing; a file whose
exact content is already on this computer is copied (cloned where the file system can) and
verified, never downloaded again. External attributions are derived on every reconciliation and are
not persisted as affiliations.

Zero or multiple candidates produce a truthful failure. There is no scoring, normalization,
filename parsing, fuzzy comparison, closest match, or package-derived public identity. Drafters and
dependencies never decide target attribution.

The managed store retains only non-derivable private affiliations: complete catalog `ModelId`, exact
package identity, source repository, and target/dependency role. Filesystem observation remains the
presence authority. Affiliations permit exact superseded-package cleanup across catalog releases
without changing the stable callable ID.

Catalog generation proves that every authored base/variant pair composes a valid `ModelId`, every
callable catalog `ModelId` is unique, current target packages are unique, and every current target
round-trips through the exact index. Changing only material or bundle composition preserves the
catalog `ModelId`; changing the authored base or variant track changes it.
