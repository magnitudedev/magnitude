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

A package an added custom model declares (the same repository, commit and target files) that is not
an exact current target is never attributed after step 1: a version added from a catalog model's
repository is a custom model even when its intrinsic identity is the catalog target's, and an
affiliation recorded for it earlier is forgotten. Packages are identified by their files' contents, so
a version whose weights are a catalog target's bytes at another commit of its repository is that
target's package (step 1): inspection reports it as the catalog model, and adding it installs the
catalog model rather than declaring a custom one that could never be installed.

An external exact target is the catalog model: it is never also a custom model, and the catalog
reports the model installed with external ownership. It serves with every catalog component that is
on this computer: the catalog's separate drafter as soon as that drafter is present, and without a
projector it lacks, which never blocks the drafter. What it lacks is reported as missing components
(`Projector`, `Drafter`, each with the bytes it would download), not as an update: update means a
newer catalog revision, and is reported only when an older revision of the model's packages is also
present. A drafter that is absent with no earlier revision here is a missing component whatever the
target's ownership.

The found weights are never copied into Magnitude's store. Acquiring a catalog target whose exact
weights are found (and not hidden) acquires only its projector package: a package of the target's
one projector file, which is the one package without weights. The inventory then lists the catalog's
target package composed of the found weights and that projector package, installed where the weights
are, and resolves it for serving to the found weights plus the stored projector. So fetching the
projector (`install_catalog_component`) downloads and stores only the projector, and installing the
catalog model fetches only the missing projector and drafter. While the drafter is missing, a target
with built-in heads that the engine binds and that keep the model fitting serves with them. Removal
deletes only Magnitude's projector package and drafter and leaves the found file, keeping a drafter
only while another model serves with it and no other location holds its bytes; the removal completes
and reports what it kept, even when nothing is left to delete; a
model with nothing of Magnitude's is retained as externally owned and Remove hides it instead. A projector or drafter that other variants of
the same repository also list is kept only for another catalog model that is installed. When the found weights disappear, the composed
target goes with them and the model is not installed; its full download then fetches the target and
reuses the stored projector. A hidden copy composes nothing: installing the catalog model then makes
Magnitude's own copy of its bytes. External attributions are derived on every reconciliation and are
not persisted as affiliations; the projector package is affiliated as a dependency.

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
