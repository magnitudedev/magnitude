---
applies_to:
  - inference/catalog/**
  - inference/crates/icn-catalog/**
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/models.rs
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
3. one exact intrinsic model/quality candidate.

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
