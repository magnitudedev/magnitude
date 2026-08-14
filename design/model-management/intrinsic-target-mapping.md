---
applies_to:
  - inference/catalog/**
  - inference/crates/icn-catalog/**
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/models.rs
  - packages/icn/src/catalog/**
  - packages/icn/src/installed/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
---

# Intrinsic catalog target mapping

This document defines exact attribution of installed target packages to catalog products.

## Identity

`CatalogIdentity` is the structured pair of a branded `CatalogModelId` and branded
`CatalogVariantId`. The model ID names the stable model product. The variant ID names the authored
format-qualified quality track, such as `gguf:q4`.

Repository, revision, filename, package identity, serving profile, speculative method, drafter, and
other dependencies are release facts and do not contribute to catalog identity. Only the local
provider boundary serializes a `CatalogIdentity` into a `ProviderModelId`.

## Exact resolution

Catalog generation derives two exact reverse indexes:

```text
current target ModelPackageId -> CatalogIdentity
(target general.name, canonical intrinsic quality) -> CatalogIdentity candidates
```

Attribution resolves in this order:

1. resolve an exact current target package ID;
2. resolve an exact persisted target-package affiliation;
3. resolve the target's exact non-empty intrinsic model and quality pair, requiring exactly one
   catalog identity; or
4. return a typed unresolved, invalid, or ambiguous result.

There is no scoring, normalization, aliasing, filename parsing, fuzzy comparison, probabilistic
choice, or closest candidate. Zero or multiple candidates do not produce an identity. Drafters and
dependencies are never used to decide target attribution.

## Package affiliations

The Magnitude store retains non-derivable catalog package affiliations. Each record contains one
`CatalogIdentity`, exact `ModelPackageId`, source repository, and role (`Target` or `Dependency`).
It contains no presence, path, revision, configuration, assessment, or lifecycle state.

Acquisition records every catalog package it observes. The filesystem remains the sole presence
authority. Invalid affiliation records are discarded independently; the document has no version
gate. Loss of the file falls back to exact current-package or intrinsic resolution.

Affiliations let ICN distinguish packages belonging to the same product across catalog revisions,
compute superseded packages exactly, and remove only packages that were explicitly affiliated with
that product.

## Catalog validity

Catalog generation fails unless:

- every entry has one valid `CatalogIdentity`;
- catalog identities and current target package IDs are unique;
- variants of one model have distinct authored variant IDs; and
- every current target round-trips through the exact package index.

Changing only a drafter, dependency, repository, or exact target package does not change catalog
identity. Different authored quality tracks never coalesce.

## Conformance

- An unchanged target package resolves without inspecting a drafter.
- A prior affiliated target remains attributable after package or repository replacement.
- Intrinsic resolution succeeds only for one exact candidate.
- Every installed target remains visible when attribution fails.
- Removing derived caches changes mapping cost only, not its result.
