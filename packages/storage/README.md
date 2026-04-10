# @magnitudedev/storage

Storage owns persisted file schemas.

## Central principle

Assume storage files on disk may be missing, malformed, stale, or partially corrupted, and recover gracefully.

That means:
- storage owns recovery for persisted data
- callers should not need to guess how to repair on-disk shapes
- once storage returns decoded data, callers should trust it

## Principles

- **Storage owns on-disk shapes.** If a value is persisted in `meta.json` or another storage file, its canonical schema lives in `packages/storage`.
- **Effect Schema is canonical.** Storage defines the persisted contract as an Effect schema and exports the inferred type for callers.
- **Reads are lenient at the schema boundary.** Old, partial, or malformed files can be normalized during schema decode, or recovered with storage-level fallbacks.
- **Callers trust decoded data.** CLI/agent code should consume decoded storage types, not re-validate fields that storage already normalized.

## Pattern

1. Define the raw persisted shape in `src/types/...` with `Schema.Struct(...)`.
2. Keep the decoded schema strict for the shape callers should receive.
3. Use schema transforms when decode-time normalization depends on Effect context or service dependencies.
4. Read files through storage helpers, then decode to the canonical stored type.
5. Keep fallback/normalization in storage schemas and helpers, not in callers.

## For session metadata

`StoredSessionMetaSchema` is the canonical `meta.json` schema.

- It uses `Schema.transformOrFail` and requires the `Version` service during decode.
- Missing `initialVersion` and `lastActiveVersion` fields default from `Version`.
- Nullable fields like `gitBranch`, `firstUserMessage`, and `lastMessage` are normalized to `string | null` in the decode transform.
- Callers receive a fully decoded `StoredSessionMeta` and should use it directly.
