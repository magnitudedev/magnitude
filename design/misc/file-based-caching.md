---
applies_to:
  - packages/storage/src/config/**
  - packages/storage/src/io/structured-file.ts
  - packages/storage/src/sessions/storage.ts
  - packages/acn/src/model-configuration.ts
  - packages/icn/src/catalog/**
  - packages/ai/src/provider/file-catalog.ts
  - inference/crates/icn-models/**
  - inference/crates/icn-hardware/**
  - inference/crates/icn-utils/**
---

# File-based cache, index, and configuration recovery

File-backed derived state is an optimization, never an availability dependency. A cache path is an
optimistic opportunity to reuse work: missing, unreadable, malformed, stale, partially valid, or
unwritable cache data behaves as a cache miss for only the affected recovery unit. It must not crash
the process, fail the user operation, or discard valid siblings.

Durable configuration uses the same granular decoding discipline. Configuration is not disposable,
so recovery preserves valid user choices and, when a damaged source must be replaced wholesale,
preserves the original bytes before installing defaults. Only catastrophic conditions escape the
configuration boundary as errors.

These guarantees apply equally to Bun/Effect and Rust implementations.

## Data classes and failure boundaries

Every file-backed format must be classified before its persistence behavior is designed:

| Class | Examples | Recovery authority | Cache/file failure visible to caller? |
| --- | --- | --- | --- |
| Recomputable cache | fit assessment, remote header, parsed provider catalog | Recompute from authoritative inputs | Never |
| Recomputable index | local-model inventory, session lookup index | Re-enumerate authoritative files or records | Never |
| Durable configuration | user preferences with defined defaults | Preserve valid values; default damaged values | Only if recovery is catastrophic |
| Irreplaceable record | session events, user content, credentials | Format-specific durability and repair protocol | Yes; this document's defaulting rules do not apply |

A file must not mix irreplaceable records with disposable cache data. If a format contains regions
with different recovery authority, split those regions into separate files or make their independent
recovery units explicit. Calling a file an “index” does not make it disposable: it is recomputable
only when all of its facts can be reconstructed from another authority.

A cache boundary is total from its caller's perspective. Its read operation returns a hit, a partial
hit, or a miss; its write operation is best effort. It has no cache-I/O or cache-decode error outcome.
The work required to regenerate a miss retains its own ordinary errors. For example, a corrupt model
metadata cache cannot fail model discovery by itself, but the remote service needed to regenerate
that metadata may independently be unavailable.

## No cache schema versions

Recomputable caches and indexes must not contain, track, migrate, or gate reads on a file-format
schema version. This prohibits version fields used to select a decoder, versioned cache filenames,
whole-file invalidation because an application release changed, and cache migration code.

Current decoders instead accept the values they understand, preserve independently valid data, and
default or discard invalid data at the smallest safe boundary. Unknown fields are ignored. A format
change therefore costs, at worst, recomputation of affected entries rather than a migration or a
whole-cache reset.

Content identities, algorithm fingerprints, native-build fingerprints, and concrete configuration
inputs are not schema versions. They are domain inputs to cache validity and belong in entry keys or
evidence. An evidence mismatch invalidates only the entries whose computed result may have changed.

Durable configuration should likewise evolve through tolerant decoding, optional fields, and
defaults rather than a global format-version gate. If a historical configuration value has changed
meaning in a way that cannot be inferred from its shape, any required semantic migration must be
local to that value and must not prevent recovery of unrelated configuration.

The canonical model configuration is only a partial map of branded slot IDs to complete explicit
slot selections (`providerId`, `providerModelId`, and `reasoningEffort`). A legacy selected local
profile, raw native model identity, repository/configuration association, or local-slot-intent flag
has no current meaning and is discarded rather than dual-read. An incomplete selection is removed
as one invalid leaf; ACN never guesses its missing identity or reasoning value.

## Granular recovery

The recovery unit is the smallest independently meaningful value:

- a top-level setting defaults without resetting unrelated settings;
- one map entry can be discarded without discarding other keys;
- one array element can be discarded without discarding other elements;
- one cached property can be recomputed without discarding unrelated properties for the same
  artifact, when those properties have independent evidence;
- an enclosing object is discarded only when a required identity, discriminator, or cross-field
  invariant cannot be recovered safely.

A field with a defined safe default is restored to that default. A malformed optional field is
removed. If defaulting a field would create a misleading or internally inconsistent entity, the
smallest enclosing entity with an independent identity is removed instead. References to missing or
discarded entities are themselves removed or recomputed; they must not be left dangling.

Recovery follows this conceptual flow:

```text
read bounded bytes
       |
       +-- missing / inaccessible / wrong file type ----------> miss or config fallback
       |
       v
parse outer representation
       |
       +-- malformed root ------------------------------------> miss or preserved config reset
       |
       v
decode independent sections, entries, and fields
       |
       +-- valid unit -----------------------------------------> retain
       +-- safely defaultable field ---------------------------> default field
       +-- invalid independent unit ---------------------------> discard unit
       |
       v
validate evidence and cross-field invariants
       |
       +-- stale cache unit -----------------------------------> miss for that unit
       v
return surviving value and regenerate only the misses
```

All external bytes are untrusted. Reads must be bounded. Empty files, random bytes, invalid UTF-8,
truncated JSON, a valid document with the wrong root shape, unknown fields, invalid collection
members, duplicate identities, impossible numeric values, broken references, and stale evidence are
normal recovery inputs, not defects.

## Cache behavior

For caches and recomputable indexes, all filesystem conditions are cache misses at the affected
scope. This includes absence, permission denial, a directory at the expected file path, broken or
unsafe links, concurrent replacement, short reads, decode failure, and resource-limit rejection.

Cache recovery must satisfy all of the following:

- valid siblings remain usable when another field, entry, or section is invalid;
- failure to read, delete, repair, create, encode, write, sync, rename, or lock cache state does not
  fail the operation that wanted to use the cache;
- regeneration may proceed without first repairing or deleting the bad file;
- a successful later write may heal the file atomically, but healing is not required for correctness;
- any diagnostics that are emitted are bounded, omit cached payload contents and secrets, and do
  not become user-facing errors; disposable caches are not required to allocate a recovery report;
- repeated callers may coalesce regeneration in memory, but an in-flight map is not durable state.

Readers cannot assume that writers are cooperative or that a lock was honored. Writers encode the
complete replacement before publication and use a uniquely named temporary file plus a
same-directory atomic rename where the platform supports it. Restrictive permissions and best-effort
temporary-file cleanup still apply. Locking may reduce duplicate work, but correctness comes from
tolerant readers and atomic publication, not from the lock.

Cache deletion and garbage collection are always safe. Size and age policies may remove any entry;
the next access regenerates it. Negative or operational results are cached only when they are stable
domain facts with complete validity evidence; transient failures are never persisted as facts.

## Durable configuration behavior

Configuration recovery is intentionally more conservative because the valid portions express user
intent:

- missing configuration uses the complete default;
- invalid fields, map entries, and array elements recover independently wherever the schema defines
  a safe default or removal behavior;
- unknown fields are preserved when rewriting so a temporarily older binary does not erase settings
  it does not understand; fields explicitly tombstoned by a completed cutover are removed instead
  of being preserved as unknown data, and closed owned namespaces such as model selection discard
  fields outside their canonical shape;
- recovery rewrites a normalized file only after the recovered value passes the current complete
  schema and semantic invariants;
- if malformed syntax or an invalid root forces a whole-file reset, the original bytes are preserved
  to a uniquely named recovery copy before the default is published;
- recovery diagnostics identify affected paths without logging sensitive values.

Model selection persistence exposes one addressed update per branded slot. It does not expose a
partial model-configuration patch or a second model-only read API. ACN loads the complete
configuration into one resident source; each update durably writes the addressed slot and then
publishes that same selection as one interruption-safe critical section.

Catastrophic configuration conditions are limited to cases where safe recovery cannot be completed:
the current default itself violates the schema, the original authoritative bytes cannot be preserved
before a required whole-file reset, storage access cannot be made safe, or the surviving values
cannot satisfy a security or cross-field invariant without guessing user intent. These failures may
be returned explicitly. A merely missing, malformed, outdated, or partially invalid configuration
is not catastrophic.

## Shared implementation requirements

Recovery mechanics must be centralized rather than reimplemented by each cache or configuration
owner.

On the Bun side, the Effect Schema structured-file recovery utility is the common decoding
foundation. Cache adapters wrap it with no-fail semantics: filesystem failures, malformed and
unrecoverable decode results become scoped misses or defaults. Configuration adapters retain the
richer recovery report, preservation, and catastrophic-error behavior. A cache must not use a
single all-or-nothing decode when its schema contains independent entries.

The Rust side uses Serde and `serde_json`, with the common file mechanics and small recovery
combinators owned by `icn-utils`. The shared layer provides bounded no-fail reads, object/root
fallback, independent map and array entry decoding, best-effort directory creation, bounded binary
and JSON publication, restrictive temporary files, and same-directory atomic replacement. Cache
callers receive `Option`, an empty/default value, or a hit/miss type without an I/O or decode error
channel.

The schema owner explicitly chooses each recovery boundary. Ordinary Serde decoding is used for a
unit that should be kept or discarded as a whole. `#[serde(default)]` supplies absent values.
`serde_with` adapters such as `DefaultOnError`, `VecSkipError`, and `MapSkipError` may be used when
defaulting an invalid field or skipping an invalid collection member is exactly the desired domain
behavior. An explicit `serde_json::Value` boundary remains preferable when identity validation,
cross-field invariants, unknown-field preservation, or custom recovery decisions are required.

`serde_with` is a convenience, not a required layer in every cache and not an abstraction to wrap or
reimplement in `icn-utils`. The Rust design deliberately excludes a runtime schema algebra, a
generic recursive schema walker, and automatic mutation-and-retry of arbitrary decode-error paths.
New recovery helpers are added to `icn-utils` only after the same mechanically correct pattern is
needed by more than one consumer or is required to keep file I/O no-fail.

Disposable-cache reads may silently return misses. A bounded recovery report is optional and is
introduced only where it materially improves operational diagnosis. Durable configuration retains
the richer behavior: when recovery changes user-authored data it records affected paths, preserves
unknown fields, and preserves the original before a root reset.

## Conformance tests

The shared file utility is tested with the applicable filesystem and representation cases below;
individual cache/index consumers test their own recovery units and evidence invalidation without
repeating the entire utility matrix:

- missing and empty files;
- random, truncated, and invalidly encoded bytes;
- a valid outer representation with the wrong root type;
- missing, unknown, and wrong-typed fields;
- one invalid map entry or array element surrounded by valid siblings;
- an invalid identity or discriminator requiring removal of its enclosing entry;
- stale evidence for one of several otherwise valid entries;
- an oversized file or collection;
- a directory, broken link, and permission failure at the expected path;
- concurrent reads during replacement and an interrupted write;
- write, rename, cleanup, and lock failure.

Tests assert that valid siblings survive, only affected work is regenerated, cache failures do not
enter the caller's error channel, and failed writes do not change the operation result. Fuzz or
property-based coverage is appropriate for shared parsers that do more than bounded
`serde_json::Value` decoding; it is not required for every thin cache consumer.

Durable-configuration tests use the same corruption matrix and additionally assert minimal
defaulting, preservation of unknown fields, preservation of original bytes before a root reset, and
explicit failure only for the catastrophic cases defined above.

## Acceptance criteria

- Removing any recomputable cache or index produces, at most, additional work.
- No recomputable cache or index has a file-format schema version or migration path.
- Corrupting one independent value does not discard valid siblings.
- Arbitrary cache-file contents and filesystem failures cannot crash the process or fail the user
  operation solely because caching was unavailable.
- Cache validity is established by complete domain evidence, not by file presence or application
  version.
- Durable configuration retains valid user intent and defaults only the smallest unsafe subset.
- Bun and Rust use shared recovery facilities with equivalent externally observable guarantees.
