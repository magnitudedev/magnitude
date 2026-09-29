# ICN parity results

Primitive-parity runs write one immutable directory under `parity/`. A run contains its resolved
plan, provenance, per-case reference and candidate evidence, raw bounded stdout/stderr captures,
comparisons, and a summary. Files are never reused across runs.

The run also contains byte-for-byte snapshots and digests of the profile, selected cases, fixtures,
registry, producer/target declarations, and schemas actually parsed for that execution. Those inputs
are reverified after the run so concurrent edits cannot produce evidence for a different contract.

Each accepted result identifies native source and binary digests, build configuration, fixture and
model digests, backend/device configuration, the exact operation boundary, and all raw timing
repetitions. A result whose work or effective configuration does not match is `invalid`, not an
approximate comparison.

Generated evidence is intentionally ignored. CI should upload complete run directories as
artifacts; a reviewed release report may reference their digests separately. Whole server
responses, SSE transcripts, and load-test results do not belong here because they are not primitive
parity evidence.
