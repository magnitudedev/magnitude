---
applies_to:
  - packages/acn/src/inference-observations.ts
  - packages/acn-protocol/src/schemas/inference-observation.ts
  - packages/acn-protocol/src/boundary/inference.ts
  - packages/acn/src/inference-observations.test.ts
---

# Inference request observation

ACN may retain bounded, opt-in observations of its local inference response stream
for companions whose host keeps response parsing private. ICN remains the owner of
inference execution. The host parser remains the owner of the caller's result.
An observation is neither a model-state mirror nor a second inference operation.

An observing caller supplies two opaque UUIDs: a fresh request observation ID for
each HTTP attempt and a group ID identifying the caller's presentation scope.
Neither is authentication, model selection, or trusted harness identity. Unrelated
requests never join merely because they selected the same model. Duplicate active
IDs are not adopted or overwritten. Missing/invalid observation metadata disables
observation without changing inference behavior.

Only native progress and timing metadata are retained. Prompts, tools, generated
text, credentials and full response frames are never stored. Metadata is decoded
against the canonical inference schemas. The original response bytes, status,
headers, cancellation and backpressure remain unchanged. An observation failure
must not fail an otherwise usable inference response. Upstream provider traffic
is not observed by this mechanism.

HTTP proxy cancellation covers the entire response lifetime, not just acquisition
of its headers. The upstream request must remain linked to the downstream HTTP
request's abort signal after the handler returns its response. Conformance tests
exercise actual upstream/proxy/client connections with observation both enabled
and disabled; cancelling an in-memory reader alone does not establish this rule.

Each observed response has one lifetime: Active → Ended. Ended means the response
observation closed, not that inference succeeded. A companion may publish a
successful summary only after its own host accepts the corresponding result.
Late frames cannot reopen an ended observation or update a replacement occurrence.
Observer detachment cannot stop inference or unload a model.

The application scope owns the observation registry. It admits at most 128 retained
requests; exhausted capacity declines observation rather than evicting active work
or blocking inference. Ended observations expire after 60 seconds. Public reads
are bounded snapshots selected by group; they never start a service, inference,
load, download or retry. Reconnecting readers reread the snapshot and retain no
independent authority. ACN replacement invalidates the registry through ordinary
instance fencing.

Conformance includes byte-identical streaming, split UTF-8 and SSE delimiters,
malformed/oversized metadata, no progress, absent timings, cancellation, concurrent
groups, duplicate IDs, late frames, bounded retention and application shutdown.
