---
applies_to:
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-reasoning/**
  - inference/crates/icn-hardware/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
---

# ICN model management

ICN maintains a durable inventory of locally usable models. Listing models is the single authoritative operation that reconciles the inventory and completes model inspection. Startup, filesystem changes, downloads, deletion, and model loading may update or invalidate known facts, but they do not independently start an inventory-wide inspection.

This design intentionally makes model listing an authoritative, potentially slower query:

> Reconcile the durable observation, complete every required inspection, and then return it.

It is not a snapshot of partially completed background work.

## Successful-list invariant

Every model returned with `Available` status has a complete, current assessment. ICN has established that:

1. The complete component set exists and has a stable content identity, including shards, projector, and MTP components where applicable.
2. The artifacts are valid GGUF files accepted by the pinned ICN runtime.
3. The exact effective chat template has been resolved with the same template-selection, fallback, BOS, EOS, and named-template behavior used when loading the model.
4. Every reasoning control supported by ICN has been conclusively classified as `Supported` or `Unsupported`.
5. The canonical product execution profile has been assessed against the current backend and hardware topology.
6. Hardware has been conclusively classified as `Fits` or `DoesNotFit`.
7. Any cached result matches every input that can affect the result.

Consequently, a successful list response must not expose `Pending`, `Assessing`, `NotAssessed`, or a generic `Unknown` for an available model. Those values describe orchestration progress or an incomplete implementation, not durable model properties.

Non-ready records, such as an in-progress or interrupted download, may still appear with their operation status. They are not `Available` models and are outside the completed-assessment invariant until their artifact set is ready.

## Authoritative reconciliation

The list-models handler performs one shared reconciliation operation with the following ordered
phases:

1. **Discover artifacts.** Enumerate the observed model directories, group related components, and
   compute each candidate's stable identity from filesystem facts. Discovery does not open GGUF
   metadata, inspect templates, or assess hardware.
2. **Recover index entries.** Treat the durable index as an optional, disposable cache and decode
   each candidate's cached model and evidence independently. An entry is reusable only when its
   complete current shape is valid and its component, content, and filesystem identities match the
   discovered files. Any missing or invalid field discards that entry without affecting unrelated
   entries. A missing, unreadable, malformed, or wholly unrelated cache file is equivalent to an
   empty cache and never fails reconciliation.
3. **Partition the candidates.** Reuse complete, current inspection results immediately. Put every
   candidate with no entry, a malformed entry, changed artifacts, or stale inspection evidence into
   one enrichment set. Entries whose artifacts are no longer present are omitted from the new
   snapshot.
4. **Enrich only the stale set.** Inspect the stale candidates concurrently with a bounded worker
   count. Each worker validates GGUF metadata, resolves the execution-equivalent effective template,
   and derives the complete public model properties. Inspection produces either a complete available
   model or a typed invalid/incompatible artifact record.
5. **Validate hardware evidence.** Reuse hardware results only when their independent evidence key
   matches the current canonical execution profile, native build, backend, and hardware topology.
   Assess only available models with missing or stale hardware evidence. Native assessment is
   serialized where required by process-global runtime state; other work remains concurrent.
6. **Verify stability and publish.** Confirm that the discovered component identities still match
   the files. If they changed during reconciliation, discard the attempt and retry under the bounded
   mutation policy. Once every available model is complete, atomically persist and publish the new
   inventory snapshot, then return it. Cache locking, serialization, or persistence failure does not
   fail the completed result; the next list may recompute it.

Concurrent list requests share the same in-flight reconciliation. They do not duplicate directory
scans, metadata inspection, template probes, or native hardware assessment.

Startup initializes storage, recovers interrupted operations, and may hydrate the durable index, but does not inspect every model. Download and acquisition completion directly records or invalidates the affected artifact. Deletion removes or invalidates the affected entry. Filesystem and runtime watchers only invalidate relevant cache entries. None of these paths launches inventory-wide enrichment.

The configured ICN model store is the product's authoritative model inventory. Its managed
Hugging Face hub is contained within that store. Default configuration does not scan or adopt the
host user's global Hugging Face cache, and ACN does not supply external cache or directory roots.
Explicit read-only roots remain an ICN deployment input, not an ACN-side discovery mechanism.

## Reasoning discovery

Reasoning discovery describes the mechanical controls exposed by the effective chat template and
normalizes them to an ordered option list with a default. A model with no detected reasoning
behavior has the complete profile `none`, defaulting to `none`. Fixed reasoning has a non-disableable
enabled profile. The detailed normalization and family examples are defined in
[reasoning detection](./reasoning-detection.md).

Inspection must use the same effective template inputs as model execution. A missing template in GGUF metadata does not imply an unknown result: the pinned native backend's fallback behavior must be applied. Actual BOS and EOS tokens and any applicable named template variant must also be resolved rather than omitted.

Probe errors must not be converted to the `none` profile. `none` is valid only after inspection
successfully establishes absence of supported reasoning behavior. Likewise, a control that accepts
a single alternate effort level, a non-default toggle value, or another supported domain shape must
not be discarded merely because the current probe implementation fails to represent it. Such a case
is an inspector defect until the public contract deliberately excludes that control.

The discovery boundary is intentionally narrower than model intelligence. ICN can determine which supported controls a template exposes and how those controls affect rendering. It cannot infer from weights whether a model is good at reasoning, and it cannot promise to enumerate arbitrary values accepted by an unrestricted template program. The public capability contract must therefore describe the finite control vocabulary ICN supports. Within that declared vocabulary, discovery is expected to be total.

## Hardware discovery

Hardware assessment is made for a versioned canonical product execution profile. Given that profile, the complete result is:

```text
Hardware = Fits(profile, memory, recommendation)
         | DoesNotFit(profile, memory deficit, limiting resource, alternative)
```

`DoesNotFit` is a complete and successful result. It is not an assessment failure.

The profile fixes every execution input that can affect the result, including context length, sequence and batch sizing, KV types, acceleration and GPU-layer policy, projector or MTP selection, and capacity policy. The native planner reads model structure, enumerates devices, constructs the model and context plan, accounts for model, context, KV, compute, projector, and MTP memory, and evaluates the preferred and permitted fallback configurations.

Inventory assessment is advisory for the canonical profile. Loading a model still performs an exact safety assessment for the execution plan actually requested. The loader must not rely on a cached inventory assessment when those plans differ.

Native hardware assessment is serialized where required by process-global native-backend state. Other metadata and template work may use bounded concurrency. Implementation limits on concurrency do not weaken the completeness requirement.

## Exact failure taxonomy

The ordinary expectation is that every valid, stable model supported by the pinned runtime produces all required properties. Failure to do so is exceptional. The following categories are exhaustive:

| Situation | Classification | Required behavior |
| --- | --- | --- |
| The effective template exposes no ICN-supported reasoning behavior | Successful discovery | Return the normalized option list `none`, defaulting to `none`. |
| The canonical execution profile exceeds available resources | Successful discovery | Return hardware `DoesNotFit`. |
| An artifact is unreadable, truncated, malformed, or not GGUF | Invalid artifact | Do not return it as `Available`. Preserve a specific artifact diagnostic if invalid artifacts are exposed. Do not attach unknown properties. |
| The pinned runtime does not support the architecture, quantization, component combination, or required execution plan | Incompatible artifact | Do not return it as `Available`. Report a specific incompatibility at model availability level, not unknown reasoning or hardware. |
| A file disappears or changes during inspection | Concurrent mutation | Discard the attempt, reconcile identity, and retry from a stable snapshot. If stability cannot be established within the bounded retry policy, fail the ensure request. |
| Filesystem access, device enumeration, allocation-free native planning, template compilation, or another required dependency fails unexpectedly | Operational failure | Fail the ensure request with the underlying diagnostic. Do not persist a partial result or return a successful response containing unknown fields. |
| The inspector or estimator cannot derive a declared property from a valid, stable, runtime-supported model despite having the required inputs | ICN implementation defect | Treat the ensure operation as failed, preserve diagnostics, and fix the implementation. Do not normalize the defect into a product state. |
| The public contract asks for a semantic fact that artifacts and deterministic execution cannot establish | Contract defect | Narrow or redefine the contract. Do not add an `Unknown` state to conceal an unobservable property. |

Invalid and incompatible artifacts are expected to be rare and are not available models. Operational failures and ICN implementation defects are also not expected paths. They should be observable as errors and must not be cached as model facts.

Invalid and incompatible artifacts discovered in configured sources are exposed as explicit top-level
artifact records so local problems remain actionable. They carry a typed status and diagnostic, are
never loadable, and never masquerade as available models with unknown properties. Managed artifacts
may still offer deletion. Their presence does not weaken the property invariant for models labeled
`Available` or prevent other stable models from being returned.

## Model-derived cache

ICN model management owns one cache for all recomputable facts derived from local or remote model
artifacts. Inventory, artifact resolution, metadata inspection, reasoning detection, hardware
assessment, and profile execution assessment are domains within this cache, not independent cache systems. ACN and clients never
read, write, or construct paths inside it.

The cache lives below the configured model-store root and has two top-level data forms:

```text
~/.magnitude/models/
  hub/                              authoritative downloaded artifacts
  cache/                            safely disposable as a whole
    blobs/                          faithful cached source bytes
      gguf-headers/
        <content-digest>
    indexes/                        computed structured projections
      inventory.json
      artifacts/
        <artifact-key>.json
      inspections/
        artifacts/
          <inspection-key>.json
      assessments/
        hardware/
          <assessment-key>.json
        execution/
          <assessment-key>.json
```

Blobs are exact byte sequences acquired from an authoritative source and retained to avoid
acquiring them again. GGUF header prefixes are blobs. They are addressed by content identity and
validated for expected length, digest, and domain-specific structure before use.

Indexes are computed structured projections regenerated from authoritative artifacts, validated
blobs, runtime state, hardware state, or other current inputs. The inventory, resolved artifacts,
artifact inspections (including template and reasoning results), hardware assessments, and preview
execution assessments are indexes. An execution assessment atomically contains the hardware fit
and optional generation-performance evidence produced from that same native placement. Each index
key covers every input capable of changing its result. Invalid or incomplete entries are misses at
the smallest independently recomputable unit.

The distinction controls identity, validation, and garbage collection, but not failure behavior.
Both use the shared [file-based cache and recovery contract](../misc/file-based-caching.md). One
model-cache service owns root resolution, domain paths, and bounded reads and writes. Domain
components supply typed keys, values, validation, and recomputation; they do not assemble
filesystem paths. Application services coalesce in-flight work by complete evidence key without
creating another persistence mechanism. `icn-utils` supplies no-fail file mechanics but knows
nothing about model domains or evidence keys.

Cache lookup precedes expensive source materialization. When the resolved-artifact index, artifact
inspection, and every requested execution assessment are independently valid, preview assembles its
response from those indexes without opening header blobs, constructing sparse files, running GGUF
or template inspection, querying the remote source, or invoking native planning. A source blob is
required to recompute a missing derived fact; its later deletion does not invalidate an otherwise
complete derived result whose evidence already includes the validated blob digest and immutable
artifact identity.

```text
authoritative local files ---------+
                                    |
immutable remote sources -> blobs -+-> artifact index -> inspection indexes
                                                        |              |
runtime/template policy --------------------------------+              |
                                                                       v
hardware + execution profile ---------------------------> execution assessment index
                                                                       |
                                                                       v
                                                           inventory index projection
```

The inventory is a materialized listing projection, not a second authority for embedded reasoning
or hardware facts. It may embed current domain results for efficient listing, but those values come
from the same inspectors and assessors and carry the same evidence identities as their independently
reusable indexes. Preview caching does not add a remote candidate to inventory. Once downloaded,
the available path reuses the same indexes only when every evidence key matches.

New model-derived data extends `cache/blobs/` or `cache/indexes/` with a domain-specific namespace.
It does not create a sibling cache root, endpoint-owned cache, or feature-specific filesystem path.
Deleting `cache/` is always safe and repeats acquisition, discovery, inspection, and assessment; it
never deletes downloaded models or user-authored data.

### Cache validity and recovery

A path existing in the cache is not evidence that its assessment remains valid. Reconciliation invalidates results when any relevant input changes, including:

- component membership, relationships, size, timestamps, or content identity;
- effective chat template, tokenizer, BOS/EOS tokens, or named-template selection;
- the complete validated shape of each cached model and evidence entry;
- pinned native-backend revision, native build, backend, or estimator fingerprint;
- canonical execution profile or capacity-policy version;
- hardware and device topology relevant to planning.

The inventory cache has no schema, format, or inspection-algorithm version. Its current decoder
recovers entries structurally and independently; data it cannot establish as a complete current
entry is discarded and recomputed. The cache is never a migration boundary or durable source of
truth.

Reasoning and hardware indexes use separate domain evidence keys because their inputs differ, while
sharing the same cache owner and mechanics. Completed results are written atomically only after all
required work for the model succeeds. Interrupted or failed inspection must leave the previous
valid entry intact only when its key is still valid; otherwise the model remains stale and the list
request fails rather than returning it as current.

Garbage collection may apply independent age or size bounds to blob and index namespaces without
changing correctness. Cache reachability is an optimization, never correctness state. Obsolete
cache locations are ignored and regenerated rather than migrated or treated as competing sources
of truth. Operational failures and partial computations are never persisted; stable negative domain
results such as `DoesNotFit` may be cached because they are complete results for their evidence key.

## Contract consequences

The durable and wire contracts should represent domain outcomes, not internal progress. For available models:

- reasoning is a completed normalized option list with a contained default;
- hardware has only `Fits` and `DoesNotFit`;
- property inspection has a completed form, not `Pending`;
- inspection errors are operation errors or top-level artifact availability diagnostics.

Progress for downloads and other explicit operations remains in their operation/status contracts. If inspection progress needs observability, it belongs in logs, traces, or an operation-specific endpoint, not in a successful model inventory snapshot.

## Acceptance criteria

The implementation satisfies this design when:

- daemon startup can complete without an inventory-wide scan or native assessment;
- the first list after a cold start reconciles and fully assesses all available models;
- a warm list discovers artifacts without opening unchanged GGUF metadata and reuses only
  independently valid inspection and hardware evidence;
- missing, malformed, changed, and stale index entries form one bounded-parallel enrichment set;
- one malformed index entry does not invalidate unrelated valid entries;
- no cache-file read, parse, shape, lock, serialization, or write failure can fail model listing;
- all ICN model-derived persistence is under the one model-management cache root and is classified
  as either a source blob or a computed index;
- domain code does not construct cache paths or create feature-specific cache services;
- preview and available flows reuse identical typed results within their cache domains exactly when
  their evidence keys match;
- deleting or making the complete cache unwritable does not change computed results or fail an
  otherwise successful operation;
- simultaneous lists share one ensure operation;
- adding, changing, regrouping, or removing artifacts invalidates exactly the affected entries;
- no successful response contains unresolved properties for an available model;
- missing template metadata follows runtime fallback semantics;
- reasoning inspection uses execution-equivalent token and template inputs;
- the reasoning `none` profile and hardware `DoesNotFit` can only result from successful inspection;
- corrupt or incompatible artifacts never masquerade as available models with unknown properties;
- unexpected native, environmental, or derivation failures fail reconciliation with actionable diagnostics;
- model loading independently validates the exact requested execution plan.
