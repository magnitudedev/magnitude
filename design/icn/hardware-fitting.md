---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - inference/native/llama-cpp-rs/**
  - packages/icn/src/hardware/**
  - packages/icn/src/recipes/**
  - packages/acn/src/icn/**
  - packages/acn/src/local-model-inventory.ts
  - packages/acn/src/local-inference-hardware.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/protocol/src/rpcs/local-inference.ts
  - packages/protocol/src/schemas/model-state.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - cli/src/features/local-inference/**
  - cli/src/features/overlays/settings.tsx
  - cli/src/components/stacked-bar.tsx
---

# ICN hardware discovery and model fitting

ICN is the sole authority for inference hardware discovery and model-memory fitting. It exposes the
hardware visible to its pinned native runtime and uses one native planning path for both downloaded
models and remote catalog artifacts. The ICN integration package owns recipe curation and product
ranking through native preview, while ACN binds the results to mirrors and clients only present
ICN-derived facts.

The central parity guarantee is:

```text
fit(remote artifact headers, execution profile, hardware)
  == fit(downloaded artifact, execution profile, hardware)
```

Equality applies to the normalized structured planner result. It does not include incidental log
text, temporary paths, timestamps, or observations taken at different times.

## Ownership boundaries

ICN owns:

- host architecture, CPU, system-memory, and runtime-visible device discovery;
- backend and device enumeration through the pinned native runtime;
- normalization of devices into non-overlapping physical memory domains;
- stable-capacity policy and current free-memory observation;
- GGUF metadata, tensor-directory, shard, projector, draft, and MTP inspection;
- native no-allocation model, context, compute, and placement planning;
- compatibility classification for the pinned runtime and requested execution profile;
- hardware and fit cache identity, invalidation, and failure classification.

Generation-throughput estimation is separate evidence produced by the same native planning path.
Its workload, calibration, confidence, and failure guarantees are defined in
[generation-performance estimation](./performance-estimation.md). A performance failure never
changes a hardware-fit result.

The `@magnitudedev/icn` recipe service owns:

- the curated set of recommended artifact identities;
- product quality and quantization-fidelity policy;
- the fixed product profiles to evaluate;
- ranking successfully assessed candidates.

ACN owns the product projection and application commands. It joins private ICN recipes, inventory,
and hardware into `LocalModelInventory`, projects `LocalInferenceHardware`, and derives the local
provider catalog only from downloaded inventory entries. It preserves ICN assessments and does not
independently estimate fit or inspect the host.

Clients own formatting and interaction only. Bun and client code must not independently inspect the
OS, invoke native-backend device commands, classify unified memory, reserve device capacity, or estimate
model, KV, recurrent, graph, or compute memory.

Clients read product-owned inventory and hardware mirrors directly. They may derive layout and
formatting values, but do not join raw ICN mirrors, reconstruct acquisition/residency, or create a
second local-inference lifecycle.

Each inventory model exposes one authoritative context window. Curated models take it from the
selected catalog profile; independently discovered local models take it from inspected model
properties. A currently loaded serving configuration is not an alternate catalog context window,
and ACN never substitutes it for the authoritative value. A missing authoritative value is a
projection failure; there is no fallback value.

## One assessment pipeline

Available and preview models are not separate fitting implementations. They differ only in API
entry point, artifact acquisition, and persistence semantics. Both paths must resolve to the same
canonical component representation before inspection, execution-profile resolution, hardware
discovery, or fitting begins.

```text
AVAILABLE PATH

GET /v1/models[/<id>] -> inventory reconcile -> local-file adapter ----+
                                                                        |
                                                                        v
                                                         canonical resolved artifact
                                                         - component roles
                                                         - exact identities
                                                         - GGUF metadata/tensors
                                                         - original logical sizes
                                                                        |
PREVIEW PATH                                                           v
                                                         shared property inspection
POST /v1/models/preview -> catalog reconcile -> remote-header adapter --+
                                                                        |
                                                                        v
                                                         shared profile resolver
                                                                        |
GET /v1/hardware ---------------------> shared hardware service --------+
                                                                        |
                                                                        v
                                                         shared native assessor
                                                                        |
                                                                        v
                                                           HardwareAssessment
                                                              /       \
                                                             v         v
                                                    persist in       cache assessment
                                                     inventory       + return preview
```

The canonical resolved artifact is the convergence boundary. Downstream code must not know whether
its metadata came from complete local files, HTTP range reads, a content-addressed fit-header
bundle, or sparse temporary materialization. It receives the same component roles, content
identities, GGUF metadata and tensor descriptions, relationships, and original logical sizes.

The following services are single shared implementations:

- hardware enumeration and memory-domain normalization;
- stable-capacity policy;
- GGUF validation and model/component interpretation;
- effective execution-profile resolution;
- native no-allocation planning and typed auxiliary adjustments;
- conversion from the native report to `HardwareAssessment`;
- assessment fingerprinting and cache-key construction;
- invalid-artifact, incompatible-artifact, `DoesNotFit`, and operational-failure classification.

Local and remote acquisition adapters may read bytes differently, but they must not interpret model
architecture, calculate memory, select devices, resolve execution policy, or classify fit. API
handlers are thin orchestration boundaries and must not contain those behaviors.

The same hardware service supplies `GET /v1/hardware`, downloaded-model reconciliation, remote
preview, and load-time safety assessment. No endpoint may maintain its own device enumeration,
backend preference, memory-domain aliasing, or capacity snapshot implementation.

## Required internal decomposition

The implementation is divided into transport-independent services with one-way dependencies:

```text
native runtime/build identity
          |
          +----> hardware discovery + memory-domain normalization
          |                         |
          |                         v
local artifact adapter ----+   shared assessment service <---- execution-profile resolver
                            |            ^
                            v            |
                  canonical artifact ----+
                            ^            |
remote header adapter ------+            v
                                model-derived cache
                                         |
                    +--------------------+--------------------+
                    v                                         v
          inventory/list application service          preview application service
                    |                                         |
                    v                                         v
             GET /v1/models                         POST /v1/models/preview

hardware discovery ------------------------------------------------> GET /v1/hardware
```

The transport-neutral contracts define one canonical hardware snapshot, execution intent,
canonical component identity, and `HardwareAssessment`. The available and preview APIs may
use different request and envelope types, but they do not define separate assessment shapes.

The ICN composition root owns one process-lifetime native-backend capability. The hardware
discovery service requires that capability and owns logical-device enumeration, physical
memory-domain aliasing, stable capacity, volatile free-memory observation, and topology
fingerprinting. It never initializes or tears down a backend as part of an observation. Its result
is reused within an operation and may be refreshed according to one service-owned policy. API
state, inventory assessment, preview, and loading receive this service by dependency injection
rather than constructing their own snapshots.

The canonical artifact resolver owns component roles and relationships, exact identities, logical
sizes, and metadata sources. Local and remote adapters implement acquisition only. The shared
assessment service owns property inspection, execution-profile resolution, native planning,
capacity evaluation, typed adjustments, normalized reporting, and assessment-key construction.
Native calls share one serialization gate wherever the pinned runtime has process-global state.

The [model-management cache](./model-management.md#model-derived-cache) owns acquired header blobs
and the computed artifact, inspection, and assessment indexes. Hardware fitting defines its typed
evidence and validation rules but does not create a fit-specific cache service or construct
filesystem paths. Inventory may embed a completed assessment in its disposable snapshot, while
preview and available paths reuse the same assessment index.

The HTTP handlers perform request decoding, call the relevant application service, and translate
typed domain failures. They do not initialize native backends, fetch model metadata, construct
sparse artifacts, resolve profiles, or perform cache operations directly.

The available path persists completed inspection and assessment in inventory under the inventory
cache rules. The preview path durably caches the candidate assessment through the shared model cache
and returns it without publishing the candidate into local inventory or making it available for
loading. “Preview” describes model availability, not cache lifetime. Download publication converts
the artifact into the available path, which may reuse the shared cached assessment only when every
cache input remains identical and the complete downloaded content identity has been verified.

## Hardware API

ICN exposes `GET /v1/hardware`. The endpoint is available before any model is loaded and reports the
inference hardware visible to the running ICN build.

The response contains:

- capture time and a stable topology fingerprint;
- platform, native architecture, CPU identity, logical cores, total system memory, and current
  available system memory;
- native build fingerprint and enabled backends;
- runtime-visible logical devices with stable IDs where the backend supplies them, names, backend,
  device kind, physical-memory-domain membership, and any backend-reported device limit;
- normalized memory domains with total capacity, stable capacity, current free memory,
  host-memory-sharing semantics, and member devices.

Logical devices and memory domains are distinct. A device is a native execution target; a memory
domain is a physical capacity pool. Multiple backend views of one physical GPU and unified host/GPU
memory must not be counted as independent capacity. If ICN cannot establish safe aliasing, it must
use a conservative non-duplicating representation or fail discovery rather than overstate capacity.
The pinned fit bridge therefore carries the backend registration name and exact backend-reported
device identity into every per-device estimate; ICN uses exact identities, never display strings,
to merge fit capacity across backend views. Host and integrated-GPU allocations are likewise
charged to one host-sharing physical domain.

On Apple Silicon, ICN reports one `unified_memory` physical domain whose capacity and current
availability come from the OS system-memory observation. CPU and Metal are member devices of that
same domain. Metal's `recommendedMaxWorkingSetSize` is not physical VRAM or total system memory; it
is exposed as a `recommended_working_set` limit on the Metal device. Fit assessment charges all CPU
and Metal allocations to the unified physical domain once, and independently requires Metal-owned
allocations to fit within the Metal working-set limit. The API uses Rust's canonical `macos` and
`aarch64` platform vocabulary; presentation layers may format those values but must not use a
different vocabulary to decide topology.

Current free memory is observational and volatile. Recommendation eligibility uses stable capacity.
The hardware response must make that distinction explicit through the reported capacity values,
without exposing ICN's internal capacity or planning implementation as a caller-selected identifier.
Every behavior-changing execution resolver, capacity reserve, estimator, projector, MTP, and typed
auxiliary policy still has an opaque ICN-owned fingerprint in assessment evidence and cache keys.
Removing a caller-visible selector must never remove these internal invalidation inputs.

A device-enumeration or normalization failure fails the request with an actionable diagnostic. An
empty device list is valid only after successful enumeration establishes that no accelerator is
visible; failure must never be converted into the product statement “no GPU detected.”

## Downloaded-model assessment

`GET /v1/models` and `GET /v1/models/{model_id}` remain authoritative for downloaded models. Every
available inventory model carries a completed canonical hardware assessment as required by
[ICN model management](./model-management.md).

Downloaded-model assessment reads the complete local component set and runs the pinned native
planner for the model's current serving configuration. Loading independently reassesses the exact
configured model and serving profile; an inventory result is advisory and cannot authorize a different load
plan.

There is no separate imperative “assess this downloaded model” endpoint. Inventory reconciliation
performs and caches required assessment before returning an available model.

The available-model handler calls the shared assessment pipeline with a local-component acquisition
adapter. It does not call preview internally, and preview does not call the inventory HTTP endpoint;
both call the same internal application service below the transport layer.

## Remote candidate API

ICN exposes `POST /v1/models/preview` for catalog artifacts that have not been downloaded. The
request identifies an immutable artifact and one or more product execution profiles. It does not
accept caller-computed architecture or memory estimates.

A Hugging Face source contains at least:

```text
repository
immutable commit revision
primary GGUF path
zero or more explicitly selected projector, draft, or MTP GGUF paths and roles
```

ICN resolves every shard of the primary GGUF and every explicitly selected execution companion at
that revision. Companion relationships are derived by ICN rather than accepted as unchecked caller
metadata. A future
content-addressed fit-descriptor source may identify a Magnitude-published header bundle by URL,
size, and digest. Arbitrary caller-controlled fetch URLs are not accepted.

Each requested profile has a caller correlation ID and high-level variable inputs such as context
length and parallel sequence count. ICN resolves those inputs through the same planner used for
loading. Callers do not select a planner implementation or assemble unchecked native flags, and no
planner or capacity-policy identifier is part of the request or response contract.

The response contains:

- resolved immutable artifact and component identities;
- artifact properties deterministically available from GGUF metadata;
- one hardware assessment per requested profile;
- the artifact, native-build, and hardware-topology fingerprints supporting each result, plus the
  normalized execution facts in the assessment itself.

Remote and downloaded models use the same `HardwareAssessment` contract. A completed result is
`Fits`, `DoesNotFit`, `InvalidArtifact`, or `IncompatibleArtifact`. Invalid and incompatible
artifacts are domain results, not service failures. The structured fit result includes normalized
context and concurrency facts plus a per-memory-domain breakdown of model, context/KV, compute,
auxiliary, required, capacity, and margin bytes. It deliberately does not serialize tensor
placement or any other executable native plan.

The endpoint may accept a batch of candidate sources or gain a batch form without changing these
semantics. Metadata acquisition for independent artifacts may be concurrent. Native assessment is
serialized within one native process wherever required by process-global state. Profile cache
misses for one artifact are sent together to one private isolated planner process, bounded by the
request's sixteen-profile limit. The planner constructs the no-allocation model once and asks
llama.cpp to construct and measure every requested context graph against it. Independent artifacts
may be planned concurrently up to the ICN host's available processor count. Planner processes use
the same release-matched binary and canonical assessment implementation,
never expose a second service endpoint, never publish cache entries, and never share mutable native
state with the loaded inference executor. Batch correlation and caching avoid repeating header
acquisition or native work for duplicate artifact/profile pairs.

The preview handler calls the shared assessment pipeline with a remote-header acquisition adapter.
Remote acquisition is responsible only for resolving immutable components and supplying their exact
metadata bytes and logical identities. After canonical resolution, preview follows precisely the
same property, profile, hardware, planner, report-conversion, and failure-classification code as an
available model.

## Metadata-only native fitting

GGUF places key/value metadata and the complete tensor directory before tensor payload data. The
tensor directory describes each tensor's name, shape, data type, and offset. Those facts, together
with the execution profile and native device capabilities, determine planner allocation and
placement. Numerical weight values are not inputs to memory planning.

When the source does not publish the exact aligned header length, bounded range discovery grows the
locally accumulated prefix by requesting only the missing suffix. A larger probe must not reacquire
bytes already validated by an earlier probe. Each response is independently checked against its
exact requested range and the immutable artifact's logical size before it is appended.

For every GGUF shard, ICN retains the exact byte prefix from offset zero through the aligned tensor
data offset and records the original logical file size and content identity. The prefix may include
large tokenizer arrays and must not be replaced by a small hand-selected metadata subset. A remote
source that does not publish the header length may require bounded, increasing prefix probes; the
terminal HTTP range can extend beyond the discovered boundary, but bytes after the aligned header
are discarded and never cached or materialized. Preview acquisition never downloads the complete
weights file merely to discover that boundary.

To reuse the ordinary pinned llama.cpp file loader, ICN may materialize a temporary sparse artifact:

1. Create an isolated temporary directory.
2. Preserve every shard's original basename and relationship.
3. Write the exact acquired GGUF prefix at offset zero.
4. Set the sparse file's logical length to the original artifact length.
5. Leave the tensor-data region as an unallocated hole.
6. Run the same native no-allocation planner and execution profile used for a local artifact.
7. Remove the temporary artifacts after assessment.

The logical length is required because llama.cpp validates that declared tensor ranges fall within
the file even in no-allocation mode. The fit path must disable tensor allocation, mmap, and mlock and
must not read tensor payload from the sparse artifact. Sparse preview artifacts must be isolated and
marked so they can never be passed to an ordinary model load.

A future native metadata-source abstraction may replace sparse files, but it must preserve the same
llama.cpp model construction, buffer-type selection, no-allocation graph construction, placement,
and memory-breakdown behavior. Replacing the native planner with a parallel formula does not conform
to this design.

Multi-profile assessment may reuse an immutable no-allocation model construction while creating a
separate native context for every profile. A requested plan that fits according to its exact native
memory breakdown requires no placement search. If llama.cpp's exact model tensor-storage byte count
alone exceeds the aggregate stable capacity of all non-overlapping physical memory domains, the
model is conclusively `DoesNotFit`; changing placement cannot eliminate tensor storage. All other
misses run the pinned upstream placement fitter and validate its resulting native allocation. The
tensor-storage bound is architecture-independent native evidence, not a parameter-count estimate,
active-expert approximation, or model-family rule.

Assessment consumes the complete native memory breakdown produced by the constructed model and
context. Every buffer type is assigned by its native owning device and physical memory domain;
CPU-owned variants such as repacked expert tensors remain host or unified-memory requirements.
Parameter counts, active-expert counts, architecture names, and tensor-name conventions are never
substitutes for allocation bytes. An unknown non-empty native allocation domain fails assessment
rather than being dropped or guessed.

## Component completeness

Fit identity covers the complete execution component set. Every shard must contribute its GGUF
header and original size. A first shard alone is insufficient.

Projector, draft, and MTP components are included whenever the execution profile selects them. If
the pinned base-model planner does not account for a component, ICN applies the same versioned typed
adjustment or specialized native planner for both remote and downloaded paths. An auxiliary
adjustment must never exist only on one side of the parity comparison.

The fit report distinguishes base model, context/KV, compute graph, projector, draft/MTP, and other
reserved memory when the underlying planner can do so. Unsupported component combinations are
`IncompatibleArtifact`, not `DoesNotFit`.

## Catalog integration

ACN owns one versioned, model-centric catalog overlay. A checkpoint appears once and groups its
curated quantization choices. The checked-in overlay contains stable Hugging Face model and artifact
repository IDs, quantization selectors, product context profiles, display identity, quality and
fidelity evidence, benchmark methodology, and license review. It does not contain mutable Hub
commit revisions, filenames, shard membership, byte sizes, or content hashes.

ICN owns live Hugging Face discovery. Repository resolution accepts a mutable ref such as `main`
and returns an immutable per-request snapshot containing the resolved commit, current GGUF files,
sizes, content identities, license data, and Hub metadata. Search returns current GGUF repositories.
ACN joins an optional curated overlay to that snapshot. A repository absent from the overlay remains
usable and can still receive artifact-derived fit and performance estimates, but it receives no
Magnitude quality, fidelity, or support claim.

Preview and download consume the exact commit returned by resolution. The commit is transient
provenance for that preview or download, never a hand-maintained catalog pin. This prevents a
mutable branch from changing between selection and acquisition while allowing normal catalog
refreshes to observe current Hub state.

ACN submits catalog artifact identities and fixed product profiles to the preview API. It
excludes invalid, incompatible, and `DoesNotFit` candidates and applies only product ranking to the
remaining results. It must not modify or reinterpret ICN memory arithmetic.

Recommendation projection has an explicit lifecycle. Before an uncached preview calculation begins,
ACN publishes `Loading`; after the calculation it
publishes either `Ready` with the complete ranked recommendation set or `Failed` with a bounded
user-facing explanation. An empty `Ready` result means that no curated candidate fit. Clients must
not infer that outcome from an absent or empty recommendation list while calculation is in flight.
Cached results may transition directly to `Ready` without an observable loading state.

### Recommendation portfolio policy

Recommendation fitting evaluates one resident native sequence. Magnitude's inference engine owns
session multiplexing and KV-cache swapping, so user session count is not a recommendation input and
clients do not ask for it. The supported product context profiles are 200K and 100K tokens. ACN
prefers 200K for a chosen artifact and falls back to 100K when the larger profile does not fit; 64K
is not a recommendation profile.

Context profiles and quantizations are fit alternatives, not separate products. ACN first keeps the
best fitting context for each artifact, then keeps one configuration per base-model checkpoint. It
prefers the highest-fidelity fitting quantization and, within that quantization, the 200K profile.
The displayed portfolio contains distinct base models only. Badges describe an actual relationship
to the primary recommendation: a smaller option must have materially lower runtime memory, a
higher-fidelity option must have a higher curated fidelity rank, and an alternative should prefer a
different model family. List position alone never determines those labels.

### Catalog overlay versus runtime artifact facts

Magnitude evidence states whether it applies to the exact artifact, a quantized checkpoint, or only
a cross-model quantization tier; evidence may never be silently promoted to a stronger scope. An
online maintainer audit verifies that every stable repository ID still exists, every selector still
resolves uniquely against current Hub state, and reviewed licenses have not unexpectedly changed.
The audit reports current revisions but never writes them into the overlay.

Runtime code must not use the overlay as a second implementation of GGUF inspection. ICN's resolved
snapshot and preview supply shard membership, sizes, content identities, tensor metadata, parameter
counts, context, KV and recurrent-state dimensions, placement, backend support, memory, and speed.
ACN may use resolved file size only as a cheap rejection test before preview; it does not estimate
fit from that size.

## Bun and client boundary

ACN obtains host/device facts from `GET /v1/hardware`, downloaded
model facts from `GET /v1/models`, current Hub snapshots and search from the Hugging Face discovery
endpoints, and remote candidate assessments from `POST /v1/models/preview`. ACN continues to submit
the fixed product profiles and rank successful candidates using curated quality, fidelity, and
portfolio-diversity policy.

ACN contains no host inspection, native-backend device probing, memory-domain normalization,
stable-capacity arithmetic, hand-maintained runtime GGUF metadata, runtime-byte formulas, or direct
fit-command assessment. Such implementations would create a second hardware authority and are not
permitted as fallbacks.

The ACN local-inference projection maps the canonical ICN results into the existing product state
only where presentation requires it. It must preserve `Fits`, `DoesNotFit`, per-domain memory,
resolved profile, and failure distinctions rather than converting them back into a Bun estimate.
Downloaded ICN models and remote recommendations therefore display the same assessment semantics.
Every local model presented by the product is an ICN inventory model or ICN preview result; there is
no external-server product route.

Local inference is exposed through separate product resources with non-overlapping meanings.
Download progress and failure live on the addressed `LocalModelInventoryEntry`; load, ready, and
unload consequences live on `ModelSlot`; hardware is observational. Commands acknowledge with an
empty result and never return operation IDs. There is no operation collection or independent
residency resource. Progress holds the applicable FSM state, while terminal native observations are
reconciled through legal FSM transitions and published atomically.

## Cache validity

Hardware fitting persists only through the
[model-management cache](./model-management.md#model-derived-cache). Exact remote GGUF prefixes or
published header bundles are source blobs. Resolved artifacts, GGUF inspections, and completed
hardware assessments are computed indexes. Preview status, inventory membership, and endpoint
identity never define a separate cache category.

Header acquisition is cached by immutable source identity, component path, original size, content
identity, and header digest. A cached prefix is valid only when its bytes and declared tensor ranges
match the expected artifact facts.

Fit results are cached by every input capable of changing the result, including:

- complete component membership, identities, original sizes, and header digests;
- pinned native-backend revision and ICN native-build fingerprint;
- enabled backend set and relevant backend capability fingerprint;
- resolved execution inputs and effective serving profile;
- hardware topology and stable physical-capacity fingerprint;
- opaque versions of every execution resolver and capacity policy; and
- projector, draft, MTP, and typed-adjustment inputs and policy versions that can change the result.

The assessment key is shared by preview and available paths; it does not contain endpoint name,
inventory membership, temporary path, or acquisition-adapter kind. A local downloaded artifact and
a remote preview therefore reuse one result only when they resolve to the same canonical component
identities and all runtime, profile, hardware, and concrete planning inputs match.

Volatile free memory may be captured in a report but is not part of a reusable stable-capacity
result. Topology or stable-capacity changes invalidate affected assessments. Cache entries are
committed only after complete successful assessment.

## Failure semantics

`DoesNotFit` is a successful assessment. It means that a valid, runtime-supported artifact exceeds
the permitted capacity for the requested profile.

The remaining outcomes are:

| Situation | Result |
| --- | --- |
| Malformed, truncated, inconsistent, or invalid GGUF metadata or shards | `InvalidArtifact` |
| Unsupported architecture, quantization, component combination, or execution plan | `IncompatibleArtifact` |
| Source changed relative to its immutable identity | Reject, discard cached work, and reconcile |
| Network, authentication, device-enumeration, filesystem, or native-planner failure | Operation failure |
| Valid supported input cannot produce the declared assessment | ICN implementation defect |

Operational failures are never cached as model facts and are never normalized into `DoesNotFit`,
an empty hardware snapshot, or an `Unknown` assessment.

## Security and resource limits

Remote acquisition validates scheme, host policy, immutable revision, redirects, HTTP range
semantics, response length, and content-range boundaries. It enforces bounded metadata bytes,
component count, tensor count, string and array sizes, request concurrency, and total temporary
sparse-file logical size. Authentication material is never returned in responses or diagnostics.

Sparse artifacts live under a uniquely created temporary directory, use exact validated filenames,
cannot escape that directory, and are removed on success, failure, cancellation, and shutdown
recovery. Their location is never registered as a model source.

Metadata-only inspection does not establish integrity of unseen tensor payload bytes. Full download
still verifies the complete artifact content identity before publication.

## Accuracy boundary

Remote-header assessment is expected to equal downloaded-artifact assessment because both invoke
the same planner with equivalent inputs. It is not a proof that a real load will succeed.

Neither path alone guarantees protection from:

- memory consumed after assessment by other processes or models;
- transient load allocations not represented by the planner;
- driver or backend allocation failures;
- corrupt unseen tensor payloads;
- performance or throughput shortfalls.

The actual load path remains the final safety boundary and reassesses its resolved execution plan.
Performance claims require separate benchmark evidence.

## Resident memory observation

`GET /v1/hardware` is also the sole public view of current resident-runtime memory. Its hardware
snapshot may carry generation-bound per-domain allocation evidence in the native categories model,
context, compute, and auxiliary. No parallel telemetry endpoint or ACN-side device probe is
permitted.

The resident executor captures actual model, context, and compute backend-buffer allocations after
initialization and warm-up. It accounts shared target/draft ownership once and retains exact
projector allocation evidence when the projector has no live allocation accessor. The existing
physical-domain resolver maps these allocations by exact backend device identity; display names are
never identity evidence.

Current free memory remains volatile. A semantic read-only hardware observation command runs on the
resident executor between scheduler batches and combines current backend-device readings with the
generation's immutable allocation evidence. It does not wait for inference to become idle and does
not replace idle-only native planning.

Resident attribution is required before a loaded runtime may be published. Failure to capture the
native allocation report fails model loading with its typed binding error. Failure to resolve a
reported native location to an exact physical memory domain fails the hardware observation; it is
never represented as absent attribution. Allocated bytes and physically resident mmap pages can
differ on shared memory, so consumers must not fabricate a negative residual category.

CLI memory bars use one reusable stacked-bar renderer in both the detailed hardware view and the
compact chat status. Segment boundaries are quantized to eighths of a terminal cell and rendered
with a partial-block foreground over the following segment's background, preserving a contiguous
bar without whole-cell rounding. Weights/fixed cost, KV cache, system/apps, and free capacity use
distinct semantic theme colors. A model-free runtime displays used versus free; a loaded runtime
never silently falls back because resident attribution is part of successful load publication.

## Conformance and acceptance criteria

The implementation conforms when:

- ICN can enumerate hardware and preview remote candidates without loading a model;
- available and preview entry points converge on one canonical resolved-artifact type before any
  property inspection, profile resolution, hardware lookup, or fit calculation;
- exactly one implementation owns hardware enumeration, memory-domain normalization, capacity
  policy, native fit invocation, native-report conversion, and fit failure classification;
- local and remote adapters are limited to acquisition and canonical artifact resolution;
- API handlers contain no model-memory formulas, backend preferences, device normalization, or
  fit-result reinterpretation;
- hardware displayed by clients originates exclusively from `GET /v1/hardware`;
- ACN and clients contain no independent hardware normalization or fit arithmetic;
- uncached recommendation calculation is represented as `Loading`, never as an empty completed result;
- recommendation profiles use one resident sequence and only the 200K and 100K context tiers;
- each displayed recommendation represents a distinct base model, with badges derived from actual
  size, fidelity, or family differences rather than candidate position;
- downloaded inventory and remote preview return the same hardware-assessment type;
- complete headers from every shard are used, with exact original logical sizes;
- every allocation in the native model/context breakdown is accounted to a physical memory domain;
- remote preview and full local artifacts produce equal normalized reports for the same inputs;
- parity coverage includes dense, MoE, recurrent/hybrid, sharded, projector, CPU-only, unified-memory,
  discrete-GPU, and supported multi-device profiles;
- instrumentation verifies that metadata-only fitting retains and materializes only the exact GGUF
  header prefix and never performs a complete weights download;
- changing any artifact, runtime, hardware, profile, component, or concrete planning input invalidates the
  relevant cached result;
- deleting, corrupting, or making the model-derived cache unwritable causes only affected
  acquisition, inspection, or
  assessment work to be repeated and never changes the resulting assessment;
- `DoesNotFit`, invalid artifact, incompatible artifact, and operational failure remain distinct;
- loading independently validates the exact execution plan it will allocate;
- catalog runtime facts derivable from Hub artifacts are resolved by ICN rather than maintained
  in a parallel Bun metadata table;
- the canonical overlay groups quantizations under stable checkpoint identities and contains no
  commit, filename, shard, size, or content-hash snapshot;
- an explicit maintainer audit verifies current repositories, unique selectors, source identities,
  and reviewed licenses without downloading model weights or writing current commits into source;
- live resolution returns an immutable commit used unchanged by preview and download;
- short-lived discovery caches expire by ref and header/assessment caches remain keyed by immutable
  artifact identity, hardware topology, and estimator policy.
