---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - packages/acn/src/local-inference/**
  - packages/protocol/src/rpcs/local-inference.ts
  - packages/protocol/src/schemas/local-inference.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - cli/src/features/local-inference/**
---

# ICN hardware discovery and model fitting

ICN is the sole authority for inference hardware discovery and model-memory fitting. It exposes the
hardware visible to its pinned native runtime and uses one native planning path for both downloaded
models and remote catalog artifacts. ACN retains catalog curation and product ranking, while clients
only present ICN-derived facts.

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

ACN owns:

- the curated set of recommended artifact identities;
- product quality and quantization-fidelity policy;
- user usage choices and the product profiles to evaluate;
- ranking successfully assessed candidates and projecting them into client RPC state.

Clients own formatting and interaction only. Bun and client code must not independently inspect the
OS, invoke llama.cpp device commands, classify unified memory, reserve device capacity, or estimate
model, KV, recurrent, graph, or compute memory.

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

The available path persists completed inspection and assessment in inventory under the inventory
cache rules. The preview path durably caches the candidate assessment through the shared fit cache
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
  device kind, total memory, current free memory, and physical-memory-domain identity;
- normalized memory domains with total capacity, stable capacity, current free memory,
  host-memory-sharing semantics, and member devices;
- the versioned capacity-policy identity used by fit assessment.

Logical devices and memory domains are distinct. A device is a native execution target; a memory
domain is a physical capacity pool. Multiple backend views of one physical GPU and unified host/GPU
memory must not be counted as independent capacity. If ICN cannot establish safe aliasing, it must
use a conservative non-duplicating representation or fail discovery rather than overstate capacity.

Current free memory is observational and volatile. Recommendation eligibility uses the versioned
stable-capacity policy. The hardware response must make that distinction explicit.

A device-enumeration or normalization failure fails the request with an actionable diagnostic. An
empty device list is valid only after successful enumeration establishes that no accelerator is
visible; failure must never be converted into the product statement “no GPU detected.”

## Downloaded-model assessment

`GET /v1/models` and `GET /v1/models/{model_id}` remain authoritative for downloaded models. Every
available inventory model carries a completed canonical hardware assessment as required by
[ICN model management](./model-management.md).

Downloaded-model assessment reads the complete local component set and runs the pinned native
planner. Loading independently reassesses the exact requested execution plan; an inventory result
is advisory and cannot authorize a different load plan.

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
```

ICN resolves every shard and associated required component at that revision. A future
content-addressed fit-descriptor source may identify a Magnitude-published header bundle by URL,
size, and digest. Arbitrary caller-controlled fetch URLs are not accepted.

Each requested profile has a caller correlation ID and a versioned product-policy identity plus the
high-level variable inputs, such as context length and parallel sequence count. ICN resolves those
inputs through the same execution-policy implementation used for loading. Callers do not assemble
unchecked native flags.

The response contains:

- resolved immutable artifact and component identities;
- artifact properties deterministically available from GGUF metadata;
- one hardware assessment per requested profile;
- the artifact, native-build, execution-policy, hardware-topology, and capacity-policy
  fingerprints supporting each result.

Remote and downloaded models use the same `HardwareAssessment` contract. A complete assessment is
either `Fits` or `DoesNotFit`. The structured result includes the resolved execution profile,
placement, selected GPU layers and tensor split where applicable, and a per-memory-domain breakdown
of model, context/KV, compute, auxiliary, required, capacity, and margin bytes.

The endpoint may accept a batch of candidate sources or gain a batch form without changing these
semantics. Metadata acquisition for independent artifacts may be concurrent; native assessment is
serialized wherever required by process-global llama.cpp state. Batch correlation and caching must
avoid repeating header acquisition or native work for duplicate artifact/profile pairs.

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

For every GGUF shard, ICN acquires the exact byte prefix from offset zero through the aligned tensor
data offset and records the original logical file size and content identity. The prefix may include
large tokenizer arrays and must not be replaced by a small hand-selected metadata subset.

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
must not read tensor payload ranges. Sparse preview artifacts must be isolated and marked so they can
never be passed to an ordinary model load.

A future native metadata-source abstraction may replace sparse files, but it must preserve the same
llama.cpp model construction, buffer-type selection, no-allocation graph construction, placement,
and memory-breakdown behavior. Replacing the native planner with a parallel formula does not conform
to this design.

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

The maintained catalog contains deliberate product facts and immutable artifact selectors. Humans
choose the repository, commit, primary artifact, product grouping, quality rank, fidelity evidence,
license decision, and any reviewed display override.

A development-time generator derives artifact facts from the pinned Hugging Face revision:

- complete shard/component paths, sizes, and content identities;
- download and source URLs;
- GGUF properties and header digests;
- fit-header bundle identity when headers are published as catalog assets.

Generated facts are checked in as a deterministic catalog lock or published as content-addressed
catalog assets. Ordinary application startup and tests do not depend on mutable Hugging Face model
metadata. Runtime header acquisition always verifies the pinned revision and expected component
identity supplied by the generated catalog.

ACN submits catalog artifact identities and usage-derived product profiles to the preview API. It
excludes invalid, incompatible, and `DoesNotFit` candidates and applies only product ranking to the
remaining results. It must not modify or reinterpret ICN memory arithmetic.

## Cache validity

### Cache storage and lifecycle

The shared fit cache lives below ICN's configured model-store root, alongside but separate from the
durable inventory. With the default model-store location its logical layout is:

```text
~/.magnitude/models/
  inventory-index.json
  hub/
  fit-cache/
    headers/
      <header-content-digest>
    artifacts/
      <resolved-artifact-key>.json
    assessments/
      <assessment-key>.json
```

The exact on-disk encoding may change, but the ownership and separation are required:

- ICN owns the cache under the configured model-store root;
- ACN and clients never read or write it directly;
- preview candidates do not become inventory records merely because they are cached;
- downloaded inventory evidence and preview requests consult the same assessment cache service;
- deleting `fit-cache/` is safe and causes reassessment, not model or inventory loss.

`headers/` is a content-addressed store for exact remote GGUF prefixes or published fit-header
bundles. `artifacts/` records the validated mapping from immutable source components to header
digests and original logical sizes. `assessments/` stores normalized completed planner results.

Fit-cache files follow the shared
[file-based cache and recovery contract](../misc/file-based-caching.md): they carry no file-format
schema version, tolerate corruption as granular cache misses, and never fail assessment solely due
to cache I/O. Writes are atomic and use restrictive filesystem permissions. The cache supports
bounded size or age-based garbage collection without consulting product catalog policy.
An in-flight map coalesces concurrent requests for the same key; it is an execution optimization,
not a second cache or source of truth. Operational failures and incomplete assessments are never
persisted.

Header acquisition is cached by immutable source identity, component path, original size, content
identity, and header digest. A cached prefix is valid only when its bytes and declared tensor ranges
match the expected artifact facts.

Fit results are cached by every input capable of changing the result, including:

- complete component membership, identities, original sizes, and header digests;
- pinned llama.cpp and ICN native-build fingerprint;
- enabled backend set and relevant backend capability fingerprint;
- resolved execution profile and execution-policy version;
- hardware topology and stable physical-capacity fingerprint;
- capacity-policy and estimator-policy versions;
- projector, draft, MTP, and typed-adjustment policy versions.

The assessment key is shared by preview and available paths; it does not contain endpoint name,
inventory membership, temporary path, or acquisition-adapter kind. A local downloaded artifact and
a remote preview therefore reuse one result only when they resolve to the same canonical component
identities and all runtime, profile, hardware, and policy inputs match.

Volatile free memory may be captured in a report but is not part of a reusable stable-capacity result
unless the requested policy explicitly uses it. Topology or stable-capacity changes invalidate
affected assessments. Cache entries are committed only after complete successful assessment.

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
- downloaded inventory and remote preview return the same hardware-assessment type;
- complete headers from every shard are used, with exact original logical sizes;
- remote preview and full local artifacts produce equal normalized reports for the same inputs;
- parity coverage includes dense, MoE, recurrent/hybrid, sharded, projector, CPU-only, unified-memory,
  discrete-GPU, and supported multi-device profiles;
- instrumentation verifies that metadata-only fitting never reads tensor payload ranges;
- changing any artifact, runtime, hardware, profile, component, or policy input invalidates the
  relevant cached result;
- `DoesNotFit`, invalid artifact, incompatible artifact, and operational failure remain distinct;
- loading independently validates the exact execution plan it will allocate;
- regular builds and client operation do not require mutable Hugging Face metadata.
