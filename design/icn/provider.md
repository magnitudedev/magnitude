---
applies_to:
  - packages/ai/src/provider/**
  - packages/ai/src/model/**
  - packages/ai/src/transport/**
  - packages/providers/src/registry.ts
  - packages/openapi-effect/**
  - packages/icn/**
  - packages/icn/src/provider/**
  - packages/icn/src/inventory/**
  - packages/acn/src/icn/**
  - packages/acn/src/shared-client.ts
  - packages/acn/src/account.ts
  - packages/agent/src/window/**
  - packages/agent/src/observer/prompt.ts
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
---

# ICN provider contract

ICN implements Magnitude's `local` AI provider. A local model is invoked through the same
provider-facing operation as any other model: bind a provider model ID, then stream a completion.
Model residency is an implementation concern of that invocation. Callers do not perform a
readiness check, load a model, select an ICN process, or use a second runtime-specific completion
operation before inference.

This design separates three concepts which must never be collapsed:

- **model identity** identifies one immutable inventory artifact and its components;
- **serving configuration** records the caller-owned execution intent for that model; and
- **runtime residency** records whether a resolved serving target currently occupies the singleton
  native executor.

Confusing these concepts causes stale fit decisions, duplicate load logic, races between load and
chat, and provider errors that appear after an unrelated readiness transition.

## Provider-facing contract

The local provider implements the ordinary provider contract:

1. Its catalog lists ICN inventory models and their provider capabilities.
2. Binding a model captures only its provider model ID and ordinary provider defaults. Binding is
   cheap, has no runtime side effect, and succeeds independently of current residency.
3. Calling the bound model submits one ordinary streamed chat-completion request naming that model.
4. The call does not return a usable response stream until ICN has admitted the HTTP request,
   resolved the model's serving target, ensured that target is resident, and acquired a lease on
   the resulting runtime generation.
5. The generation lease remains held until the response body completes, fails, or is canceled.

The agent, provider registry, and generic AI package see only `BoundModel<BaseCallOptions>`. They do
not see load targets, runtime generations, hardware profiles, MTP choices, or ICN lifecycle types.
The `@magnitudedev/icn` provider translates the generic prompt and call options to the generated ICN
chat schema exactly once. ACN registers that provider without implementing a second provider
protocol.

`POST /v1/chat/completions` is the only ICN inference operation. There is no runtime-chat endpoint,
restart-chat endpoint, preflight endpoint, or state-check/load/chat sequence. The chat request names
the inventory model and contains inference inputs only. It does not carry native flags, resolved
paths, a runtime generation, or a second copy of the model target.

## Identity, configuration, and residency

### Model identity

An ICN model ID resolves to a canonical inventory artifact containing its weights, shards,
projector, draft, and MTP relationships. Artifact resolution and component interpretation are ICN
responsibilities. ACN never reconstructs identity by comparing repository names or component
paths during a provider call.

### Serving configuration

Every callable inventory model has one authoritative serving configuration in the running ICN.
The durable user choice remains in ACN configuration; it is not artifact metadata and is not
written to installation manifests or derived inventory caches. The minimum caller-owned
configuration is:

```text
ServingConfiguration
  context length
  parallel sequence count
```

The configuration contains intent, not a native plan. Device placement, batching details, KV
layout, projector attachment, and draft or MTP selection are resolved by ICN from that intent and
the canonical artifact.

An available model always has a valid default serving configuration. Publishing or reconciling a
model establishes that default using the canonical product profile that was assessed for the
artifact; a process launch default such as 4096 tokens is not a model serving configuration.
Product selection may replace the configuration through the idempotent generated operation
`PUT /v1/models/{model_id}/serving-configuration`.
ACN owns the product choice and reconciles that durable selection before publishing its local
provider catalog, as well as when an explicit activation applies the choice. Both paths invoke one
idempotent selection-to-serving-configuration translation; they do not maintain separate profile
logic. This startup reconciliation rehydrates the process-local ICN state before the catalog can
advertise the model. ICN validates it with the same assessment pipeline used by preview and load
and projects it from model listing for the current process lifetime.

Managed download requests carry the selected serving profile so publication can establish the
current process configuration before availability. On a later ICN start, ACN reapplies its durable
selection. ICN may derive a temporary default for standalone inventory use, but that default is not
persisted as user intent and cannot override ACN's selection.

Changing serving configuration is distinct from loading. It does not require the caller to issue a
load before the next completion. If the model is resident under an older configuration, the next
demand resolves the new target and the runtime coordinator performs the required replacement.

### Runtime residency

ICN owns one native runtime coordinator and at most one resident serving target. A target is the
internal combination of canonical model identity and its effective serving profile. Pinned native
policy and the resolved execution plan remain ICN implementation details. The target is never
assembled in ACN and is never a provider request field.

Catalog availability and residency are independent dimensions:

```text
availability: available | unavailable
residency:    not_resident | loading | resident | load_failed | unloading
```

`available` means the artifact is valid, configured, supported, and fits the applicable hardware
policy, so it may be called directly. It does not mean the model is already loaded. A retryable
load failure does not silently make a valid model disappear from the catalog; the failure remains
visible in residency state. Permanent artifact, compatibility, or fit failures make the model
unavailable with their canonical reason.

Model listing is the authoritative projection of both dimensions. ACN mirrors that projection and
does not infer it from provider calls, maintain a parallel active-model cache, or manufacture a
successful `loaded` provider result.

## One fulfillment path

Both ordinary inference and explicit eager load call the same transport-independent operation:

```text
acquire(model ID)
  -> read the model's serving profile
  -> resolve canonical artifact and auxiliary components
  -> assess and resolve the execution plan
  -> serialize and perform the required runtime transition
  -> return a lease for the resident generation
```

Chat continues from that lease into template application, tokenization, generation admission, and
streaming. Explicit load discards the lease after readiness is established. Unload enters the same
coordinator, closes new admission, waits for active generation leases to drain, and then releases
native resources.

The public model-residency operations are therefore limited to:

- **load(model ID)**: eagerly run the same acquisition transition; useful for user-requested warmup, but never a
  prerequisite for chat;
- **unload(model ID)**: idempotently release that model when resident after its leases drain.

They are explicit `POST` commands on the named model's `/load` and `/unload` resources. There is no
public `runtime` resource; model listing is the authoritative state projection.

Load does not accept a second competing execution policy. Serving configuration is changed through
model management, while load realizes the currently configured target. Restart is not a domain
operation: when genuinely required it is the ordered composition of unload and load by the owner
of that user operation. It is never a distinct inference path.

HTTP handlers contain no load algorithm. Chat and load handlers both delegate to the runtime
coordinator. The inventory service contains no runtime mutex. The ACN provider adapter contains no
readiness algorithm.

## Concurrency and lifetime

The runtime coordinator is the only mutation/admission authority. It uses one mutation lock and a
monotonically increasing resident generation.

- Concurrent demand serializes on that lock. After acquiring it, every caller re-reads the
  effective profile and rechecks residency; if the preceding call loaded the same target, the next
  call returns readiness without loading again.
- There is no cached transition promise or second single-flight state machine. A failed call does
  not poison or own later calls; a later caller may retry under the same serialization rule.
- Demand for a different target queues behind the current mutation and is resolved only after
  acquiring mutation authority, so stale configuration cannot win a race.
- New generation leases are closed before replacement or unload begins. Existing leases drain
  before native resources are mutated.
- A completion holds exactly one lease and cannot switch generations mid-stream.
- Completion of a load transition publishes the resident generation once. Model-list residency
  derives from that transition.
- Failure publishes one typed transition failure. It never leaves a half-installed backend or a
  state that claims readiness without a leasable generation.

There is no ACN-side mutex, target cache, shared promise, polling loop, or state-check/load race.

The coordinator's core transition returns a typed terminal result. Progress is a non-blocking
broadcast observation and is never the result channel; an internal chat admission cannot deadlock
because a bounded progress consumer is absent.

## Deferred work and user-visible state

Provider binding is not model loading. Property discovery `Deferred` means that an optional model
property has not been requested or resolved; it does not describe runtime residency. Provider
health `loading` describes ACN/ICN process startup; it does not describe a model transition.

Model loading is deferred until the first completion unless an explicit load warms it earlier.
Once a completion has entered ICN:

- loading is normal progress of that same request, not a provider failure;
- the request waits for serialized residency establishment rather than failing and being retried by
  the agent;
- only a terminal load failure rejects the request; and
- the model-list residency projection may update independently, but a `loaded` transition is never
  presented as the completion's outcome.

ACN active-work accounting covers the complete interval from provider request admission
through loading and generation. It does not end after starting an ICN stream or after receiving a
load operation ID.

## Request validation and prompt history

The ACN adapter encodes the prompt with the shared native chat-completions codec and validates the
generated request schema before transport. ICN performs structural validation before beginning a
destructive runtime replacement when validation does not require a resident tokenizer. Validation
which requires the resident model occurs after acquiring the target lease but before accepting the
SSE response.

An assistant message sent to ICN must contain text, reasoning, or at least one tool call. A failed
model attempt is not an assistant message. Agent history stores failure feedback separately from a
committed assistant turn, so retry rendering cannot fabricate an empty assistant message. The
provider adapter rejects invalid locally generated history as a client correctness violation; it
does not repair, omit, or reinterpret messages silently.

Context capacity is checked against the serving configuration used to acquire the lease. Catalog
context limits, compaction policy, load-time planning, and request admission must all refer to that
same effective profile. An inventory assessment made with a generic process default cannot be
substituted for the configured context.

## MTP and execution-plan fidelity

MTP remains part of the ICN execution plan. The canonical model artifact records MTP and draft
relationships; the single ICN resolver selects compatible auxiliary components when resolving a
serving target. Preview fitting, available-model fitting, load-time safety assessment, and actual
backend construction use the same artifact, serving configuration, native planner, and selection
rules.

ACN neither chooses MTP files nor moves MTP outside the plan. A provider request cannot disable or
replace MTP by omitting an ACN-derived target. The resolved plan and loaded model state expose
enough typed evidence to verify which auxiliary components were selected without duplicating the
selection algorithm.

## Streaming and failures

The provider boundary distinguishes request admission from response-body streaming. The generated
transport must expose admission as an Effect which either returns accepted response metadata plus
an event stream, or fails with a generated start error. It must not hide the HTTP request inside a
lazy stream when doing so loses the rejected response status and body.

Failure classification is determined exactly once from the real boundary:

| Boundary | Provider failure |
| --- | --- |
| Local prompt or schema encoding fails | stream-start client correctness violation |
| ICN cannot be reached before response | stream-start operational failure |
| ICN returns a declared non-2xx response | stream-start provider rejection with the actual status and body |
| ICN accepts SSE and emits a provider error envelope | stream provider error with the accepted status |
| An accepted response body cannot be read or terminates incorrectly | stream operational failure with decoded progress |
| Chunk decoding violates the declared schema | stream provider correctness violation |

The generated OpenAPI runtime owns HTTP, SSE framing, response metadata, declared error decoding,
cancellation, and body cleanup. `@magnitudedev/icn` and ACN do not wrap an actual 400 or 500 in a
synthetic accepted-200 `BodyReadFailure`, create endpoint-specific SSE parsers, or stringify a typed
remote error into an opaque cause.

ICN does not send status 200 until the target is resident, the request has passed admission, and an
SSE body can be produced. A load or validation failure therefore remains a real non-2xx start
failure. Once status 200 is sent, subsequent inference failures are typed in-band stream failures.
When a valid provider error envelope contains an actionable message, the normalized failure
snapshot preserves that message separately from its diagnostic rendering and user-facing error
presentation displays it. A generic stream-failure label must not replace a known provider reason.

## Ownership summary

`@magnitudedev/icn` owns recipe curation, the local provider catalog projection, generic prompt
encoding for ICN, and generated chat failure mapping. ACN owns user selection, provider
registration, catalog aggregation, and projection into client RPC state. It translates a changed
user choice into one generated ICN model-configuration request. It does not own runtime readiness or
execution planning.

The generated ICN boundary owns typed requests, response admission, streaming transport, and exact
error preservation. Authored ICN services may coordinate observation, recipes, or provider
adaptation, but never rename generated operations or introduce an alternate HTTP implementation,
runtime state machine, or lifecycle cache.

ICN model management owns canonical artifact identity, serving configuration, inventory
availability, inspection, and fitting. The ICN runtime coordinator owns target resolution,
resident generations, mutation serialization, leases, load/unload, and inference admission. The
native executor owns the resolved model, projector, MTP/draft runtime, scheduler, and token work.

## Forbidden duplicate logic

A conforming implementation has none of the following:

- an ACN state-check followed by load and chat;
- an ACN runtime-target resolver, profile fallback, component matcher, or active-model cache;
- a second runtime-specific completion endpoint;
- different load implementations for explicit load and completion;
- provider catalog availability derived from current residency;
- hardware, fitting, projector, draft, or MTP selection in Bun;
- hand-written ICN HTTP/SSE parsing or endpoint-specific transport-error conversion;
- a process launch default used as an available model's serving context;
- failed attempt feedback encoded as an empty assistant message; or
- a synthetic HTTP 200 attached to a rejected ICN response.

## Conformance criteria

The provider design conforms when:

- any available model can be called directly from an empty runtime and the call either streams a
  valid response or returns one correctly classified failure;
- direct chat and explicit load for the same configured model perform at most one effective load
  and produce one resident generation;
- changing serving configuration cannot race a call into loading the previous target after the new
  configuration has won admission;
- catalog context, fit assessment, loaded context, and request capacity all use the same serving
  profile;
- a 100K or 200K configured model is never implicitly loaded at a 4096-token process default;
- ACN startup reconciles its selected product profile before the local provider catalog advertises
  that model's context capacity;
- concurrent incompatible calls never overlap native mutation and an active response never changes
  generation;
- model listing exposes availability and residency without ACN inference or polling;
- MTP and projector selection are identical in assessment and actual load and remain ICN-owned;
- an ICN HTTP 400 is observed as a start rejection with status 400 and its typed error body;
- an in-band ICN provider error is shown with its provider-authored reason rather than only a
  generic stream-failure label;
- no failed turn can produce an empty assistant message in the next provider request; and
- killing an ACN terminates its one private ICN child, while model calls within that ACN all use
  that same generated client and runtime coordinator.
