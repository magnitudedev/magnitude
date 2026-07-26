---
applies_to:
  - packages/ai/src/provider/**
  - packages/ai/src/response/**
  - packages/harness/src/events.ts
  - packages/harness/src/turn/dispatcher.ts
  - packages/icn/src/provider/**
  - packages/acn/src/local-provider-**
  - packages/acn/src/model-*.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/agent/src/model/**
  - packages/agent/src/errors/model-start.ts
  - packages/protocol/src/schemas/model-state.ts
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
---

# ICN provider contract

ICN implements Magnitude's `local` AI provider. Generic provider and agent code sees an ordinary
provider model ID and `BoundModel`; it does not see packages, downloads, assessments, native plans,
or runtime residency.

## Provider offerings

ACN owns durable local provider offerings. Each offering contains:

- a stable local provider model ID;
- the stable model-offering-target ID presented to product clients;
- one exact ICN-issued model serving configuration;
- its creation origin.

Capabilities are not persisted on the offering. ACN resolves them from the recommendable catalog
or the installed package inspection, which are the authoritative evidence for the target.

The local provider model ID deterministically namespaces the configuration identity. ACN never
hashes package or profile data to create another configuration identity.

An offering exists independently of current installation, fit, slot selection, or residency.
ACN's local-offering projection combines the durable offering with installed-package and assessment
observations to produce the provider model catalog entry. This is the only place that derives local
provider availability. When ICN can assess the exact configuration, that catalog entry carries the
complete per-domain memory accounting unchanged: capacity, required allocation, compatibility
reserve, warning reserve, and remaining headroom. Consumers derive aggregate requirements,
system-domain load admission, and warning presentation from that one value. They do not receive
parallel aggregate or fit-label fields. Generic and cloud provider entries do not fabricate local
memory accounting.

The target ID groups every serving configuration of the same standalone package or speculative
pair into one product model. Provider model IDs continue to distinguish configurations.

## Selection and resolution

A slot selection contains only provider ID, provider model ID, and reasoning effort. It references
an offering rather than copying its configuration.

The ACN slot boundary normalizes reasoning effort before persistence: it preserves a supported
requested effort and otherwise selects the provider model's default. Stored selections are
normalized through the same operation when the catalog becomes available. The client and agent do
not independently repair reasoning effort.

The local provider resolver maps the selected provider model ID to the offering's exact
configuration ID. Provider binding is cheap and has no runtime side effect.

Existing recency-based slot substitution remains product behavior. It operates on stable provider
model IDs and does not create, refit, or rewrite offerings.

## Explicit residency

ICN owns one native runtime coordinator and at most one resident configuration.

ACN's slot coordinator is the product lifecycle authority. A manual load and local request
preparation use the same service-owned slot transition:

1. resolve the selected offering;
2. require all target packages to be installed;
3. submit the offering's exact configuration to ICN load;
4. wait for terminal readiness; and
5. admit chat only against that resident configuration.

The submitted configuration fixes per-request context capacity. ICN independently resolves the
resident parallel allocation and reports it as load execution evidence; ACN does not persist or
select that allocation.

Loading another configuration replaces the singleton residency through the same serialized
transition. Unload addresses the returned residency identity and waits for active generation
leases to drain.

ICN chat never loads, configures, or selects a model. A chat request for a configuration that is
not resident fails without mutating runtime state.

## Concurrency and lifetime

The ICN runtime coordinator is the sole native mutation and lease authority.

- Load and unload mutations serialize.
- Equivalent callers join one shared completion; caller interruption never cancels the mutation.
- A loading or unloading slot always has a matching live coordinator owner.
- An identical load is idempotent after current state is rechecked.
- Replacement closes new admission and waits for existing generation leases.
- A completion holds one generation lease until its body completes, fails, or is canceled.
- A failed mutation does not poison later attempts.
- Unexpected resident-worker loss is observed with the configuration identity and becomes a typed
  blocked slot state; it is not inferred from generic provider unavailability.

ACN serializes product slot transitions and rechecks the attributed slot after preparation is
admitted. Preparation joins an in-progress load rather than treating it as ready. Progress is
observation only; terminal operation results drive slot state.

## Prompt and request boundary

The ICN provider encodes prompts once with the shared native chat-completions codec. The generated
client validates the request before transport. ICN validates structural inputs before accepting a
stream and validates tokenizer-dependent constraints under the resident lease.

Local-model preparation is a scoped agent-request phase injected by ACN. It is outside the generic
provider contract: preparation failure is a generic model-not-ready start result, never a fabricated
provider response or provider rejection. After preparation succeeds, the ordinary provider request
begins. ACN retains preparation admission until ICN accepts the request and owns its generation
lease, then releases it. Provider start and response-stream failures keep their existing provider
semantics and metadata.

Context admission uses the resident configuration's context length. Catalog metadata, compaction,
load planning, and request admission must agree on that exact configuration.

ICN lifecycle control chunks are process-local request observations, not assistant output. The
local provider removes them before the provider-neutral response codec and forwards queue, prefill,
and generation-start state through optional request attribution. ACN's preceding preparation phase
uses that same progress sink, so the display sees one continuous request lifecycle. A failed or
canceled start clears preparation progress; after acceptance, ending, failing, or canceling the
response stream clears provider-owned progress. Providers that do not support granular observation
remain valid and expose no synthetic progress.

ICN also publishes one final cumulative timing snapshot for every accepted generation. The local
provider translates its generated-token count, decode duration, native decode rate, and time to
first token into the optional provider-neutral generation-performance contract. This final
measurement is independent of transient request progress and requires no per-token timing stream.
Generic agent code consumes the optional capability without branching on the local provider ID.

## Speculative decoding

A speculative target is explicit in the offering's configuration. ACN does not attach or remove a
draft during provider resolution or chat.

ICN resolves target and draft components through one native planning path. Assessment and loading
use the same target identity and speculative-selection policy. Runtime evidence reports whether
drafting actually ran.

## Failure behavior

- Missing offering: ACN rejects provider resolution.
- Missing package: the provider catalog entry and slot are unavailable; chat does not trigger a
  download.
- Configuration no longer fits or is incompatible: the provider catalog entry is disabled and load
  fails with the typed ICN result.
- ICN unavailable or malformed response: ACN preserves the dependency/transport failure.
- Nonresident chat: ICN rejects it without a load side effect.

## Acceptance criteria

- Every local provider call resolves through one durable offering.
- Runtime load receives the stored ICN configuration unchanged.
- Local availability is derived in one ACN projection.
- Every assessed local provider catalog entry exposes ICN's complete per-domain memory accounting
  for that exact serving configuration.
- Provider binding does not load a model.
- Local preparation is not represented as a provider response or provider failure.
- Preparation admission remains held until ICN accepts the request's generation lease.
- Chat cannot mutate residency.
- Slot selection and recency refer only to stable provider model IDs.
- Target/draft composition is identical during assessment, load, and inference.
