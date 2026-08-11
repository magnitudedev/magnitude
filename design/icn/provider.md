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
  - packages/acn-protocol/src/schemas/model-state.ts
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
---

# ICN provider contract

ICN implements Magnitude's `local` AI provider. Generic provider and agent code sees an ordinary
provider model ID and `BoundModel`; it does not see packages, downloads, assessments, native plans,
or runtime residency.

## Provider offerings

ACN owns durable retained model serving configurations. The local provider adapter projects each
configuration as one offering containing a stable local provider model ID and that exact
configuration. The offering is a projection rather than another persisted record.

Capabilities are not persisted as offering state. ACN resolves them from the recommendable catalog
or installed package inspection. When neither source can establish capabilities, the retained
configuration remains visible as disabled with conservative metadata rather than claiming support.

Within the separate `local` provider namespace, the provider model ID is exactly the serving
configuration ID. ACN never prefixes or hashes package or profile data to create another identity.

An offering exists independently of current installation, assessment, slot selection, or residency.
ACN's local-offering projection combines the durable configuration with installed-package and assessment
observations to produce the provider model catalog entry. This is the only place that derives local
provider availability. When ICN can assess the exact configuration, that catalog entry carries the
complete per-domain memory accounting unchanged: capacity, required allocation, compatibility
reserve, warning reserve, and remaining headroom. Consumers derive aggregate requirements,
system-domain load admission, and warning presentation from that one value. They do not receive
parallel aggregate or capacity-label fields. Generic and cloud provider entries do not fabricate local
memory accounting.

Initial and invalidation-driven projection runs in one scoped background worker. Native assessment
never gates ACN service readiness.

The local-offering projection consumes the shared per-configuration assessment state; it never
invokes native assessment. Package notifications may change offering availability, while the
separate local-model assessor decides assessment admission from semantic assessment keys.
Download progress may update product acquisition presentation but cannot transition an offering's
configuration to `Assessing`.

The aggregate provider catalog remains usable when aggregation completes with typed provider or
catalog failures. Such a snapshot is `Degraded`, including when every successful source contributes
an empty model list; an empty result does not turn a partial provider failure into aggregate
unavailability.

Catalog refresh is catalog-owned shared work. Equivalent callers join one refresh; conflicting
targeted refreshes serialize. Every Effect `Exit` publishes a terminal Ready, Degraded, or
Unavailable snapshot before releasing ownership, so caller loss cannot strand `Refreshing`.

Product visibility and grouping belong to
[Local-model product projection](../model-management/local-model-product-projection.md). The provider
adapter contributes one offering facet for each retained configuration; it does not enumerate
installed packages or decide whether a product row exists. Bundle identity is its tagged structure
and ordered package identities; the retained configuration is the provider's complete operational
authority. Provider model IDs continue to distinguish configurations.

## Selection and resolution

A slot selection contains only provider ID, provider model ID, and reasoning effort. It references
an offering rather than copying its configuration.

The ACN slot boundary normalizes reasoning effort before persistence: it preserves a supported
requested effort and otherwise selects the provider model's default. Stored selections are
normalized through the same operation when the catalog becomes available. The client and agent do
not independently repair reasoning effort.

The local provider resolver maps the selected provider model ID directly to the exact retained
configuration. Provider binding is cheap and has no runtime side effect.

Existing recency-based slot substitution remains product behavior. It operates on stable provider
model IDs and does not create, reassess, or rewrite offerings.

## Explicit instances

ICN owns one `ModelInstanceController` and currently permits at most one Ready local instance.

ACN's `ModelSlotController` is the product intent authority. A manual load and local request
preparation use the same canonical slot and instance observation:

1. resolve the selected retained configuration through its provider offering;
2. require all bundle packages to be installed;
3. create a fresh `ModelInstanceId` and submit the exact configuration to ICN;
4. bind the slot after native admission and observe that exact instance to Ready; and
5. start chat with that exact instance ID and configuration.

The submitted configuration fixes per-request context capacity. ICN independently resolves the
resident parallel allocation and reports it as load execution evidence; ACN does not persist or
select that allocation.

Loading another configuration replaces the singleton residency through the same serialized
transition. ACN creates a branded model-instance identity before loading and ICN preserves it
through loading and residency. One Stop operation addresses only that exact identity, whether it is
still loading or already resident, and waits for partial cleanup or active generation leases to
drain. A delayed Stop cannot affect a newer instance of the same configuration.

ICN chat never loads, configures, or selects a model. A chat request for a configuration that is
not resident fails without mutating runtime state.

## Concurrency and lifetime

The ICN `ModelInstanceController` is the sole native mutation and lease authority.

- Load, Stop, and replacement mutations serialize.
- Equivalent instance admissions are idempotent; caller interruption never cancels admitted work.
- A projected Loading or Stopping lifecycle always has a matching live ICN owner.
- An identical load is idempotent after current state is rechecked.
- Replacement closes new admission and waits for existing generation leases.
- A completion holds one generation lease until its body completes, fails, or is canceled.
- A failed mutation does not poison later attempts.
- Unexpected resident-worker loss is observed with the configuration identity and becomes a typed
  blocked slot state; it is not inferred from generic provider unavailability.

ACN rechecks the attributed slot and exact instance after preparation. Progress and terminal state
come from `ModelInstancesSnapshot`; the load response stream is never the lifecycle authority.

## Prompt and request boundary

The ICN provider encodes prompts once with the shared native chat-completions codec. The generated
client validates the request before transport. ICN validates structural inputs before accepting a
stream and validates tokenizer-dependent constraints under the resident lease.

Local-model preparation is a scoped agent-request phase injected by ACN. It resolves and observes
the exact Ready instance, then installs an Effect fiber-local instance binding for the provider
request. The local provider refuses to encode a request without that binding. ICN atomically
acquires the exact `ModelInstanceLease`; no ACN residency-admission bridge exists.

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

A speculative-decoding pair is explicit in the offering's configuration. ACN does not attach or
remove a draft during provider resolution or chat.

ICN resolves target and draft components through one native planning path. Assessment and loading
use the same exact bundle structure and speculative-selection policy. Runtime evidence reports whether
drafting actually ran.

## Failure behavior

- Missing retained configuration: ACN rejects provider resolution.
- Missing package: the provider catalog entry and slot are unavailable; chat does not trigger a
  download.
- Configuration no longer fits or is incompatible: the provider catalog entry is disabled and load
  fails with the typed ICN result.
- ICN unavailable or malformed response: ACN preserves the dependency/transport failure.
- Nonresident chat: ICN rejects it without a load side effect.

## Acceptance criteria

- Every local provider call resolves through one retained exact configuration and one read-only
  provider projection.
- Runtime load receives the stored ICN configuration unchanged.
- Local availability is derived in one ACN projection.
- Provider projection consumes assessment state and never admits assessment work.
- Provider availability never determines local-model product visibility.
- A completed aggregate catalog with provider failures is degraded even when it contains no models.
- Every assessed local provider catalog entry exposes ICN's complete per-domain memory accounting
  for that exact serving configuration.
- Provider binding does not load a model.
- Local preparation is not represented as a provider response or provider failure.
- Preparation admission remains held until ICN accepts the request's generation lease.
- Chat cannot mutate residency.
- Slot selection and recency refer only to stable provider model IDs.
- Target/draft composition is identical during assessment, load, and inference.
