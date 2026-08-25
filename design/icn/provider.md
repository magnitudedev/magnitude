---
applies_to:
  - packages/harness/src/**
  - packages/icn/src/provider/**
  - packages/acn/src/local-provider-**
  - packages/acn/src/model-*.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/agent/src/**
  - packages/acn-protocol/src/schemas/model-state.ts
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
---

# ICN provider contract

ICN implements Magnitude's `local` AI provider. Generic provider and agent code sees an ordinary
provider model ID and `BoundModel`; it does not see packages, downloads, assessments, native plans,
or runtime residency.

## Provider offerings

The local provider adapter projects each callable catalog model as one offering containing its
canonical model ID and current exact serving configuration. The configuration and offering are
derived rather than persisted.

ACN resolves configurations and capabilities from issued catalog entries. Catalog capabilities describe an uninstalled
offering. Once the effective target package has a completed inspection, that inspection is
authoritative because it describes the artifact inference will actually execute. When neither
source can establish capabilities, the offering remains disabled with conservative metadata rather
than claiming support.

Within the separate `local` provider namespace, the provider model ID is exactly the canonical
catalog model ID. Serving configurations have no separate public identity.

An offering exists independently of current installation, assessment, slot selection, or residency.
ACN's local-offering projection combines the resolved configuration with installed-package and assessment
observations to produce the provider model catalog entry. This is the only place that derives local
provider availability. When ICN can assess the exact configuration, that catalog entry carries the
complete per-domain memory accounting unchanged: capacity, required allocation, compatibility
reserve, and remaining assessment headroom. The local-model product projection consumes that
availability and derives application memory guidance from assessment, normalized hardware, and
resident allocation evidence on the same model row. Provider offerings do not carry application
warning policy, parallel aggregate fields, or capacity labels. Generic and cloud provider entries
do not fabricate local memory accounting.
Provider catalog presentation keeps the base display name and optional variant label as separate
fields. The local adapter derives both through the same ACN resolver used by the local-model product
projection; generic provider and agent code treat the label as presentation only.

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
adapter contributes one offering facet for each resolved configuration; it does not decide whether
a product row exists. Bundle identity is its tagged structure and ordered package identities.
Provider model IDs distinguish callable models, not derived configurations.

## Selection and resolution

A slot selection contains only provider ID, provider model ID, and reasoning effort. It references
an offering rather than copying its configuration.

Model-slot availability preserves incomplete authority as `Pending`. A configured slot remains
pending while its provider-catalog identity is unresolved or, for a local selection, while local
offerings or installed-package inventory are not yet authoritative. Agent configuration derives
the same lifecycle from the slot instead of reconstructing dependency readiness. A turn waits
interruptibly on its coherent configuration-and-toolkit projection without delaying message
publication. Once the observation terminalizes, a ready slot runs normally and a genuinely
unassigned or unavailable slot produces the typed `ModelNotReady` turn outcome; initialization is
never reported as an agent defect.

The ACN slot boundary normalizes reasoning effort before persistence: it preserves a supported
requested effort and otherwise selects the provider model's default. Stored selections are
normalized through the same operation when the catalog becomes available. The client and agent do
not independently repair reasoning effort.

The local provider resolver maps the selected canonical model ID to its exact current catalog
configuration. Provider binding is cheap and has no runtime side effect. Temporary
authority failure preserves the selection as unavailable. A complete authoritative projection
that lacks the identity clears the selection; neither case chooses another configuration.

Existing recency-based slot substitution remains product behavior. It operates on stable provider
model IDs and does not create, reassess, or rewrite offerings.

## Explicit instances

ICN owns one `ModelInstanceController` and currently permits at most one Ready local instance.

ACN's `ModelSlotController` remains only the product-intent authority. Inference addresses the
canonical model ID. ICN resolves its current configuration, joins or admits a physical instance,
waits for readiness, and acquires the inference lease. Explicit warm-load uses the same residency
coordinator without creating another loading path.

The submitted configuration fixes per-request context capacity. ICN independently resolves the
resident parallel allocation and reports it as load execution evidence; ACN does not persist or
select that allocation.

Loading another configuration replaces the singleton residency through the same serialized
transition. ICN creates and preserves a branded model-instance identity through loading and
residency. One Stop operation addresses only that exact identity. Stopping during Loading ends that
occurrence and every joined preparation waiter with the canonical non-retryable
`model_instance_stopped` condition. Stopping a Ready instance terminates active generation and
returns `model_instance_stopped` to those streams. Graceful replacement and idle
release retain lease draining. A delayed Stop cannot affect a newer instance of the same model.

## Concurrency and lifetime

The ICN `ModelInstanceController` delegates every physical decision to one residency actor, which
is the sole native mutation and lease authority.

- Same-model demand joins; conflicting demand is FIFO through the actor mailbox.
- Equivalent instance admissions are idempotent; caller interruption never cancels admitted work.
- A projected Loading or Stopping lifecycle always has a matching live ICN owner.
- Loading success and inference-lease grants are one state transition.
- Replacement closes new admission and drains existing leases; explicit Stop interrupts them.
- A completion holds one exact-instance lease until its body completes, fails, or is canceled.
- A failed mutation does not poison later attempts.
- Unexpected resident-worker loss is observed with the exact instance ID and becomes a typed
  blocked slot state; it is not inferred from generic provider unavailability.

Progress and terminal instance state come from `ModelInstancesSnapshot`; an individual inference
response is never the instance-lifecycle authority.

## Prompt and request boundary

The provider follows the shared [native Chat Completions contract](../ai/native-chat-completions.md).
It uses the shared request builder and encoder with the canonical model ID. The generated ICN
client validates and transports the request. ICN validates structural inputs before admission and
tokenizer-dependent constraints under the acquired resident lease. No instance identity or serving
configuration crosses the chat boundary.

Context admission uses the resident configuration's context length. Catalog metadata, compaction,
load planning, and request admission must agree on that exact configuration.

ICN lifecycle control chunks are process-local request observations, not assistant output. The
local provider reports meaningful preparation through the generic `preparation_update` stream event.
Ordinary provider-neutral response events remain unchanged. ICN defines the concrete preparation
payload; the provider-neutral AI contract does not. The agent model decorator owns transient
activity projection, preparation/response partitioning, and terminal cleanup, so the display sees
one continuous lifecycle while downstream agent consumers receive response events only. Providers
without preparation data use `TPreparation = never`.

If the backing instance is explicitly stopped during preparation, ICN emits the exact
non-retryable `model_instance_stopped` error. Explicit Stop after readiness terminates active
worker requests with the same condition. The local adapter maps that exact condition directly to
`ModelInstanceStopped` while adapting the known ICN error envelope, before generic stream-failure
construction. Downstream layers never recover this meaning by inspecting a failure cause.
Caller-side stream cancellation remains Effect interruption.

ICN also publishes one final cumulative timing snapshot for every accepted generation. The local
provider translates its generated-token count, decode duration, native decode rate, and time to
first token into the optional provider-neutral generation-performance contract. This final
measurement is independent of transient request progress and requires no per-token timing stream.
Generic agent code consumes the optional capability without branching on the local provider ID.

## Speculative decoding

A speculative-decoding bundle is explicit in the offering's configuration, including its method
and embedded or separate draft source. ACN does not attach, remove, or infer a draft or method
during provider resolution or chat.

ICN resolves embedded and separate draft capability through one native planning path. Assessment
and loading use the same exact bundle structure and speculative-selection policy. Runtime evidence
reports the selected method, effective parameters, and whether drafting actually ran.

## Failure behavior

- Selected identity absent from an authoritative offering projection: ACN clears the selection.
- Incomplete catalog, local-offering, or installed-package-inventory authority: agent slot
  resolution remains pending and turn execution waits without a timeout.
- Missing package: the provider catalog entry and slot are unavailable; chat does not trigger a
  download.
- Configuration no longer fits or is incompatible: the provider catalog entry is disabled and load
  fails with the typed ICN result.
- ICN unavailable or malformed response: ACN preserves the dependency/transport failure.
- Explicit Stop: every affected request terminates as `ModelInstanceStopped`
  and does not retry or immediately reload the model.

## Acceptance criteria

- Every local provider call resolves through one exact current configuration and one read-only
  provider projection.
- Runtime load receives the resolved ICN configuration unchanged.
- Local availability is derived in one ACN projection.
- Provider projection consumes assessment state and never admits assessment work.
- Provider availability never determines local-model product visibility.
- A completed aggregate catalog with provider failures is degraded even when it contains no models.
- Every assessed local provider catalog entry exposes ICN's complete per-domain memory accounting
  for that exact serving configuration.
- Provider binding does not load a model.
- Local preparation is represented only by generic request-local `preparation_update` events with ICN-owned detail.
- Inference admission remains held until ICN atomically acquires the request's generation lease.
- Chat, Responses, and explicit warm-load share one residency coordinator.
- Slot selection and recency refer only to stable provider model IDs.
- Speculative method and embedded/separate draft composition are identical during assessment, load,
  and inference.
