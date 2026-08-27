---
applies_to:
  - inference/**
  - packages/icn/**
  - packages/icn-protocol/**
  - packages/acn/**
  - packages/acn-protocol/**
  - packages/agent/src/ambient/config-ambient.ts
  - packages/sdk/**
  - packages/client-common/**
  - cli/**
  - web/**
---

# Client, ACN, and ICN ownership

## Boundary

```text
first-party clients -> one Effect Query client -> ACN RPC -> private ICN management
external inference clients --------------------> ACN /inference/{v1,anthropic}/** -> serving
```

ACN is the complete first-party application boundary. Clients do not call ICN management routes,
consume ICN resource events, or reconstruct ACN product state from native resources. The only
public inference boundary exposes the OpenAI-compatible data plane under `/inference/v1/**` and
the Anthropic gateway under `/inference/anthropic/**`.
`/inference/api/**` is not public.

## ICN responsibility

ICN owns physical inference truth: native models and packages, installation and download
occurrences, hardware observations, planning, assessment, safety, instance lifecycle, residency
policy, and inference execution. ICN operations revalidate native preconditions at admission.

ICN does not own Magnitude Slots, favorites, ranking preference, provider policy, presentation
states, or client interaction flows. Native changes may originate from ACN commands, chat
completions, residency policy, recovery, or another internal caller, so ACN observes them
independently of command attribution.

## ACN responsibility

ACN owns the complete client-facing model contract:

- `ModelCatalog`: the unified local and remote product view. Each local row carries one
  `acquisitionState` union covering the model's whole materialization lifecycle — disk truth,
  the single transfer that may exist for it, unacknowledged transfer failure, update
  availability, and (once any version is installed) runtime residency. Every variant is a
  reachable product state; progress and failure payloads exist only under the states they
  belong to, and native occurrence identities (download and instance ids) never appear;
- `ModelSlots`: durable selection resolved to truthful client-ready Slot states;
- `LocalInferenceEnvironment`: normalized hardware and memory presentation facts; and
- model-management commands and their application-level validation. Commands are addressed by
  product identity (`modelId`, `slotId`); ACN resolves the native occurrence.

ACN also owns stable ranking scores, warnings, provider status, favorites, Slot reference
integrity, and the atomic configuration admitted to agent work. It may project ICN facts into
application semantics, but it does not become the authority for the native resource.

The public resources remain separate because they have different identities and failure causes.
They are not combined into one result or one stored God object. A screen may compose their query
results for presentation without creating another authority.

### State shape

Queries return their domain value directly. There is no public `{ revision, state }` envelope, no
generic mirrored-state abstraction, and no client-visible source revision.

A service may retain state only when it owns a real domain lifecycle that cannot be derived from a
current authoritative read. Such state has a domain-specific representation and invariants. Views
fully determined by current inputs are computed directly and are not stored:

- hardware queries and per-model residency project current ICN observations directly;
- the unified Catalog is a pure union of the current remote catalog and local product projection;
- Slots retain their real resolved lifecycle because ACN owns durable selection, reference
  reconciliation, and admitted agent configuration together; and
- onboarding reads durable storage directly.

Subscription references are an implementation primitive, not an architecture or wire contract.

### Native observation

ACN consumes ICN's private resource observation once. A native event is notification to reread
authority; it is never installed as state. A chat-triggered automatic load follows the same path
as an explicit load command:

```text
ICN transition -> native invalidation -> authoritative ICN read -> ACN projection change
               -> ACN query-name poke -> client query invalidation and reread
```

The inference response stream and resource observation have independent lifetimes. Cancelling a
chat request does not corrupt ACN observation, and an observation failure does not terminate or
reinterpret the response proxy.

## Client and SDK responsibility

The SDK owns typed transport and daemon lifecycle. Client-common owns one connection-scoped Effect
Query client over ACN, shared hooks, invocation state, identity-preserving selectors, and shared
formatting. Individual clients own wording, layout, icons, and ephemeral interaction state.

Every first-party model interaction is a member of the ACN `Models` Query/Mutation group. There is
no raw RPC client, native inference management client, parallel event drain, or second model cache
in CLI, web, or client-common.

Clients render each query Result independently. Waiting or failure is not authoritative emptiness.
A failed Catalog observation cannot clear Slots; a failed Instance observation cannot erase a
durable selection; and no model-domain Result can select the daemon-disconnection screen.

Clients do not fabricate descriptors from IDs, synthesize zero-valued context, or join partial
facts into backend policy. `Unassigned` has exactly the established “No model configured”
presentation. `Resolving` is explicit and never displayed as a raw-ID fallback.

## Mutation causality

An ACN mutation acknowledges only after its owning ACN service has committed the immediate
application postcondition, or after ICN has admitted a named long-running occurrence. The returned
Download or Instance identity is the causal acknowledgement for admitted work; progress remains a
query concern.

The client mutation synchronizes affected queries before reporting success. For model mutations,
that means one awaited refetch of the ACN snapshot the owner has already committed; it is not a
poll or a predicate-based proof protocol. ACN change pokes remain necessary for changes caused
externally or by continuing native work. These paths are complementary:

- mutation synchronization closes the initiating client's causal window;
- change pokes converge every client and every non-command transition.

Primary Slot assignment and agent work admission share one serialization scope. This orders the
owner commit against work capture; it does not make the client authoritative.

## Failure containment

ICN failure, restart, malformed native evidence, projection failure, model absence, assessment
failure, stopped instances, and command rejection are model-domain outcomes. They cannot close the
ACN serving scope or be classified as daemon disconnection. Only confirmed loss of ACN transport
may select the daemon-disconnection presentation.

Retained state is not cleared by transient or incomplete evidence. Slot intent may be cleared only
by explicit user action or complete authoritative proof that the referenced identity is invalid.

## Conformance

- No first-party code accesses `/inference/api/**`, native ICN management operations, or ICN
  events.
- `/inference/v1/**` remains a transparent proxy; `/inference/anthropic/**` routes reserved local
  aliases to ICN and preserves non-reserved upstream Anthropic request bytes.
- Every client-visible model query and mutation belongs to the ACN `Models` group.
- Query successes are direct domain values with no generic revision envelope.
- No generic mirrored-state helper or schema exists.
- Derived Catalog, Instance, and Environment views are not stored as parallel authorities.
- Model mutation synchronization awaits one fresh committed ACN snapshot; it does not poll or
  reread for proof.
- Model sync, cancellation, failure acknowledgement, and removal are addressed by canonical
  model ID. ACN publishes the resulting `LocalModel.acquisitionState` before returning success;
  it privately correlates admitted model syncs with exact ICN download IDs. ICN continues to
  expose raw download-ID operations and never acquires ACN model-sync vocabulary. Native
  occurrence and Package identities remain private from clients.
- Automatic chat loading, explicit loading, stopping, switching, installation, cancellation,
  recovery, and multi-client observation converge through the same ACN resources.
- Model-domain failures leave the same ACN process serving and never render daemon disconnection.
