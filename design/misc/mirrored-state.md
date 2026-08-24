---
applies_to:
  - packages/acn-protocol/src/schemas/mirrored-state.ts
  - packages/acn-protocol/src/boundary/configuration.ts
  - packages/sdk/src/inference.ts
  - packages/acn-protocol/src/boundary/onboarding.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/acn/src/mirrored-state.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/local-models.ts
  - packages/acn/src/model-slot-controller.ts
  - packages/acn/src/onboarding/**
  - packages/client-common/src/hooks/use-local-inference-state.ts
---

# Mirrored state

A mirror is an ACN-owned versioned authoritative snapshot whose commits publish change pokes. To
clients it is an ordinary contract query with infinite freshness; "mirror" is not a client concept.

## Definition and identity

The boundary query (`Query.make("GetModelSlots", …)` with `staleTime: Infinity`, `gcTime: Infinity`)
owns the snapshot schema `{ revision, state }`; its name is the sole identity a change poke carries
(`Change.query`). The ACN's versioned state is constructed with that query name. Encoded schemas
are JSON-safe.

## Updates

State and revision commit atomically. A semantic change increments revision once, stores the new
snapshot, then publishes `{ query, revision }` on the ACN change registry. A no-op publishes
nothing. The registry stream is bounded and coalescing, so intermediate revisions may be skipped.
Reconnection invalidates every consumed query.

## Ownership

ACN owns the public product mirrors: `GetProviderModelCatalog`, `GetLocalModels`, `GetModelSlots`,
and `GetOnboardingState`. `LocalModels` groups by canonical model identity and publishes
acquisition, serving, recommendation, provider availability, and advisory
memory facets on the same row. Every catalog bundle and independently servable installed package
contributes the same `LocalModel` shape. Raw package and package-attempt state, model-download
records, recommendation-policy, and provider-offering working state remain private ACN
observations. Hardware, Models, Packages, Downloads, and Instances are native ICN Queries consumed
through the same connection-scoped Effect Query cache; ACN does not mirror them. The complete
Magnitude-specific local-model projection is defined by
[Local-model product projection](../model-management/local-model-product-projection.md).
A backend may bind directly only when it owns the exact public schema and versioned replay.

`ModelSlots` also carries the provider-qualified model preferences needed to present model
selection, including favorites and recency. Preference mutations durably commit before the mirror
publishes the new snapshot.

The connection's Effect Query client drains `StreamChanges` once and invalidates the named query;
no domain code owns invalidation. Query atoms remain distinct by query name, and a domain has one
canonical client query cache.

The same client drains ICN's multiplexed `/api/v1/events` stream and invalidates native inference
Queries. This is a second observation transport inside the same cache, not a second cache or mirror.

Clients retain each query's waiting, failure, and success Result independently. Screens may derive
presentation from successful domain values; they do not combine domain Results into an aggregate
authority or reconstruct state. An initial snapshot failure does not terminate the change
subscription. A later poke or reconnection retries the same canonical query.

## Client retention and rendering

One query atom exists for each mirror and client connection. Public mirror query values remain
resident until that client connection's registry is disposed; component, menu, gate, and route
unmounts do not clear the latest Result. This is retained query cache, not copied server state.
Invalidation and reconnect reread ACN into the same query authority.

```text
ACN mirror -> connection-resident query atom -> Result -> pure presentation
```

Clients interpret Results without inventing domain state:

| Observation | Presentation |
| --- | --- |
| No complete value yet | Loading |
| Refresh with a prior success | Prior value with refresh status |
| Failure with a prior success | Prior value with degraded status |
| Complete success with an empty collection | Empty |
| Complete success with entries | Entries |

Waiting, refresh, failure, and connection loss never become an empty authoritative collection.
Independent mirrors remain independently renderable: failure or waiting in catalog, slot, hardware,
or recommendation observation cannot erase a successful `LocalModels` value.

Client surfaces subscribe through read-only selector atoms when they consume only part of a mirror.
A selector may retain its prior result reference while its newly derived value is semantically
equivalent. This is an equality cache over the same query atom, not copied state: it has no writer,
authority, synchronization, or independent lifetime.

A mirrored nonterminal state is valid only while its owning backend service has a live operation
capable of terminalizing it. The initiating RPC and its progress stream are never the owner.
Disconnecting every client does not alter admitted shared work; a later client receives the same
authoritative current snapshot.

## Conformance

- A semantic no-op increments no revision and publishes no invalidation.
- Reconnect and invalidation always converge by rereading the authoritative snapshot.
- Public query Results survive component unmount for the client-connection lifetime.
- Only a complete successful empty snapshot renders authoritative emptiness.
- One mirror's observation lifecycle cannot erase another mirror's successful value.
- Selector atoms preserve one mirror authority while preventing unrelated source-field changes from
  invalidating a complete presentation surface.
