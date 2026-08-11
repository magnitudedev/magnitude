---
applies_to:
  - packages/acn-protocol/src/schemas/mirrored-state.ts
  - packages/acn-protocol/src/rpcs/config.ts
  - packages/acn-protocol/src/rpcs/local-inference.ts
  - packages/acn-protocol/src/rpcs/onboarding.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/acn-protocol/src/rpcs/group.ts
  - packages/acn/src/mirrored-state.ts
  - packages/acn/src/observed-state.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/local-model-packages.ts
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/local-models.ts
  - packages/acn/src/model-slot-controller.ts
  - packages/acn/src/local-inference-hardware.ts
  - packages/acn/src/onboarding/**
  - packages/acn/src/handlers.ts
  - packages/acn/src/server.ts
  - packages/client-common/src/hooks/use-mirrored-state.ts
  - packages/client-common/src/hooks/use-model-config.ts
  - packages/client-common/src/hooks/use-slot-profiles.ts
  - packages/client-common/src/hooks/use-settings-state.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
---

# Mirrored state

A mirror is a versioned authoritative backend snapshot plus an invalidation-only watch. Watch events
are not an event log; clients refetch the latest snapshot.

## Definition and identity

One definition owns the state schema, error schema, and typed Get RPC. The Get RPC tag is the sole
mirror identity and client reactivity key. Encoded schemas are JSON-safe.

## Updates

State and revision commit atomically. A semantic change increments revision once, stores the new
snapshot, then publishes `{ Get-RPC tag, revision }`. A no-op publishes nothing.

The shared watch is bounded and coalescing, so intermediate revisions may be skipped. Subscription
keepalives are consumed below the domain stream. Initial connection and reconnection invalidate all
currently consumed mirrors.

## Ownership

ACN owns the public product mirrors: `ProviderModelCatalog`, `LocalModels`, `ModelSlots`,
`LocalInferenceHardware`, and `Onboarding`. `LocalModels` groups by servable-bundle identity while
preserving every catalog, retained, and assessment-derived configuration. Every independently
servable installed package contributes the same `LocalModel` shape, regardless of catalog
association. Raw package, download-attempt, and recommendation working state remain private ACN
observations. Private ICN types, native paths, and native field names do not cross the protocol
boundary. The complete local-model projection is defined by
[Local-model product projection](../model-management/local-model-product-projection.md).
A backend may bind directly only when it owns the exact public schema and versioned replay.

`ModelSlots` also carries the provider-qualified model preferences needed to present model
selection, including favorites and recency. Preference mutations durably commit before the mirror
publishes the new snapshot.

Client-common owns one watch per client connection and all query invalidation. Query atoms remain
distinct by Get RPC tag, and clients retain each query's waiting, failure, and success Result
independently. Screens may derive presentation from successful domain values; they do not combine
domain Results into an aggregate authority, reconstruct state, or open their own operation streams.

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
