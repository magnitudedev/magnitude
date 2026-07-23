---
applies_to:
  - packages/protocol/src/schemas/mirrored-state.ts
  - packages/protocol/src/rpcs/config.ts
  - packages/protocol/src/rpcs/local-inference.ts
  - packages/protocol/src/schemas/model-state.ts
  - packages/protocol/src/rpcs/group.ts
  - packages/acn/src/mirrored-state.ts
  - packages/acn/src/observed-state.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/local-model-packages.ts
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/local-models.ts
  - packages/acn/src/model-slot-coordinator.ts
  - packages/acn/src/local-inference-hardware.ts
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

ACN owns the public product mirrors: `ProviderModelCatalog`, `LocalModels`, `ModelSlots`, and
`LocalInferenceHardware`. `LocalModels` is the stable target-level product projection; package,
download-attempt, and recommendation working state remain private ACN observations. Private ICN
types and native field names do not cross the protocol boundary. A backend may bind directly only
when it owns the exact public schema and versioned replay.

Client-common owns one watch per client connection and all query invalidation. Query atoms remain
distinct by Get RPC tag. Screens consume snapshots; they do not reconstruct state or open their own
operation streams.
