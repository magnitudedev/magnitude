---
applies_to:
  - packages/protocol/src/schemas/mirrored-state.ts
  - packages/protocol/src/rpcs/config.ts
  - packages/protocol/src/rpcs/local-inference.ts
  - packages/protocol/src/schemas/model-state.ts
  - packages/protocol/src/rpcs/group.ts
  - packages/acn/src/mirrored-state.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/local-model-inventory.ts
  - packages/acn/src/model-slot-coordinator.ts
  - packages/acn/src/local-inference-hardware.ts
  - packages/acn/src/model-configuration.ts
  - packages/acn/src/handlers.ts
  - packages/acn/src/server.ts
  - packages/client-common/src/hooks/use-mirrored-state.ts
  - packages/client-common/src/hooks/use-model-config.ts
  - packages/client-common/src/hooks/use-slot-profiles.ts
  - packages/client-common/src/hooks/use-settings-state.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - packages/agent/src/ambient/config-ambient.ts
  - packages/agent/src/coding-agent.ts
---

# Mirrored state

A mirrored state is authoritative backend state exposed to clients as a versioned snapshot plus an
invalidation-only watch stream. The snapshot is the source of truth. Watch events contain only the
new revision and tell clients to refetch; they are not an event log and clients must not attempt to
reconstruct state from them.

Each mirror has one shared protocol definition containing its typed Get RPC, state schema, and
error schema. The Get RPC tag is the mirror's sole identity and its client reactivity key; parallel
keys and endpoint-name configuration are forbidden. Protocol declarations, ACN state, and client
hooks consume the same definition so endpoint names, result types, and invalidation identities
cannot drift independently.

The encoded side of every mirror schema must be JSON-safe. Every optional domain value uses
`Schema.optionalWith(valueSchema, { as: "Option", exact: true })`. This preserves `Option` in the
decoded domain and omits `None` fields from the encoded object. `OptionFromSelf`, `OptionFromNullOr`,
bare `Schema.optional`, and `UndefinedOr` are forbidden for optional domain values.

All mirrors publish tagged invalidations through one shared watch RPC. Opening that watch requires
no mirror-specific payload. Each changed event identifies the Get RPC tag and its new revision.
Heartbeats and reconnects cause clients to invalidate every mirror they currently consume, so a
dropped coalesced event cannot leave a mounted query permanently stale.

ACN serializes state transitions and revision increments. A transition that changes state publishes
one invalidation after the new snapshot is stored. A no-op transition changes neither the revision
nor the stream. Invalidation delivery is coalescing and bounded: a slow watcher may skip intermediate
revisions because its next action is always to fetch the latest complete snapshot.

When a backend service already owns the exact public schema as a versioned replaying source, ACN
may bind it directly. Private ICN sources do not qualify: ACN projects them into four product-owned
resources—`ProviderModelCatalog`, `LocalModelInventory`, `ModelSlots`, and
`LocalInferenceHardware`—whose owners assign product revisions. Raw ICN hardware, inventory,
recipes, residency, generated unions, and native field names never cross the protocol boundary.

Client-common owns one resident mirrored-state watch per client connection and all query
invalidation. Screens consume query-backed state and do not open their own progress streams or make
operation state local to the component that started the work.

Long-running mutation functions consume generated ICN streams through their terminal event.
Download progress updates the addressed `LocalModelInventoryEntry`; native load progress updates
every `ModelSlot` selecting the same local model. Commands call their domain owner directly. The
domain function itself is idempotent with respect to authoritative state; there is no admission
cache, mutation request identity, operation collection, history, latest-operation selection, or
command stream. Unary mutation pending state must never be used as the source of model, hardware,
or page-wide state.

A failed load terminal event publishes `Blocked(LocalModelLoadFailed)` with the native code,
message, and retryability and fails the command through its typed channel. A failed unload request
does not invent a blocked or unloaded slot: the last authoritative residency remains visible until
native inventory reports a real transition. Download failures always leave `Downloading` through
the entry FSM, and retrying an already active download never resets observed progress.

The agent-facing model configuration remains an agent-owned ambient contract. ACN derives that
existing shape from the catalog, slots, and stored context-limit policy in one stream at the
composition boundary. The agent composition root is the sole publisher into the ambient; workers
do not query ACN state, and ACN does not retain a second agent-configuration snapshot. Every
selected slot state remains callable at this boundary because ordinary provider completion demand
owns readiness and serialization; slot lifecycle state is observation, not client-side admission.
Only an unassigned slot or a selection missing from the provider catalog is unavailable to the
agent contract.

Durable model configuration also owns a bounded per-slot recency list for local provider model
identities. Starting a local provider request moves that model to the front of the addressed slot's
list. If a selected local model ceases to be selectable, the slot coordinator selects and persists
the first still-selectable model in that slot's recency order, regardless of whether it is currently
resident. Recency is policy input, not a second slot selection or a client mirror; a valid current
selection is never replaced merely because another model is more recent.

An owning service may continuously sample volatile observational input. Such a sampler updates only
its own source: an ICN hardware tick refreshes the exact hardware snapshot and does not list models,
rebuild recommendations, or rewrite inventory. Structurally unchanged samples are no-ops. A failed
sample retains the last good value and retries; it does not create a user-visible failure state or
erase unrelated authoritative fields. The sampler's lifetime belongs to the source service scope
and its activity does not become ACN mirror logic or a client-owned polling loop.
