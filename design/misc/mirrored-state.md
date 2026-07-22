---
applies_to:
  - packages/protocol/src/schemas/mirrored-state.ts
  - packages/protocol/src/rpcs/config.ts
  - packages/protocol/src/rpcs/local-inference.ts
  - packages/protocol/src/rpcs/group.ts
  - packages/acn/src/mirrored-state.ts
  - packages/acn/src/account.ts
  - packages/acn/src/handlers.ts
  - packages/acn/src/server.ts
  - packages/acn/src/icn/mirrors.ts
  - packages/client-common/src/hooks/use-mirrored-state.ts
  - packages/client-common/src/hooks/use-model-config.ts
  - packages/client-common/src/hooks/use-slot-profiles.ts
  - packages/client-common/src/hooks/use-settings-state.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
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

When a backend service already owns a versioned replaying source, ACN binds that source directly.
The Get RPC reads the source snapshot and its change stream publishes invalidations with the same
revision. ACN does not copy the state into another `Ref`, assign a second revision, or run a second
equality check. ICN hardware, inventory, and recipe mirrors use this source-backed form.

Client-common owns one resident mirrored-state watch per client connection and all query
invalidation. Screens consume query-backed state and do not open their own progress streams or make
operation state local to the component that started the work.

Long-running commands acknowledge after validation and state acceptance, then continue in a
service-owned scope. Their progress and terminal result live in the applicable mirrored state. A
client-provided request ID makes retries idempotent, while domain target identity coalesces duplicate
active work. Unary mutation pending state describes only command acceptance and must never be used as
a page-wide busy flag.

An owning service may continuously sample volatile observational input. Such a sampler updates only
its own source: an ICN hardware tick refreshes the exact hardware snapshot and does not list models,
rebuild recommendations, or rewrite inventory. Structurally unchanged samples are no-ops. A failed
sample retains the last good value and retries; it does not create a user-visible failure state or
erase unrelated authoritative fields. The sampler's lifetime belongs to the source service scope
and its activity does not become ACN mirror logic or a client-owned polling loop.
