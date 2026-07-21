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
  - packages/acn/src/local-inference/service.ts
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

All mirrors publish tagged invalidations through one shared watch RPC. Opening that watch requires
no mirror-specific payload. Each changed event identifies the Get RPC tag and its new revision.
Heartbeats and reconnects cause clients to invalidate every mirror they currently consume, so a
dropped coalesced event cannot leave a mounted query permanently stale.

ACN serializes state transitions and revision increments. A transition that changes state publishes
one invalidation after the new snapshot is stored. A no-op transition changes neither the revision
nor the stream. Invalidation delivery is coalescing and bounded: a slow watcher may skip intermediate
revisions because its next action is always to fetch the latest complete snapshot.

Client-common owns one resident mirrored-state watch per client connection and all query
invalidation. Screens consume query-backed state and do not open their own progress streams or make
operation state local to the component that started the work.

Long-running commands acknowledge after validation and state acceptance, then continue in a
service-owned scope. Their progress and terminal result live in the applicable mirrored state. A
client-provided request ID makes retries idempotent, while domain target identity coalesces duplicate
active work. Unary mutation pending state describes only command acceptance and must never be used as
a page-wide busy flag.

An owning service may sample volatile observational input into one field of an existing mirror.
Such a sampler must perform a narrow source-owned transition: a local-inference hardware tick, for
example, may fetch hardware and replace the host field but must not list models, rebuild
recommendations, or rewrite operations. Structurally unchanged samples are no-ops. A failed sample
silently retains the last good value and retries; it does not create a user-visible failure state or
erase unrelated authoritative fields. The sampler's lifetime belongs to the service scope and its
activity does not become a client-owned polling loop.
