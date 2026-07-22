---
applies_to:
  - packages/acn/src/activity-tracker.ts
  - packages/acn/src/resource-use-gate.ts
  - packages/acn/src/acn-shutdown.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/acn-subscription-protocol.ts
  - packages/acn/src/agent-runtime.ts
  - packages/acn/src/session-commands.ts
  - packages/acn/src/session-lifecycle.ts
  - packages/acn/src/session-drafts.ts
  - packages/acn/src/active-session-statuses.ts
  - packages/acn/src/display-view-streams.ts
  - packages/acn/src/server.ts
  - packages/protocol/src/rpcs/**
  - packages/protocol/src/schemas/subscription.ts
  - packages/sdk/src/jit-rpc/**
  - packages/sdk/src/acn-jit/**
  - packages/client-common/src/**
  - packages/agent/src/session-work-status.ts
  - packages/agent/src/coding-agent.ts
  - packages/agent/src/compaction/worker.ts
  - packages/agent/src/events.ts
  - packages/agent/src/index.ts
  - packages/agent/src/execution/execution-manager.ts
  - packages/agent/src/execution/types.ts
  - packages/agent/src/process/detached-process-registry-live.ts
  - packages/agent/src/process/detached-process-registry.ts
  - packages/agent/src/projections/compaction.ts
  - packages/agent/tests/session-work-status.test.ts
  - cli/src/**
  - desktop/src/**
  - web/src/**
---

# ACN and session lifetime

ACN and session runtimes stay alive while they are doing work, then shut down independently after
an idle period.

## ACN lifetime

ACN shuts down after 30 minutes without work. The first idle period begins only after ACN, its HTTP
server, registration, and private ICN are ready.

Finite RPCs keep ACN alive for the full request. Work that continues after a request—such as an
agent turn or model operation—keeps its own claim until it finishes.

Observation does not keep ACN alive. This includes health checks, subscriptions, status and file
watches, mirrored-state refresh, display streams, ICN observation, telemetry, and introspection.

All stop causes use the same shutdown path: stop accepting work, close the application, terminate
and reap ICN, then release machine ownership. Closing the application closes resident session scopes
directly and cancels their work; ACN shutdown never waits for a session idle-retirement gate whose
leases are owned by those same scopes.

## Session runtime lifetime

A session runtime unloads after two minutes without session work. Commands, agent execution,
display materialization, shape changes, resync, and preload count as work. Merely watching a session
does not.

Agent work has one authoritative status covering turns, queued triggers, workers, compaction, and
owned detached processes. Both UI status and runtime lifetime use that status.

Session startup is single-flight. Unloading closes the current runtime before publishing it as
absent; later work creates a new runtime. A draft stores session intent, not a runtime. Deletion
blocks new work, waits for current work to finish, closes the runtime, then deletes durable state.

## Display streams during session unload

The display subscription belongs to ACN, while its live display attachment belongs to the current
session runtime. An open display subscription therefore does not keep the session loaded.

When the session unloads, ACN detaches the live display and tells the subscription it is suspended.
The client keeps the last display state. A later materialization, shape change, or resync reloads the
session, reattaches the display, and sends a full snapshot.

Closing the final subscription removes the display registration. There is no separate close RPC.

## Subscription protocol

ACN subscriptions wrap domain values in a small transport protocol:

| Frame | Meaning |
| --- | --- |
| `payload` | Domain value |
| `keepalive` | Connection is alive |
| `suspended` | Session runtime unloaded; subscription stays open |
| `terminated` | ACN is shutting down |

Framing is handled below RPC handlers and client consumers. Invalid frames, or a stream ending
without `terminated`, are protocol errors rather than evidence that ACN died.

## Concurrency guarantees

Work is represented by scoped claims tied to a specific ACN or session-runtime generation. The last
claim starts that generation's idle timer. Starting work and committing shutdown are serialized, so
work cannot be admitted into a generation that is being destroyed. Stale or duplicate cleanup from
an older generation cannot affect its replacement.
