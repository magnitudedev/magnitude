# Magnitude Project Context

You are working on Magnitude, an AI coding agent platform.

## Package Layering

```
clients (cli/web) → client-common → sdk → acn (daemon)
```

- **Clients** import only from `@magnitudedev/client-common` and `@magnitudedev/sdk`. Never from `acn`, `agent`, `protocol`, `ai`, or `providers` directly.
- **client-common** — shared state, hooks, display sync. Uses `AgentClient` (AtomRpc over `MagnitudeRpcs`).
- **sdk** — typed RPC client, daemon lifecycle (`DaemonSpawner`), `ProviderClient`, binary resolution. Re-exports protocol types.
- **acn** — server daemon hosting agent runtime, sessions, file ops, display streams. Implements protocol RPCs.
- **protocol** — wire contract (RPCs, schemas). Shared by SDK and ACN. Not imported by clients.
- **ai** — provider-agnostic contract (`Provider`, `ModelCatalog`, `BoundModel`, `BaseCallOptions`).
- **providers** — concrete provider implementations + registry. See `packages/providers/AGENTS.md`.
- **agent** — agent runtime, projections, workers, tools, display materialization.
- **event-core** — event sourcing, projections, addressed state.
- **roles** — role/slot definitions for worker specialization.
- **storage** — persistent session/config/auth storage.

If a client needs something not in the SDK/ACN: add an RPC to `packages/protocol/src/rpcs/`, wire into `MagnitudeRpcs` in `group.ts`, implement handler in `packages/acn/src/handlers.ts`. SDK exposes it automatically. Add a `client-common` hook if shared.

## Session Inspection

Use `bun session` to inspect past sessions. Sessions are stored in `~/.magnitude/sessions/` with UTC timestamp folder names.

```bash
bun session list                                    # list recent sessions (ID, title, date, messages)
bun session events <id>                             # list all events with index, type, timestamp (no payloads)
bun session events <id> --type turn_started,user_message  # filter by event type(s)
bun session events <id> --from 10 --to 50          # slice by index range
bun session event <id> <index>                      # show one event's full payload as JSON
bun session search <keyword> <id>                   # search event payloads for keyword, shows index/type/snippet
bun session search <keyword> --last 5              # search across last 5 sessions
bun session projection <id> Window                  # replay events and dump named projection state as JSON
bun session projection <id> all --at 42            # all projections at point-in-time (after event 42)
```

Supported projections: `Window`, `Fork`, `TaskGraph`, `Turn`, `Display`, `Compaction`, `WorkingState`, `SessionContext`, `Proposal`, `AgentRegistry`, `Artifact`, `ChatTitle`, `Replay`, `all`

Projection output is JSON — pipe to `jq` for querying. Events are 0-indexed.

`bun logs` — view CLI logger output for the current session.

## Testing

Run tests with `bunx --bun vitest` (not `bun vitest` — without `--bun`, vitest workers run under Node and Bun globals aren't available).

```bash
cd packages/agent && bunx --bun vitest run    # single run
cd packages/agent && bunx --bun vitest        # watch mode
```

## Type Checking

Run targeted type checks per package — do not run project-wide `tsc -b`

##  Project Documents

When significant bug is being reported or a large spec is being created, place under `bugs/YY-MM-DD/` or `specs/YY-MM-DD/`.

## Design Documents

`design/` contains the durable source of truth for architecture and behavior. Follow `design/AGENTS.md`: use `bun design-docs` to find applicable documents, preserve their guarantees, and update them and their applicability whenever the design or ownership changes.

```bash
bun design-docs inference/crates/icn-engine/src/scheduler.rs
bun design-docs --changed --explain
```

## Info Docs

Use `info/` for concise, high-level Markdown documents for humans and LLMs. These docs should describe architecture, systems, expected behavior, and durable project context without depending on brittle file names or code snippets unless they are highly relevant.

## Motel Tracing

Motel is a local OpenTelemetry collector with an HTTP API at `http://127.0.0.1:27686`. Query it with `curl` to inspect traces, spans, and logs from the application.

Key endpoints:

```
GET /api/health                              liveness check
GET /api/services                            services reporting telemetry
GET /api/traces?service=<service>            recent traces for a service
GET /api/traces/<trace-id>                   full trace tree with spans
GET /api/spans/<span-id>                     single span + logs
GET /api/logs?service=<service>              recent logs
GET /api/traces/search?...                   structured trace search
GET /api/logs/search?...                     structured log search
GET /api/ai/calls                            AI SDK call inspector
```

Full OpenAPI spec at `http://127.0.0.1:27686/openapi.json`.

## Client State Patterns

When working with client-side state (CLI, web, or client-common), read `packages/client-common/AGENTS.md` for the reactive state policy — which patterns to use and when.

## Effect Language Service

Use `bun els overview --file <path>` to list Effect exports (services, layers, errors) and `bun els layerinfo --file <path>` for layer dependency info. Example: `bun els overview --file packages/agent/src/index.ts`.

# Engineering invariants

## Principles

The following general engineering principles should always be followed.

### Form meaningful abstractions

The only way to produce sustainable and understandable long-term code is to choose and maintain the correct abstractions.
This does not mean creating indirection or unnecessary abstractions. It means identify the key behaviors of the system, the pieces at play, and the optimal way in which those compose into clean abstractions that also follow code patterns and architecture idiomatically.

### Do not overengineer / no patchwork code

Do not engineer for cases that will not happen. Do not add complex solutions for problems that are solvable by stepping back and addressing the root problem precisely. Be wary of tacking on changes to other imperfect changes, as this is a clear indication of patchwork code. Be self-aware in such scenarios and refactor to cleaner and more meaningful abstractions.

### No unnecessary backwards compatibility

Unless the user explicitly asked for it, do not add backwards compatibility shims or retain "legacy" code. Such code is unnecessary and will only pollute the codebase unless it is part of an explicitly user-designed, justified mechanism.

## Effect usage

This codebase is Effect-TS native.
- New code must be Effect-TS native
- Any TS code that touches effect code must remain or become effectful
- Code may be freely effectified, and never the opposite

### Effect patterns

Follow established Effect conventions in the project, while referring to the following when relevant.

#### Effect DI

Always use the effect DI system where appropraite. Break abstractions into services.
Services should be a Context.Tag, with an interface of the same name as the tag.

### Effect Schemas

For any data that must be serializable, introspected, or validated, it must be represented as an Effect Schema.
Any optional values must use `Schema.optionalWith(Schema.String, { as: 'Option', exact: true })` to ensure that (1) these values serialize to exactly existing or not existing (`undefined` is not serializable) and (2) so that the idiomatic Option is used in the decoded side.

### Branded types

Use Effect branded types for any string values that have semantic attribution to a particular type of ID or other value.
