TRESTING 
# Magnitude Project Context

You are working on Magnitude, an AI coding agent platform.

## Package Layering

```
clients (cli/web) → client-common → sdk → acn (daemon)
```

- **Clients** import only from `@magnitudedev/client-common` and `@magnitudedev/sdk`. Never from `acn`, `agent`, `acn-protocol`, `ai`, or `providers` directly.
- **client-common** — shared state, hooks, display sync. Uses one connection-scoped Effect Query `AgentClient` over the shared SDK instance.
- **sdk** — private, portable Effect RPC client, fixed-endpoint admission/recovery, and optional service-starter capability. Re-exports pure protocol contracts; no SQLite, process supervision, or query cache.
- **daemon-management** — private desktop discovery/control, native child supervision, login registration. Only privileged host composition roots (CLI bootstrap, desktop main, development server) import it. The desktop bundles and owns ACN; clients never acquire or elect a standalone daemon.
- **client-common** owns first-party Query/Mutation/Subscription definitions over SDK operations; `providers` owns provider-client construction.
- **acn** — server daemon hosting agent runtime, sessions, file ops, display streams. Implements ACN protocol RPCs.
- **acn-protocol** — wire contract (RPCs, schemas) shared by SDK and ACN. Not imported by clients.
- **ai** — provider-agnostic contract (`Provider`, `ModelCatalog`, `BoundModel`, `BaseCallOptions`).
- **providers** — concrete provider implementations + registry. See `packages/providers/AGENTS.md`.
- **agent** — agent runtime, projections, workers, tools, display materialization.
- **event-core** — event sourcing, projections, addressed state.
- **roles** — role/slot definitions for worker specialization.
- **storage** — persistent session/config/auth storage.

For a new backend operation, declare its `Rpc.make` once in the appropriate `packages/acn-protocol/src/boundary/` domain, implement its handler in ACN, and consume the derived SDK method. Add first-party caching/synchronization policy in `packages/client-common/src/operations/` as needed. Effect Query does not generate RPCs. The SDK may ask an injected starter to ensure availability; only daemon-management owns native application control. Plugin packages bundle the private SDK and inherit its exact RPC-version check.

Finite RPC declarations must apply `replaySafe` or `atMostOnce`; the RPC tree requires a declared
replay policy. Release preparation, Changesets orchestration, and protocol/plugin version allocation
belong in `packages/release`; `packages/version` only generates build identity.

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

## Running checks and measurements

Checks and measurements run on hardware through `mrig` (`uv run --project inference/evaluation mrig …`). `mrig --help` lists what each instrument answers. Run each one as a single background command; you are notified when it ends.

Runs are the slowest part of the work, and you don't notice their time passing, so a long run feels free when it isn't. Before every run, go through these steps:

1. **Decide what the result will change.** Write down what you expect it to show and what you will do if it shows something else. If you would do the same thing either way, skip the run. If reading the code can answer the question, read the code instead.
2. **Pick the cheapest check that can decide it.** In order of cost: read the code, compile, run a unit test, test or time the kernel at its real shape, and only then run a whole model. A kernel timed at its shape answers in seconds what a model run answers in many minutes, with less noise. Prefer an exact check (identical bits, a count, a value derived from the code) to a statistical comparison, which needs many samples and still leaves doubt.
3. **Check only what the change can affect.** Work out from the diff which code paths and shapes it reaches, and check one example of each. Don't add other models, contexts or backends: each one adds time and another chance of a failure unrelated to your change.
4. **Know how long it should take, and how small an effect it can see.** Pass that duration as `--estimate`. If the run takes longer, your picture of the system is wrong, so find out why before you run anything else. If the effect you expect is smaller than the run's noise, the result tells you nothing, so choose a finer check.
5. **Run once, on the finished change.** A run measures the code as it was when the run started, so finish editing first. If a run fails, read the failure before running anything again.

While a run is in progress, keep working on whatever doesn't depend on it, and put independent runs on different hosts. If no instrument answers your question, write the check you need (a test, or a timing at the shape) rather than using a broad, slow one; it pays for itself the next time.

## Type Checking

Run targeted type checks per package — do not run project-wide `tsc -b`

##  Project Documents

When significant bug is being reported or a large spec is being created, place under `bugs/YY-MM-DD/` or `specs/YY-MM-DD/`.

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

Always use the effect DI system where appropriate. Break abstractions into services.
Services should be a Context.Tag, with an interface of the same name as the tag.

### Effect Schemas

For any data that must be serializable, introspected, or validated, it must be represented as an Effect Schema.
Any optional values must use `Schema.optionalWith(Schema.String, { as: 'Option', exact: true })` to ensure that (1) these values serialize to exactly existing or not existing (`undefined` is not serializable) and (2) so that the idiomatic Option is used in the decoded side.

### Branded types

Use Effect branded types for any string values that have semantic attribution to a particular type of ID or other value.
