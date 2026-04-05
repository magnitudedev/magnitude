# Magnitude Project Context

You are working on Magnitude, an AI coding agent platform.

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
bun session projection <id> Memory                  # replay events and dump named projection state as JSON
bun session projection <id> all --at 42            # all projections at point-in-time (after event 42)
```

Supported projections: `Memory`, `Fork`, `TaskGraph`, `Turn`, `Display`, `Compaction`, `WorkingState`, `SessionContext`, `Proposal`, `AgentRegistry`, `Artifact`, `ChatTitle`, `Replay`, `SkillMode`, `all`

Projection output is JSON — pipe to `jq` for querying. Events are 0-indexed.

`bun logs` — view CLI logger output for the current session.

## Testing

Run tests with `bunx --bun vitest` (not `bun vitest` — without `--bun`, vitest workers run under Node and Bun globals aren't available).

```bash
cd packages/agent && bunx --bun vitest run    # single run
cd packages/agent && bunx --bun vitest        # watch mode
```

## Effect Language Service

Use `bun els overview --file <path>` to list Effect exports (services, layers, errors) and `bun els layerinfo --file <path>` for layer dependency info. Example: `bun els overview --file packages/agent/src/index.ts`.
