# Observer Escalation System

## Overview

The observer is a dedicated monitoring thread that evaluates leader turns for signs of struggle, tunneling, or strategy decay. When it detects problems on the leader, it queues an advisor-required communication for the next root turn. Worker observation is currently disabled in `worker.ts`; the parent-communication path remains in projections for when it is re-enabled.

## Architecture

```
                  turn_outcome (forkId=null)
                         │
                    ┌────▼────┐
                    │ Observer │  (per-fork, at most one running)
                    │   LLM   │
                    └────┬────┘
                         │
                  pass/escalate tool call
                         │
                    ┌────▼─────────────────┐
                    │  escalate=false?     │──→ observer_outcome published, done
                    │  escalate=true?     │
                    │  forkId !== null?    │──→ worker escalation (communication to parent)
                    │  forkId === null?    │──→ advisor-required root communication
                    └──────────────────────┘
```

### Worker escalation (forkId !== null)

Observer publishes `observer_outcome` with `escalate=true`. TurnProjection enqueues a communication on the parent fork. The parent agent sees the escalation in its inbox and can intervene (kill worker, reassign, send message).

Worker observation is disabled today, so this path is dormant.

### Leader escalation (forkId === null)

Observer publishes `observer_outcome` with `escalate=true`.

TurnProjection handles that event by appending a root inbound communication marked `requiresAdvisor: true` and ensuring a communication trigger exists. The next root `turn_started` is the single claim point: it consumes pending inbound communications and records `requiresAdvisor` on that exact active turn. Cortex reads TurnProjection for the active turn and requires `message_advisor` only when that same turn claimed the requirement.

Observer scheduling also reads TurnProjection. While a root advisor requirement is still pending, observer evaluation is deferred so an unrelated observer outcome cannot clear or race the requirement.

## Event Flow

```
turn_outcome ──→ ObserverWorker ──→ observeOnce()
                                        │
                                  observer LLM
                                        │
                                  observer_outcome (published)
                                        │
                                  ┌─────┴───────┐
                                  │ escalate?    │
                                  │ forkId=null? │
                                  └─────┬───────┘
                                        │ yes
                                  TurnProjection.observer_outcome handler
                                        │
                                  append requiresAdvisor communication
                                  + ensure communication trigger
                                        │
                                  TurnController sees idle fork + due trigger
                                        │
                                  turn_started claims pending communication
                                        │
                                  Cortex requires message_advisor
                                        │
                                  turn_outcome ──→ GOTO top
```

## Concurrency Model

### Per-fork serialization

At most one observer evaluation runs per fork at a time. The observer loop (`observerLoop` in `worker.ts`) processes events sequentially:

1. Run `observeOnce` for the current event
2. Check if a new `turn_outcome` arrived while running → if so, re-evaluate with the latest
3. Continue until no pending event remains

If the observer is already running for a fork when a new `turn_outcome` arrives, the event is stored as `pendingEvent` (replacing any previous pending event). The loop picks it up after the current evaluation finishes.

### Advisor-required deferral

There is no separate observer-owned notification gate. TurnProjection is the source of truth:

- Pending root inbound communication with `requiresAdvisor: true` means the leader has not seen the escalation yet.
- Root `turn_started` atomically claims that requirement and records it on the active turn.
- Cortex requires `message_advisor` only for the claimed active turn.
- ObserverWorker defers root evaluation while TurnProjection still has a pending advisor requirement.

```
turn_outcome arrives
      │
 ┌────▼──────────────┐
 │ Advisor required?│
 │     yes           │──→ do not start observer
 │     no            │──→ normal ObserverRun logic
 └───────────────────┘
      │
 ┌────▼─────────────┐
 │ observer running? │
 │     yes           │──→ store as pendingEvent
 │     no            │──→ start observer loop
 └──────────────────┘

observer loop iteration:
      │
 ┌────▼──────────────┐
 │ Advisor required?│
 │     yes           │──→ mark observer idle and exit
 │     no            │──→ run observeOnce()
 └───────────────────┘
```

This structurally prevents:
- **Observer escalation races:** Unrelated turn outcomes cannot clear a pending advisor requirement.
- **Missed force options:** Only the root turn that claims the requirement receives the required `message_advisor` call.
- **Idle stranding:** Escalation enqueues a communication trigger, including when root was waiting for user.

## Justifications

Escalation uses a single justification string, defined in `justifications.ts`:

- `difficulty`
- `churn`
- `frustration`

The tool schema in `schema.ts` and message templates in `justifications.ts` derive from that set.

## File Index

| File | Purpose |
|------|---------|
| `index.ts` | Public exports |
| `worker.ts` | ObserverWorker definition — event handlers, observer loop, observeOnce, advisor-required deferral |
| `schema.ts` | Harness toolkit schemas (`pass` / `escalate`) |
| `justifications.ts` | Justification type and leader-facing escalation message templates |
| `prompt.ts` | Observer prompt builder — maps ForkWindowState to observer LLM prompt |
| `state.ts` | ObserverStateTag — Effect Ref with per-fork run state (fiber, pendingEvent) |
| `types.ts` | Event types (ObserverOutcome) |
| `render.ts` | Shared observer_turn rendering for inbox XML |

## Observer Prompt

The observer sees a compact version of the agent's context window:

- **Time markers** rendered with the same minute/day boundary logic as the main inbox timeline
- **User messages** rendered as `<user>...</user>`, with adjacent same-minute entries coalesced under `<entry>` children
- **Agent turns** rendered as `<magnitude>` blocks with thoughts, tool work, visible message text, and feedback
- **Tool work** rendered generically as `<tool_name><params>...</params><result|error|interrupted/></tool_name>`
- **Prior observer turns** rendered as real tool call history (AssistantMessage + ToolResultMessage pairs)
- **Compacted history** included as system entries; session and fork context wrappers are omitted from the observer transcript

The observer must call exactly one tool: `pass` or `escalate`. No prose output.

## Lifecycle

The observer worker runs for the entire session lifetime. It:

1. Listens for `turn_outcome` events on all forks
2. For each fork, maintains a single observer loop fiber
3. Cancels observer fibers when agents are killed or go idle
4. Publishes `observer_outcome` after each evaluation

The observer model is resolved via `AgentModelResolver.resolveObserver()` — typically a fast/cheap model since it runs after every turn.
