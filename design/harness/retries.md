---
applies_to:
  - packages/ai/src/errors/**
  - packages/agent/src/errors/**
  - packages/agent/src/execution/**
  - packages/agent/src/util/retry-backoff.ts
  - packages/agent/src/workers/turn-executor.ts
  - packages/harness/src/**
---

# Turn execution retries

## Model

A **turn** is one logical agent execution with one eventual outcome.

An **attempt** is one ephemeral execution attempt made while completing that turn. A turn may make
multiple attempts, but attempts are not turns and have no durable lifecycle.

```text
turn starts
    |
    +-- attempt fails transiently -> backoff -> attempt again
    |
    `-- attempt succeeds or terminates
              |
              v
       one turn outcome
```

The turn remains active throughout attempts and backoff. Only the eventual outcome ends the turn.

## Guarantees

- `TurnExecutor` owns the entire retry loop.
- Retry count, intermediate failures, and remaining delay exist only in the active executor fiber.
- Intermediate failures do not publish turn outcomes or create events, continuations, triggers, or
  projection state.
- Exactly one eventual `turn_outcome` is published for the turn.
- Backoff and active attempts are interrupted with the turn.
- Failed-attempt assistant output does not enter the turn's committed context.
- Tool execution is never repeated automatically.
- Different turns and forks retry independently.

## Ownership

| Owner | Responsibility |
| --- | --- |
| AI | Classify provider failures and preserve `Retry-After` evidence |
| Harness execution | Run one attempt and report output and tool-execution evidence |
| `TurnExecutor` | Decide, count, delay, retry, interrupt, and publish the eventual turn outcome |

Turn projections, turn initiation, hydration, and scheduling have no retry responsibility. They do
not observe attempts or retry timing.

## Retry policy

Only transient upstream connection failures are automatically retried. Authentication failures,
invalid requests, context-window rejection, unavailable model selection, safety termination, and
correctness defects terminate the retry loop immediately with their normal turn result.

The policy permits five retries after the initial attempt. For zero-based retry index `n`:

```text
backoff = min(500ms * 2^n, 30s)
delay   = max(backoff, provider Retry-After)
```

When the retry budget is exhausted, the executor returns the existing terminal connection-failure
result. Exhaustion does not introduce a new event or lifecycle.

## Execution

```text
TurnExecutor
    |
    `-- loop
          |
          +-- run one attempt
          |
          +-- success --------------------------> publish final outcome
          |
          +-- non-retryable failure ------------> publish final outcome
          |
          +-- tool execution began -------------> publish final outcome
          |
          +-- retry budget exhausted -----------> publish final outcome
          |
          `-- retryable connection failure
                    |
                    +-- discard failed-attempt output
                    +-- interruptible backoff
                    `-- continue loop
```

The retry loop does not leave `TurnExecutor`. It does not end and restart the turn, and it does not
ask event-core to schedule another attempt.

## Attempt isolation

Assistant output is provisional until its attempt succeeds. Output from a failed retryable attempt
may be shown as transient progress, but it is excluded from the durable turn result and future model
context.

Each retry starts from the same committed turn input. Failed-attempt output is never added to the
next attempt's prompt.

## Tool-effect safety

Once any tool execution begins, the turn is no longer safe to restart automatically. A later
connection failure terminates the retry loop and produces the normal non-retrying turn result.

This boundary applies when execution starts, not when a tool result is received, because an
external effect may already have occurred. Tools requiring exactly-once behavior must additionally
provide idempotency or transactional execution.

## Interruption and process loss

The executor fiber owns both the active attempt and its backoff. Interrupting the turn interrupts
whichever phase is active and produces the normal cancellation outcome. No detached timer or worker
can restart it later.

Attempts, counts, and delays are not persisted. If the process exits during an attempt or backoff,
that retry state is lost. General unfinished-turn recovery determines what happens to the turn;
retry does not add replay or hydration behavior.

Hydration performs no retry work, sleep, provider call, or tool call.

## Exclusions

- No retry controller or retry worker.
- No retry events, attempt events, persisted counters, or persisted deadlines.
- No retry continuation or retry-specific turn trigger.
- No `wake` event or trigger.
- No projection-owned clock, timer, eligibility check, or retry state.
- No retry behavior in `TurnInitiator`.
- No change to the public `turn_outcome` contract.
- No retry-driven redesign of ordinary continuation, context disposition, or turn resolution.
- No shared retry lifecycle for compaction or unrelated operations.

Other operations may share backoff mathematics, but they own their own retry behavior according to
their own execution and durability boundaries.

## Conformance

- A transient pre-stream or streamed connection failure retries without ending the turn.
- Multiple failed attempts produce one eventual turn outcome, not one outcome per attempt.
- Backoff progression, the retry limit, and `Retry-After` are enforced inside `TurnExecutor`.
- Failed-attempt output is absent from the committed turn and the next attempt's prompt.
- Tool execution prevents any later automatic restart of the turn.
- Interruption during an attempt or backoff produces cancellation and no later retry.
- Replay and hydration contain no retry effects or retry state.
- Event-core contains no retry scheduling path and no `wake` event or trigger.
