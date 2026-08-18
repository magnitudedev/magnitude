---
applies_to:
  - packages/ai/src/errors/**
  - packages/harness/src/**
  - packages/agent/src/execution/**
  - packages/agent/src/workers/**
  - packages/agent/src/compaction/**
  - packages/agent/src/projections/turn.ts
  - packages/agent/src/util/retry-backoff.ts
---

# Model request retries

Model request retry is an execution-local reliability policy. It retries one provider request
without turning that retry into another agent turn, a durable workflow, or event-sourced timing
state.

## Boundary

```text
TurnExecutor / CompactionWorker / other operation owner
                         |
                         v
                      Harness
                         |
                         +--> completed tool effects
                         |
                         `--> model request runner
                                  |
                                  +--> provider attempt
                                  +--> interruptible backoff
                                  `--> provider attempt
```

A logical operation may make several model requests as it processes tool calls. Each request has
its own retry loop. The loop retries only the provider request; it never reruns a completed tool
effect or restarts the logical operation.

## Ownership

| Owner | Responsibility |
| --- | --- |
| AI failure policy | Classify a provider failure as retryable or terminal and preserve any retry-after hint |
| Model request runner | Execute attempts, apply bounded backoff, discard failed attempt output, and return one terminal result |
| Harness | Sequence model requests and tool effects without repeating completed tool effects |
| Operation owner | Publish the logical operation's final outcome |
| Projections | Retain durable turn or workflow state, never transient request-retry state |

There is no retry worker or retry controller. Turn initiation does not participate in request
retry, and no generic event exists merely to wake projections after a delay.

## Retry flow

```text
provider attempt
      |
      +-- success ------------------------------> return response
      |
      +-- terminal failure ---------------------> return failure
      |
      `-- retryable failure
              |
              +-- budget remains --> backoff --> new provider attempt
              |
              `-- budget exhausted ------------> return exhausted failure
```

Intermediate failures do not publish a turn outcome. The operation owner observes only the final
request result and publishes at most one terminal outcome for the logical operation.

## Attempt isolation

Output from a provider attempt is provisional until that request succeeds. If the request fails
and is retried, its partial assistant output is discarded from the logical conversation. Progress
may be displayed transiently, but it is not committed as successful assistant output.

The retry boundary surrounds one model request, not the surrounding harness operation. Tool results
completed before that request remain inputs to every retry attempt. A retry therefore cannot
repeat an already executed tool. If an integration cannot isolate a request from a non-repeatable
side effect, it must return the failure instead of retrying across that side effect.

## Timing and cancellation

- Retry count and remaining delay are process-local and bounded by one shared policy.
- Backoff is exponential and capped; a provider retry-after hint is a minimum delay.
- Waiting and active attempts are interruptible by operation cancellation.
- Concurrent logical operations have independent retry loops; provider admission remains the
  authority for shared capacity.
- Context rejection, authentication failure, invalid requests, and correctness defects are not
  connection retries. They return immediately to the owning workflow for domain-specific handling.

## Replay and recovery

Retry attempts, counts, deadlines, and sleeps are not events or projection state.

```text
process survives       -> continue the in-memory retry loop
process restarts       -> recover the logical operation with a fresh retry budget
operation completed    -> replay its committed terminal outcome without new effects
```

Recovery need not reproduce the previous attempt count or remaining delay. It must preserve the
logical operation and committed tool results so recovery does not repeat completed side effects.
Higher-level workflows may durably decide what to do after an exhausted request, but that is a
workflow decision rather than continuation of the transient request-retry loop.

## Conformance

- A retryable provider failure can cause another provider attempt without publishing a new
  `turn_started` event.
- No retry delay requires a `wake` event or wall-clock eligibility in TurnProjection.
- A retry never repeats a completed tool effect.
- Failed-attempt assistant output is absent from the logical conversation after a successful retry.
- Cancellation interrupts both the active request and its backoff.
- Restarting during backoff may reset the retry budget without corrupting durable turn state.
- Exhaustion produces one typed final request failure for the operation owner.
