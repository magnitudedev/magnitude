---
applies_to:
  - inference/engine/scheduler/**
  - inference/engine/generation/**
  - inference/engine/src/worker/**
  - inference/engine/serving/**
  - inference/engine/src/chat/**
  - inference/engine/chat/src/stream.rs
  - inference/engine/chat/src/lib.rs
---

# Request publication

Each admitted request has one ordered outcome stream from the numerical worker to its host
receiver. Output events, terminal success, and terminal failure use the same ordering authority.
The worker never waits for a receiver to make space.

The stream has a bounded FIFO data ring and a separate one-value terminal slot. The terminal slot
can always record one final outcome after the last accepted data event, even while the data ring is
full. The receiver observes all queued data before the terminal. Recording a terminal closes the
stream to further output; a second terminal is an invariant violation.

A full data ring makes the request ineligible to start a round until space returns. The first drain
from full to non-full coalesces output credit and wakes the worker on a reserved path independent of
ordinary command capacity. The worker clears that credit while processing it so another full-to-
non-full transition cannot be lost. A processed credit makes the request schedulable only if the
ring is still non-full and the receiver is still open. Receiver cancellation likewise wakes the
worker through a reserved path. A closed receiver never grants credit: it cancels its request.
Worker teardown records terminal failure for a receiver that remains open.

When the host is another process, the worker forwards the stream under host credit: the host
grants one output batch per batch its consumer drains, and the worker drains the queue only while
it holds credit, so a slow consumer backs up to the queue's bound. The terminal outcome follows
the stream's data in the same order; a connection that ends before a request's terminal outcome
is a worker failure for that request.

Every execution stop (close, persistent Reclaim, or a failure the execution owner cannot isolate)
reaches every live request with its cause: each terminates with it, every pending control and later
admission is refused with it, and the worker unloads with it. A failure stops execution with one
classified error, so a device failure reaches the host as device loss on every path, never as an
engine invariant.

A live request's status reports its prompt size, the leading prompt tokens restored from a
retained prefix rather than computed, its resident position and its output count, so prefill
progress distinguishes reused from computed input.

Shared queue state contains only device-free publication data, endpoint state, and wake state. Live
generation, state transactions, device resources, and submissions remain confined to the worker.
Host receiver progress does not require periodic polling commands or sleeps.
Before a generation round starts, the owner reserves output slots for every token the round may
accept, including tokens forced by a constraint or emitted while prefill advances without a
selection row. A request without that credit waits without starting the round, alone, until a
publication wake; permits unused at reconciliation stay with the request. Credit therefore never
runs out midway through a round, and accepted output always has a matching publication permit.

Terminal success carries measured physical prompt and predicted durations accumulated by the
execution owner at completed program boundaries. The host response converts these durations to
milliseconds without inferring them from token counts or scheduler estimates. A successful response
requiring timing metadata fails closed if those measurements are absent.
Draft history prepared from prompt or replay rows contributes to prompt duration, including
separate-draft injection after each prompt or replay chunk. Proposal and catch-up work during generation
contributes to predicted duration. The executor lane used for an operation does not determine its
timing phase.

## Acceptance criteria

- At capacity, rejected data stays with the sender and already accepted data is never replaced.
- Forced and prefill-emitted tokens publish without exhausting unreserved output slots.
- Success and failure follow all accepted output in the same stream, including when the ring is full.
- Output credit and cancellation wakes remain observable when the ordinary worker mailbox is full.
- A request without credit for its next round does not start it until credit is processed; its
  peers keep running.
- A closed receiver cancels its request.
- Every execution stop terminates each live request and fails each pending control with its cause.
- Dropping the worker sender gives an open receiver one terminal failure.
- Every terminal path releases request-owned resources after physical work and state reconciliation.
- Terminal timing fields represent measured execution and remain consistent in streaming and complete responses.
- Prompt and replay draft preparation are charged to prompt time even when they use the head executor lane.
