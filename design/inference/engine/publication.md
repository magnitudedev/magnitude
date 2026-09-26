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

A full data ring makes the request ineligible for scheduling until space returns. The first drain
from full to non-full coalesces output credit and wakes the worker on a reserved path independent of
ordinary command capacity. The worker clears that credit while processing it so another full-to-
non-full transition cannot be lost. A processed credit makes the request schedulable only if the
ring is still non-full and the receiver is still open. Receiver cancellation likewise wakes the
worker through a reserved path. Closing either endpoint deterministically closes the request;
worker teardown records terminal failure for a receiver that remains open.

When the host is another process, the worker forwards the stream under host credit: the host
grants one output batch per batch its consumer drains, and the worker drains the queue only while
it holds credit, so a slow consumer backs up to the queue's bound. The terminal outcome follows
the stream's data in the same order; a connection that ends before a request's terminal outcome
is a worker failure for that request.

A failure the execution owner cannot isolate stops it with one classified error: every live request
terminates with it, every later admission is refused with it, and the worker unloads with its cause.
A device failure therefore reaches the host as device loss on all three paths, never as an engine
invariant.

A live request's status reports its prompt size, the leading prompt tokens restored from a
retained prefix rather than computed, its resident position and its output count, so prefill
progress distinguishes reused from computed input.

Shared queue state contains only device-free publication data, endpoint state, and wake state. Live
generation, state transactions, device resources, and submissions remain confined to the worker.
Host receiver progress does not require periodic polling commands or sleeps.
Before submitting numerical work, the owner reserves output slots for every token the suspended
generation round may accept, including tokens forced by a constraint or emitted while prefill
advances without a selection row. Accepted output always has a matching publication permit.

Terminal success carries measured physical prompt and predicted durations accumulated by the
execution owner at completed program boundaries. The host response converts these durations to
milliseconds without inferring them from token counts or scheduler estimates. A successful response
requiring timing metadata fails closed if those measurements are absent.

## Acceptance criteria

- At capacity, rejected data stays with the sender and already accepted data is never replaced.
- Forced and prefill-emitted tokens publish without exhausting unreserved output slots.
- Success and failure follow all accepted output in the same stream, including when the ring is full.
- Output credit and cancellation wakes remain observable when the ordinary worker mailbox is full.
- Output-blocked requests do not receive new numerical service until credit is processed.
- Dropping the worker sender gives an open receiver one terminal failure.
- Every terminal path releases request-owned resources after physical work and state reconciliation.
- Terminal timing fields represent measured execution and remain consistent in streaming and complete responses.
