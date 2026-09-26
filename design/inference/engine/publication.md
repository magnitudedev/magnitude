---
applies_to:
  - inference-v4/engine/service/**
  - inference-v4/engine/src/service/**
  - inference-v4/engine/src/serving/**
  - inference-v4/engine/src/chat/**
  - inference-v4/engine/chat/src/response.rs
  - inference-v4/engine/chat/src/lib.rs
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

Shared queue state contains only device-free publication data, endpoint state, and wake state. Live
generation, state transactions, device resources, and submissions remain confined to the worker.
Host receiver progress does not require periodic polling commands or sleeps.

Terminal success carries measured physical prompt and predicted durations accumulated by the
execution owner at completed program boundaries. The host response converts these durations to
milliseconds without inferring them from token counts or scheduler estimates. A successful response
requiring timing metadata fails closed if those measurements are absent.

## Acceptance criteria

- At capacity, rejected data stays with the sender and already accepted data is never replaced.
- Success and failure follow all accepted output in the same stream, including when the ring is full.
- Output credit and cancellation wakes remain observable when the ordinary worker mailbox is full.
- Output-blocked requests do not receive new numerical service until credit is processed.
- Dropping the worker sender gives an open receiver one terminal failure.
- Every terminal path releases request-owned resources after physical work and state reconciliation.
- Terminal timing fields represent measured execution and remain consistent in streaming and complete responses.
