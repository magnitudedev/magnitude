---
applies_to:
  - inference/crates/icn-engine/src/scheduler.rs
  - inference/crates/icn-engine/src/lib.rs
  - inference/crates/icn-contracts/src/lib.rs
  - inference/crates/icn-server/src/main.rs
---

# Inference scheduler design

The scheduler multiplexes requests through one persistent llama.cpp context. It owns request
admission, native sequence IDs, batch construction, sampling transitions, cancellation, cleanup,
and retained prompt-state selection. Its priorities are correctness, responsive decode, bounded
resource use, and useful continuous batching.

Each loaded model has one executor thread and bounded command and result channels. The executor is
the sole mutator of the model context, sequence pool, active requests, and sampling state. Native
work is serialized even though callers and transport work remain concurrent.

```text
queued → prefill → ready to sample → decode ─┐
              └──────────────────────────────┘
                         ↓
                      terminal
```

## Admission and prompt reuse

The waiting queue is FIFO. Admission requires a free native sequence, a prompt that leaves room for
generation, and capacity within the context. Oversized requests fail before allocation so they
cannot block the queue indefinitely.

For cacheable text requests, the scheduler chooses the free sequence whose committed token history
has the longest exact prefix. It trims unmatched suffix state with standard llama.cpp sequence
operations and prefills the remainder. If no useful prefix exists, it clears the sequence and uses
cold prefill. Reuse never crosses a model context and never relies on a parallel logical or physical
KV representation.

## Batch construction

Each iteration adds at most one decode token for every runnable decode sequence, ordered by sequence
ID, then spends remaining logical capacity on prompt work. Prefill start order rotates, and each
request receives at most one configured quantum per pass until the batch is full or no work remains.
Decode-first ordering protects token latency while rotating chunked prefill prevents a long prompt
from monopolizing the context.

Multimodal prefill and MTP use specialized preparation paths but enter the same request state
machine. MTP keeps its target and draft sequence states synchronized through native linked contexts
and the native speculative controller.

## Commitment and failure

Prompt history becomes reusable only after native decode succeeds. A sampled token remains
uncommitted until its decode or speculative-verification step succeeds. Cancellation is observed
before admission, sampling, and batch selection.

Completed, cancelled, disconnected, and failed requests flow through sequence cleanup. A shared
native decode failure can leave several sequences ambiguous; the executor synchronizes and clears
the affected context instead of guessing what committed. Cleanup failure quarantines a sequence.

Command, event, and per-request outbound queues are bounded. Overload is explicit. A slow consumer
temporarily stops only its own request from receiving native work while other runnable sequences
continue. Exclusive native tasks run only while inference is idle.

Hardware observation is a read-only command class on the existing bounded command stream. The
scheduler performs at most one capture between native batches, and then continues ordinary request
work. Observation cannot mutate request or context state and cannot be implemented as an exclusive
task, because exclusive tasks intentionally wait until inference is idle.

## Current limitations

- Queue admission is FIFO and can head-block; there are no priorities or deadlines.
- Waiting requests are not reordered by prefix benefit or estimated cost.
- Running requests are not preempted.
- Retained prompt state is sequence-local and process-local; concurrent sequences do not share
  physical KV pages.

## Acceptance criteria

- One executor exclusively owns all mutable native state for a loaded model.
- Decode work is serviced ahead of prefill and long prefills are chunked fairly.
- Reuse requires an exact committed-token prefix and uses upstream sequence operations only.
- Cancellation, backpressure, overload, and native failure have bounded, explicit outcomes.
