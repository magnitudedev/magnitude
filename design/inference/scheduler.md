---
applies_to:
  - inference/crates/icn-engine/src/scheduler.rs
  - inference/crates/icn-engine/src/lib.rs
  - inference/crates/icn-contracts/src/lib.rs
  - inference/crates/icn-server/src/main.rs
---

# Inference scheduler

The scheduler multiplexes requests through one persistent llama.cpp context. Its single executor
owns admission, active requests, native sequences, batch construction, sampling, cancellation, and
cleanup.

The sequence-pool size is resolved load evidence, not provider concurrency. Loading selects one to
four sequences while preserving the configured context limit for each request.

## Request states

```text
waiting -> prefill -> ready -> sample -> decode --+
                       ^                         |
                       `-------------------------'

any state -- stop / cancel / disconnect / fail --> terminal
```

Cancellation is checked before admission, sampling, and batch selection.

## Executor iteration

```text
commands -> observation -> idle-only native task -> cleanup -> admission
   -> sample -> build/decode one batch -> flush -> cleanup -> brief poll
```

Each step is bounded. Scheduling and native KV mutation share one owner thread.

## Admission

The FIFO waiting queue admits a request only when:

- a native sequence is available;
- its prompt leaves room for generation; and
- the context has capacity.

Oversized requests fail before allocation. Every request is prepared once into an ordered semantic
prompt before sequence selection. Cache-enabled requests take the available sequence with the
longest exact reusable text/media prefix only when that prefix exceeds ten percent of the incoming
prompt. Without a qualifying match, admission takes the least recently used sequence; unused empty
sequences therefore precede retained sequences. Before native mutation, cancellation or validation
failure returns that sequence unchanged. Native setup transfers it to active ownership, trims any
unmatched suffix at its native-position boundary, and prefills the remainder.

## Batch policy

```text
batch capacity
  1. one decode token per runnable decode sequence, ordered by sequence ID
  2. remaining capacity split into rotating prompt quanta
```

Decode-first service protects latency; rotating prompt starts prevent monopolization.

Batch construction stages all request effects:

```text
BatchCommit { prompt advances, speculative indices, logits }
                         |
       target decode + linked speculative processing succeed
                         |
                         v
                 mutate request state
```

Failure drops the commit, so staged prompt work cannot become reusable. Text spans participate in
the ordinary fair batch planner. A media span is evaluated atomically at its semantic boundary, and
then scheduling returns to ordinary text batching. At most one media span is evaluated per
scheduler iteration, so projector execution remains serialized without disabling continuous
batching for other resident sequences. Multimodal and speculative preparation share this state
machine; target and draft state remain natively linked. Each successful multimodal embedding
sub-batch is processed speculatively before a later decode may depend on it.

Batch assembly records one draft-sequential position for every target row. The binding preserves
the target batch's token or embedding storage, sequence IDs, row order, and logits flags while
substituting only that position view for linked draft processing. Multimodal callbacks advance and
return both the target-native and draft-sequential boundary, so later scheduling, caching, and
verification continue from the same explicit pair.

## Terminal handling

| Outcome | Sequence disposition |
| --- | --- |
| Complete, cancelled, disconnected, request-local failure | Retain committed semantic prefix when eligible |
| Cache-disabled | Clear and return empty |
| Shared native batch failure | Reset contexts and invalidate all affected reusable prefixes |
| Cleanup failure | Quarantine; never return the sequence to the pool |

Request outcome alone does not determine whether committed native state is reusable.

## Backpressure and exclusive work

Command, event, and outbound queues are bounded. A slow consumer pauses only its request. Read-only
hardware observation runs at most once between batches; mutating exclusive work waits for idle.

## Limitations

- FIFO admission can head-block; there are no priorities or deadlines.
- Waiting requests are not reordered by prefix benefit or estimated cost.
- Running requests are not preempted.
- Reusable state is sequence-local and process-local; concurrent sequences share no physical pages.

## Acceptance criteria

- One executor owns every scheduler and native mutation.
- Decode precedes prefill, and long prefills are chunked fairly.
- Prompt progress becomes visible only after successful native processing.
- Reuse requires an exact committed text/media prefix and upstream sequence operations.
- Cancellation, backpressure, overload, and native failure have bounded outcomes.
