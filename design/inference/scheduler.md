# Inference scheduler design

Status: implemented policy, with planned sophistication explicitly excluded below.

The scheduler multiplexes requests through one persistent llama.cpp context. It owns request admission, native sequence IDs, KV-capacity reservation, batch construction, sampling transitions, cancellation, and cleanup. Its priorities are correctness under shared KV ownership, responsive decode, and useful continuous batching.

KV storage and movement are described in [kv.md](./kv.md). The executor and request lifecycle around the scheduler are described in [engine.md](./engine.md).

## Influences

- **llama.cpp** informs the persistent-context execution loop, stable sequence ordering, continuous batching constraints, and the legacy prompt-checkpoint fallback. Magnitude owns admission and batching policy rather than adopting llama-server's slot system.
- **SGLang** motivates longest-prefix-aware admission, in-flight shared-prefix materialization, path locking, and eventual queue clustering by radix locality. The implemented 64-token producer/sibling rule is a small version of that idea; general prefix-weighted queue ordering is not yet implemented.
- **vLLM** motivates allocating against concrete next-step KV demand, chunking long prefills, distinguishing referenced from reclaimable cache pages, and making cache availability part of admission. Priority preemption and asynchronous external-cache transfer states remain future work.

## Execution model

Each loaded model has one dedicated executor thread. That thread is the sole mutator of the model context, active requests, sequence pool, and radix cache. Requests arrive through a bounded command channel and stream results through bounded per-request channels. Native operations stay serialized; client-side transport work does not run inside the native critical section.

The scheduler tracks requests in four phases:

```text
queued → prefill → ready to sample → decode ─┐
              └──────────────────────────────┘
                         ↓
                      terminal
```

`ready to sample` means a batch has produced logits for that sequence. Sampling transitions the request either to its next one-token decode or to terminal completion. Backpressured requests are temporarily excluded from native batch candidates until their outbound events drain.

## Admission

The waiting queue is FIFO. Admission continues from the front while all required resources are available:

- a free native sequence ID;
- a valid prompt that fits the per-sequence context while leaving room to generate;
- enough shared unified-KV cells for admitted but not-yet-materialized prompt work;
- enough cells to promote any RAM/disk cache hits.

Oversized prompts fail before capacity reservation so an impossible request cannot block the queue indefinitely. The scheduler does not admit around a blocked head request.

With radix KV enabled, admission tokenizes the candidate and asks the cache for both its logical matched prefix and the portion already resident on device. Device pages consume no new cells when attached; lower-tier pages and uncached prompt tokens do. The scheduler includes prompt cells already promised to active requests in its reservation, evicting unlocked cache suffixes when possible. If active locks leave insufficient capacity, the request remains queued.

For concurrent shared prefixes, a small materialization rule prevents duplicate prefill. If an active request is computing a common prefix of at least 64 tokens beyond the cache's committed point, a queued sibling waits. The producer publishes complete pages before generation; the sibling then attaches those pages and proceeds concurrently.

When radix pages are unavailable, the legacy path can assign a free sequence whose retained prompt checkpoint has the longest token prefix. This is sequence-local reuse and does not share physical pages among concurrent siblings.

## Batch construction

Batch planning is intentionally simple and deterministic:

1. Add at most one decode token for every runnable decode sequence, ordered by sequence ID.
2. Spend remaining logical batch capacity on prompt work.
3. Rotate the starting prefill sequence each iteration.
4. Give each prefill at most `prefill_quantum` tokens per pass, repeating passes until the batch is full or no work remains.

Decode-first ordering protects token latency. Rotating, chunked prefill prevents one long prompt from monopolizing every remaining batch slot. Stable decode ordering also helps the backend reuse a consistent execution graph.

Before native decode, the scheduler computes the exact cell demand of the selected work, including speculative draft tokens when MTP is active. In radix mode it frees enough unlocked device pages or rejects the batch if the capacity cannot safely be produced. It never knowingly overcommits unified KV.

Multimodal prefill and MTP use specialized preparation paths but feed the same request state machine. MTP is mutually exclusive with radix-page caching in the current implementation.

## Cache interaction

The scheduler and KV cache form one allocation protocol:

```text
match → reserve → lock → promote → attach → prefill/decode → publish → retain/unlock
```

Lookup alone does not authorize reuse. A path must stay locked from attachment until the request releases or extends it. Promotion failures shorten the lock to the successfully attached prefix and convert the remainder to cold prefill. At request release, cacheable committed history is pinned into complete pages before the native sequence is cleared and returned to the free pool.

Cache publication happens at a ready-to-sample boundary, after native KV is known to be committed. The currently sampled token is not added to committed history until its decode/verification step succeeds.

## Fairness, cancellation, and failure

Fairness is local rather than global: queued admission is FIFO, decode is serviced every batch, and prefill start order rotates. There is no request priority, deadline, or explicit starvation score.

Cancellation is observed before admission, sampling, and batch selection. Completed, cancelled, disconnected, and failed requests all flow through sequence cleanup. A failure from a shared native decode can leave context state ambiguous; the engine synchronizes, clears the whole affected context, and fails resident work instead of guessing which sequence committed.

Exclusive native tasks run only when resident inference work is idle. Shutdown stops admission, fails queued work, drains cleanup for active requests, and joins the executor thread.

## Backpressure and overload

Command, event, and per-request outbound queues are bounded. A full command queue or a total tracked-request count above the configured bound returns overload rather than allowing unbounded memory growth. A slow consumer stops its own request from receiving more native work while other runnable sequences continue.

## Current limitations

- Waiting requests are not ordered by longest prefix, subtree locality, expected cache benefit, or age-weighted cost.
- There is no temporary radix over the waiting queue beyond the 64-token producer/sibling rule.
- Disk promotion blocks the executor; there is no asynchronous transfer-wait state or read-priority I/O worker.
- Admission does not compare avoided prefill time with promotion cost, so a disk hit may be slower than recomputation.
- There is no benefit-aware preemption of running requests.
- Prefill quanta are not aligned deliberately to KV page publication boundaries.
- FIFO head blocking is possible when the oldest otherwise-valid request cannot currently obtain KV capacity.
