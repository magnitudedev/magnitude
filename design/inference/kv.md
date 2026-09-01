---
applies_to:
  - inference/crates/icn-engine/src/scheduler.rs
  - inference/crates/icn-engine/src/lib.rs
  - inference/crates/icn-contracts/src/lib.rs
  - inference/native/llama-cpp-rs/llama-cpp-2/src/context/kv_cache.rs
---

# KV state reuse

Prompt reuse is a disposable optimization over standard llama.cpp sequence state. Prepared prompt
input remains canonical; inability to reuse KV always falls back to cold prefill.

## Ownership

```text
SequencePool
  |
  `-- AvailableSequence { ID, prefix? }
          |
          | acquire (move)
          v
      ActiveSequence { ID } + committed request progress
          |
          | terminal + safe cleanup
          v
      AvailableSequence { ID, new prefix? } -- return --> SequencePool
```

Prefix metadata cannot exist apart from its available sequence. Native KV stays inside llama.cpp.

## Reuse

Admission selects the available sequence with the longest exact committed semantic prefix when the
match exceeds ten percent of the incoming prompt. Otherwise it selects the least recently used
sequence, allowing unused empty capacity to absorb weakly related requests without displacing a
recent retained prefix. A semantic prompt is an ordered sequence of text and media spans:

```text
Text(tokens) -> Media(identity, logical tokens, native positions) -> Text(tokens)
```

Text matches token-by-token. Media matches only when its type, decoded-content identity,
dimensions, logical token count, and native position count are identical. A reusable boundary may
split text but never media. An absent or ambiguous media identity is a cache miss.

```text
retained:  [ A B C D E ]
incoming:  [ A B C X Y ]
match:     [ A B C ]
action:    trim D E; prefill X Y
```

No match means cold prefill. Reuse never crosses a model, tokenizer, context, adapter, or process.

## Commit boundary

```text
assemble batch -> native decode -> linked speculative processing -> commit request progress
                     |                    |
                     `------ failure -----'--> discard staged progress + reset native state
```

Logical prompt tokens are not proof of prefill. Only the post-decode processed count is reusable.
A sampled token is likewise uncommitted until decode or speculative verification accepts it.
Every committed boundary records both logical tokens (for progress and capacity) and native
positions (for KV removal and continuation). They differ for M-RoPE media.

For linked speculation, those values are also the two cache coordinates: target KV uses the native
position and draft KV uses the logical token position. The binding accepts both coordinates for
mirroring, drafting, verification, rollback, and trimming. It never derives draft position from
target position or from text-only token history.

## Retention policy

| Terminal condition | Reusable state |
| --- | --- |
| Complete, cancelled, disconnected, request-local failure | Retain committed semantic prefix when caching is enabled |
| Acquired but not natively mutated | Return the original available sequence unchanged |
| Cache-disabled request | Clear and return without a prefix |
| Native cleanup failure | Quarantine the sequence |
| Shared target/draft failure | Reset contexts; invalidate available prefixes; return active sequences empty |

Speculative execution retains only at linked target/draft boundaries. Its checkpoint is a single
state variant containing both native sequence snapshots and any method-owned state; target plus an
optional draft snapshot is not a valid state. Media encoding and its embedding decodes form one
commit operation: failure or cancellation invalidates ambiguous native state rather than retaining
a partial image.

Checkpoint policy may request a logical position inside a media span, but that is not a legal
semantic boundary. The prompt layout advances such a request to the end of the indivisible media
span so the cache can retain the completed image without representing partial media state.

## Capacity

Each sequence receives one configured-context partition. The pinned unified-KV API cannot preserve
that per-sequence limit independently of total capacity, so adopting it requires a new approved
design.

## Boundaries

No restart persistence, cross-process sharing, concurrent page sharing, physical-page model,
storage tiers, or cache-administration API.

## Acceptance criteria

- Reuse uses only upstream llama.cpp sequence/state primitives.
- A reusable prefix contains exact, committed text/media semantics from the same loaded context.
- Media is matched by stable preprocessing identity and is never partially reused.
- Logical token and native-position boundaries remain distinct.
- Prefix metadata cannot exist independently of its available sequence.
- Cache loss cannot fail or alter a request that can cold-prefill.
- No public surface exposes physical-page or storage-tier policy.
