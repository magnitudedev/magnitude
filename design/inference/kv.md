---
applies_to:
  - inference/crates/icn-engine/src/scheduler.rs
  - inference/crates/icn-engine/src/lib.rs
  - inference/crates/icn-contracts/src/lib.rs
  - inference/native/llama-cpp-rs/llama-cpp-2/src/context/kv_cache.rs
---

# KV state reuse

Prompt reuse is a disposable optimization over standard llama.cpp sequence state. Tokens and
request content remain canonical; inability to reuse KV always falls back to cold prefill.

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

Admission selects the available sequence with the longest exact committed-token prefix.

```text
retained:  [ A B C D E ]
incoming:  [ A B C X Y ]
match:     [ A B C ]
action:    trim D E; prefill X Y
```

No match means cold prefill. Reuse never crosses a model, tokenizer, context, adapter, or process.

## Commit boundary

```text
assemble batch -> native decode -> linked MTP processing -> commit request progress
                     |                    |
                     `------ failure -----'--> discard staged progress + reset native state
```

Logical prompt tokens are not proof of prefill. Only the post-decode processed count is reusable.
A sampled token is likewise uncommitted until decode or speculative verification accepts it.

## Retention policy

| Terminal condition | Reusable state |
| --- | --- |
| Complete, cancelled, disconnected, request-local failure | Retain committed text prefix when caching is enabled |
| Acquired but not natively mutated | Return the original available sequence unchanged |
| Multimodal or cache-disabled request | Clear and return without a prefix |
| Native cleanup failure | Quarantine the sequence |
| Shared target/draft failure | Reset contexts; invalidate available prefixes; return active sequences empty |

MTP retains only at aligned target/draft boundaries. Multimodal reuse remains disabled until exact
token-and-media equivalence can be guaranteed.

## Capacity

Each sequence receives one configured-context partition. The pinned unified-KV API cannot preserve
that per-sequence limit independently of total capacity, so adopting it requires a new approved
design.

## Boundaries

No restart persistence, cross-process sharing, concurrent page sharing, physical-page model,
storage tiers, or cache-administration API.

## Acceptance criteria

- Reuse uses only upstream llama.cpp sequence/state primitives.
- A reusable prefix contains exact, committed tokens from the same loaded context.
- Prefix metadata cannot exist independently of its available sequence.
- Cache loss cannot fail or alter a request that can cold-prefill.
- No public surface exposes physical-page or storage-tier policy.
