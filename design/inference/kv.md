---
applies_to:
  - inference/crates/icn-engine/src/scheduler.rs
  - inference/crates/icn-engine/src/lib.rs
  - inference/crates/icn-contracts/src/lib.rs
  - inference/native/llama-cpp-rs/llama-cpp-2/src/context/kv_cache.rs
---

# KV state reuse

Magnitude uses llama.cpp's standard sequence state as a disposable prompt-reuse optimization. It
does not patch llama.cpp with a second KV ownership model, expose physical KV pages, or maintain
device, host, and disk cache tiers.

Each loaded model owns one native context and a fixed sequence pool. When a request completes, its
committed token history may remain attached to that sequence. Admission selects the free sequence
with the longest exact token prefix, removes any unmatched suffix through llama.cpp's standard
sequence operation, and prefills the remainder. A sequence with no useful prefix is cleared before
reuse. Model, tokenizer, context, adapter, and process isolation follow naturally because state never
leaves its owning native context.

Prompt state becomes reusable only after native decode succeeds. The sampled token is not committed
until a later decode or speculative-verification step accepts it. Cancellation and request failure
clear or restore sequence state through standard llama.cpp APIs; failed cleanup quarantines the
sequence instead of returning ambiguous state to the pool.

MTP keeps target and draft sequence state aligned through their native linked contexts. Multimodal
requests may reuse sequence state only when their exact prepared token and media semantics are safe;
otherwise they begin from an empty sequence.

The cache is never authoritative. Tokens and request content remain canonical, and any inability to
reuse state falls back to ordinary prefill without changing inference results. There is no
restart-persistent KV cache, cross-process sharing, page-level sharing between concurrent sequences,
or cache-administration API.

## Acceptance criteria

- KV reuse uses only upstream llama.cpp sequence and state primitives.
- Reuse requires an exact committed-token prefix in the same loaded context.
- Cache failure cannot corrupt or fail an otherwise valid request when cold prefill remains possible.
- No execution plan, API, metric, or CLI option exposes removed physical-page or storage-tier policy.
