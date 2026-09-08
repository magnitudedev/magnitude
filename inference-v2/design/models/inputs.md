# Model input and continuation

**The model owns how inputs become executable computation. Shared execution and
service policy consume valid boundaries and explicit requirements, without
interpreting a model family or modality.**

Input adapters produce immutable decoder sequence layouts and semantic spans.
Spans identify inputs whose token IDs alone do not describe their computation,
declare whether interior prefixes are independent continuation boundaries, and
distinguish language history from conditioning positions. Identity covers all
content and preparation choices that affect the declared input.

Ordinary text has no exceptional spans and can advance at every token boundary.
An indivisible dependency span can be numerically tiled, but cannot publish or
retain an interior prefix as complete state. Fixed prompt inputs and independently
committable input prefixes are distinct properties.

Prompt advancement enforces dependency boundaries within its allowance. Divisible
spans do not force decoder execution boundaries. Prefix retention may request
semantic transitions so preceding text and completed conditioning can be retained
independently; disabling retention removes that optional execution cost. Service
consumes a complete indivisible unit when that unit is next.
Allowances are soft; hard input and resource limits are validated before service.
An indivisible unit cannot be silently split or indefinitely starved by a smaller
allowance. Physical storage pages do not define semantic boundaries.

The final input that predicts the first output is a legal unit. Leaving one bare
token for generation is valid only when that token is independently executable
with its complete input semantics. Generated language tokens append to semantic
history without changing the identity of earlier conditioning.

Prefix indexing consumes semantic input identities and complete checkpoints. It
does not inspect image bytes or token IDs to recover their meaning. Generation
uses the adapter's eligible language history for history-dependent algorithms;
conditioning placeholders are not ordinary repeated language.

These contracts refine [model composability](composability.md) and the valid
advancement supplied to [batching](../engine/batching.md). They describe model
semantics, not a second scheduler or numerical execution system.

## Preparation and execution

Serving resolves bounded image content and applies the bound model's CPU preparation
component. That immutable artifact binding belongs to the template lifetime and is
reused across CPU request work; request data and processing options remain local.
The worker validates processor identity, tensor geometry, and the exact expanded
prompt before admission. The model owns placeholder interpretation; the
scheduler, generation methods, and storage do not recognize special image IDs.

Stateless prerequisites use the same execution owner and reservation ledger as the
decoder. They expose compatible ready computations without allocating fictitious KV
state. Working capacity is acquired before numerical execution. A batch that cannot
reserve its workspace can split while preserving unchanged prepared inputs.

Input state assembles typed, row-local numerical operands and pins their resources
through device completion. Features no longer needed by a live continuation can be
released without invalidating a checkpoint or an in-flight consumer.

## Complete continuation

A model checkpoint combines physical decoder state with the input semantics needed
after the same committed boundary. Storage remains responsible for KV and recurrent
history; input state retains coordinates, dependency identity, and any unfinished
conditioning. Replay preserves the same input operands as the original forward.

[Qwen](architectures/qwen35.md) keeps rotary coordinates independent of physical KV
positions, including the continuation offset after images. A partial causal image
checkpoint retains its projected features and requires the corresponding prepared
source to resume the image. [Gemma](architectures/gemma4.md) treats bidirectional
image spans as atomic continuation units. Image embeddings do not themselves
require an attention override: causal image rows use ordinary causal visibility,
and mixed batches preserve each row's declared bounds. Once either model has consumed an image,
its ordinary language continuation needs no image re-encoding.

## Feature and source ownership

Completed projected features have bounded retention independent of decoder prefixes.
The cache keys the bound encoder and prepared content identity. Continuations,
checkpoints and unfinished executions hold independent leases; eviction releases
only cache ownership. Memory pressure reclaims unborrowed entries before decoder
prefixes. Cold computation, feature reuse and decoder-prefix reuse are distinct
events; a reused feature still requires its decoder computation.

Input preparation owns validated CPU tensors at the artifact's declared image
quality. Numerical values travel as lossless binary buffers, separately from JSON
control metadata. Transport bounds and capacity admission do not silently change
resolution or tensor precision. The worker reserves transport and interpretation
capacity before reading those buffers, including while queued; a rejected payload
is drained through bounded scratch so subsequent control messages remain readable.
Model work separately reserves projected features and encoder workspace before
submitting a graph. Cancellation and failure release request ownership only after
dependent execution can safely retire. No encoder result is published to shared retention
before its computation completes. A continuation publishes its feature lease only
after successful completion; failure or cancellation releases unpublished ownership
while execution retains its independent pins.

The serving boundary accepts ordered text and inline image content. It validates
source bytes, decoded dimensions, frame count and expanded context before admission.
Source retrieval is a transport policy; it does not belong in model execution.
Unsupported conditional families or media forms fail explicitly during binding or
preparation. Audio processing and generation are outside the supported input contract.

## `MODEL:INPUT_FEATURES`

**Contract.** Retain completed, content-identified model features independently of
consumed decoder history. Borrowers retain explicit leases; eviction cannot release
storage still used by a request, checkpoint or execution.

`MODEL:INPUT_FEATURES:MAG:LRU` bounds retained feature bytes, refreshes access order,
and reclaims only unborrowed allocations under pressure. The live bound encoder
forms part of the key, so incompatible projections cannot share entries.
`MODEL:INPUT_FEATURES/EXEC` is configuration/bookkeeping attribution in captured
assemblies; it makes no neural throughput or efficiency claim.
