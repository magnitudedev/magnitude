---
applies_to:
  - inference-v3/src/engine/inputs/**
  - inference-v3/src/engine/models/qwen35/**
  - inference-v3/src/engine/models/sequence.py
  - inference-v3/src/engine/serving/**
  - inference-v3/src/engine/service/**
  - inference-v3/tests/generation/**
  - inference-v3/tests/inputs/**
  - inference-v3/tests/models/**
---

# Conditioned generation

## Input meaning and ownership

A request binds an immutable model input plan before decoder admission. The plan
identifies token inputs, conditioning spans, processor identity, rotary coordinates
and continuation semantics. Image ordering and repeated occurrences are meaningful;
equal image content does not merge distinct positions in the prompt. Serving owns
validated source decoding and host preparation; model formulas own numerical
encoding and projection through the ordinary Ops execution owner.

Projected features replace precisely their declared decoder embedding rows. Physical
cache positions count consumed inputs; rotary positions express model semantics.
They must not be substituted for one another after an image span. Chunking preserves
the same embeddings and coordinates as an unchunked request.

Host preparation, numerical conditioning and decoder work have separate completion
boundaries. Encoding an image does not consume decoder tokens. Admission accounts
for live conditioning and decoder state under one resource budget.
Borrowed views retain their backing allocations; accounting counts shared backing
once. A request source retains the inputs needed for replay independently of resident
decoder state. Checkpoints retain the features still needed after their boundary.
Cancellation during host preparation transfers late-result cleanup to its worker;
an abandoned request cannot leave a newly prepared native prompt without an owner.

Ordinary text generation does not construct an image encoder or retain image
features. Its existing numerical and execution path remains the regression baseline.

## Qualification

Qualification covers image ordering, repeated images, multi-turn images, chunking,
mixed text and image requests, constraints, cancellation, and conditioning ownership.
Image preprocessing and numerical encoding have independent reference checks.

Ordinary-text regression comparisons use matched frozen and candidate builds with
identified source, model artifacts, hardware, context, batching, and warmup.
Compilation, host preparation, image encoding, and decoder service are reported
separately.
