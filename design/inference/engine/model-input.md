---
applies_to:
  - inference/engine/artifacts/**
  - inference/engine/families/**
  - inference/engine/executor/**
  - inference/engine/chat/src/artifacts.rs
  - inference/engine/chat/src/tokenizer.rs
---

# Model input boundary

Artifact preparation owns process-local package identity, payload interpretation,
decoding, and bounded generic media. It may inspect a bounded header bundle
before weight payloads exist; this validates
metadata, tensor geometry, and declared non-overlapping ranges without claiming
that the payloads are present. Executable package admission requires the full
payload ranges. A package component is one GGUF or every shard of a split GGUF
(llama.cpp `gguf-split`): the first shard names the component, the remaining
shards are found beside it by the split naming convention and must declare their
positions, and header and payload opens both present one directory (the first
shard's metadata, every shard's tensors) under one component identity. The header
reader accepts every GGML tensor type a GGUF can store; whether an encoding
executes is decided by planning, where an unsupported encoding is an unsupported
representation, never a malformed artifact. Artifact preparation does not assign model-family coordinates or
position-table interpolation. A family adapter consumes
the model definition, token plan, and generic prepared media to create one closed numerical input
contract. The contract contains the final token coordinates, media spans, patch order, attention
coordinates, attention window ranges, and position-encoding interpolation indices and coefficients
needed by execution.
The adapter is configured with tokenizer-derived family marker identities before it prepares a
request; the model definition and media alone cannot identify those tokens.
For text-only input, the same closed contract records the family-supplied coordinates directly;
constructing it performs alignment checks and does not derive coordinate semantics.
Every prompt token is one input row, in order, except each image's placeholder token, which the
adapter expands to its span's rows; the input's layout alone therefore maps a prompt position to its
row.

The tokenizer is adapted from the container's tokenizer facts, never from the model family: the
declared scheme selects one implemented profile, and an unimplemented scheme or uninterpreted
tokenizer metadata is an unsupported representation. Implicit sequence-start insertion is such a
fact. A tokenizer that begins every sequence with its BOS applies it exactly once: a prompt whose
rendered text already begins with the BOS (templates that render it) is not given a second one.
End-of-generation and never-generated tokens are likewise tokenizer facts.

Generic execution validates shapes and bounds against the model definition and consumes the
prepared contract. It does not repeat family-specific spatial derivation or infer semantic meaning
from artifact tensor names. Prepared input and media cross the host/worker boundary as owned,
device-free values; a worker process re-establishes every construction invariant of a decoded
input against its model definition before admitting it. A worker process opens exactly the package
the host admitted (same component files and sizes and tensor directories) and carries the host's package
identity, so both sides name one package. Live device resources remain worker-confined.

A separate block drafter declares block causality independently of history
windows. Its GGUF `dflash.attention.causal` is either BOOL (broadcast to all draft
layers) or ARRAY<BOOL> with exactly one element per layer. No sliding-window or
model-name heuristic supplies omitted semantics. Interpretation normalizes this
once to per-layer block attention; execution consumes only that normalized model.

## Acceptance criteria

- Artifact output contains no model-family numerical controls.
- Any valid GGUF header, split or not, parses; a split package plans and loads as one package.
- The family adapter computes every spatial value consumed by the vision lane.
- A prepared numerical input cannot contain mismatched token, span, media, or patch domains.
- A prepared numerical input holds each distinct image once, keyed by content identity: every
  span names a held image and every held image is placed by a span.
- A prepared prompt holds the tokenizer's implicit BOS exactly once, and no BOS otherwise beyond
  what the template renders.
- Execution performs no model-family coordinate or interpolation calculation.
