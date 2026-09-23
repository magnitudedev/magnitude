---
applies_to:
  - inference-v4/engine/artifacts/**
  - inference-v4/engine/model-contracts/**
  - inference-v4/engine/models/**
  - inference-v4/engine/model-executor/**
---

# Model input boundary

Artifact preparation owns package and payload identity, decoding, and bounded generic media. It
does not assign model-family coordinates or position-table interpolation. A family adapter consumes
the model definition, token plan, and generic prepared media to create one closed numerical input
contract. The contract contains the final token coordinates, media spans, patch order, attention
coordinates, and position-encoding interpolation indices and coefficients needed by execution.
The adapter is configured with tokenizer-derived family marker identities before it prepares a
request; the model definition and media alone cannot identify those tokens.
For text-only input, the same closed contract records the family-supplied coordinates directly;
constructing it performs alignment checks and does not derive coordinate semantics.

Generic execution validates shapes and bounds against the model definition and consumes the
prepared contract. It does not repeat family-specific spatial derivation or infer semantic meaning
from artifact tensor names. Prepared input and media cross the host/worker boundary as owned,
device-free values. Live device resources remain worker-confined.

## Acceptance criteria

- Artifact output contains no model-family numerical controls.
- The family adapter computes every spatial value consumed by the vision lane.
- A prepared numerical input cannot contain mismatched token, span, media, or patch domains.
- Execution performs no model-family coordinate or interpolation calculation.
