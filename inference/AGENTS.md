# Inference development instructions

## Fork maintenance

Any change to a fork under `inference/native/` must be clearly explained to the user and explicitly approved before implementation. Keep fork divergence minimal and prefer upstream behavior or Magnitude-owned code whenever possible.

### llama.cpp

Change llama.cpp only when strictly necessary and no upstream API, binding-only solution, or Magnitude-owned solution can implement the requirement correctly. Convenience, minor performance improvements, heuristics, duplicated native logic, and model- or architecture-specific patches are not sufficient justification.

### Rust bindings

Rust-binding changes are allowed more readily, but still require explicit user approval and must be well reasoned, carefully designed, narrowly scoped, and tested. Bindings should expose authoritative upstream behavior rather than recreate it.
