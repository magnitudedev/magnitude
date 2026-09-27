# Remote benchmarking

- Edit locally and sync `inference-v2/` to an idle benchmark machine with `rsync`, excluding environments, caches, models, and results.
- Use configured SSH aliases to install dependencies with `uv sync --frozen` and run existing benchmark commands. Cache model weights on each machine.
- Run one measurement at a time per machine; parallelize across machines. Do not sync while a run is active. Compare baseline and candidate on the same machine.
- Copy results and logs back locally, preserving which source was measured.

# Kernel implementation

- Owned Metal kernels live in computational categories under `src/magnitude_engine/kernels/`.
  Use `kernels/core` for capture, typed bindings, automatic composition, source assembly
  and the single MLX invocation boundary; keep numerical
  source in `.metal` files and model/state policy in its existing owners.
  Author through `@kernels.kernel` and compose with `kernels.compile`; use
  `kernels.explain/artifact` for inspection. The native graph bridge is built at package
  installation, never on first inference. Follow `design/kernels.md`; do not restore
  numerical Primitive subclasses, export/replay fallbacks or duplicate launch signatures.

# Performance ceilings

- A performance ceiling must remain an upper bound on what the declared contract permits.
  Too high is acceptable; too low is a modeling error. Tighten it only with a mathematical
  derivation proving the new bound under explicit premises. Benchmarks and reference speeds
  are evidence, not proofs of ceilings. Do not exclude legal optimizations to lower a bound.
- Keep theory and benchmarking outside production execution. Reuse production identities
  and actual runtime bindings; never maintain a separate benchmark model definition.
- `@blueprint` declares serialized construction. `@component("FULL:CANONICAL:SOURCE:VARIANT")`
  declares an execution ID once.
  Other Python code references the declaring class or instance through `component_id(...)`;
  never reconstruct IDs from pieces or maintain a parallel ID catalog.
  Numerical implementations contain no performance descriptions or tree builders. Keep
  typed binding schemas, operand inspection and formulas under `performance/`. If needed,
  a small `bindings()` method may expose existing typed runtime objects, never build metadata.
- Preserve raw history. Changed implementations need new evidence; unchanged components
  retain compatible evidence. Formula changes recompute assessments without new measurements.
- Architecture assembly sections show semantic hierarchies with canonical IDs, collapsing
  repeated instances and identifying optional/shared components. Exact runtime trees and
  current assessments belong in performance tooling. Put investigation
  notes and provenance in results or session logs. Do not use sub-agents unless explicitly asked.
