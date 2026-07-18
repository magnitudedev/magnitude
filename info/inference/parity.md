# Inference primitive parity

Magnitude validates its llama.cpp inference path at primitive boundaries. Parity is not inferred
from a generated answer, an HTTP exchange, or a single aggregate score.

## Evidence lanes

The suite has four reference adapters and one comparison stage:

1. **Upstream tests** build and run selected tests from the exact nested llama.cpp source unchanged.
   They qualify the native dependency and backend; they do not prove the Rust bindings.
2. **Upstream tools** run official machine-readable tools such as `llama-bench`,
   `llama-batched-bench`, and `llama-perplexity` with explicit work definitions.
3. **Native qualifications** run one-sided upstream performance tools when ICN links the same native
   implementation. They qualify the dependency/build and do not claim binding parity.
4. **Differential cases** send one neutral case to a thin native C++ oracle and to an ICN Rust
   probe. Both return the same evidence shape.
5. **External comparison** applies a declared exact, structural, numeric-tolerance, capability, or
   performance-ratio comparator. Neither engine decides whether it passed.

The case and evidence protocols describe native operations, not Rust types or HTTP requests. This
keeps fixtures stable when binding APIs change and prevents the probe from becoming a second
implementation of production behavior.

## Primitive families

Correctness covers native baseline integrity, parameter mapping, model metadata, tokenization,
chat preparation, grammar conversion, sampling, reasoning-budget transitions, output parsing,
batch/decode plans, KV and sequence state, cancellation, multimodal preprocessing/evaluation, and
numerical model fidelity.

Performance covers native backend operations, prompt processing, token decode, upstream-defined
combined prompt/decode work, context-depth effects, native multi-sequence batches, and isolated
sampler, chat-preparation, parser, and state operations. A correctness failure invalidates the
corresponding performance claim.

## Reproducibility

Every accepted run identifies the native revisions and source inventory, build flags and compiler,
binary and fixture digests, model digest, backend/device configuration, operation boundary, raw
outputs, and raw timing repetitions. Performance gates require identical work and effective
configuration. Results from an uncontrolled developer machine are diagnostic, not release gates.

Model-backed cases normally expand over the accepted model IDs selected by a profile or CLI. The
runner filters each model by case tags and the registry's primitive compatibility declaration, then
injects a verified local artifact path only at subprocess invocation time. Model-specific native
token vectors and upstream CTest fixtures remain explicitly pinned instead of pretending to be
portable.

Generated evidence lives under `inference/results/parity/`. Declarative inputs and the thin oracle
live under `inference/parity/`; orchestration and comparison code lives in the fork-independent
`icn-parity` crate. The binding-dependent ICN probe is a separate executable so the core runner and
native baseline can be built and tested while the bindings evolve.

Composite server load, queueing, streaming latency, and full generated responses remain useful ICN
integration tests, but they are not primitive parity evidence.
