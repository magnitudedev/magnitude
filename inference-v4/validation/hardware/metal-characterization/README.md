# Metal hardware characterization proof of concept

This directory makes the hardware-characterization boundary executable without
touching the production estimator or Metal backend. It is deliberately a
standalone Rust crate. It has no dependencies on candidates, the solver, the
planner, the compiler, Metal handles, or estimator-core.

The boundary is exclusive to analytical evaluation:

```text
closed Metal cost-fact vocabulary
        + exhaustive fact-to-protocol mapping
                    |
          [constructibility gate]
                    |
 exact device identity + one fixed probe library
                    |
                    v
         RawObservationBundle
                    |
       structural validation only
                    |
                    v
       model-specific pure interpreter
                    |
                    v
   model-owned CertifiedMetalProfile
```

`FeedbackEvaluator`, `CandidateDomain`, the generic evaluator contract, and the
planner do not depend on this directory or on a certified profile.

## What the proof establishes

- `MetalCostFact` is the exact closed union of all 30 service classes in the
  audited estimator/core and estimator/metal algebra. `protocol_for` is an
  exhaustive match with no fallback. Adding a fact breaks compilation until
  its acquisition obligation is classified.
- `CompleteProbeManifest` is constructed by mapping that closed union. Its
  constructibility gate rejects every `NotObservable` fact before hardware.
- `RawObservationBundle` must contain all facts in canonical order. Structural
  validation reports every failure or omission together.
- The whole probe library has one digest and one measurement endpoint.
- Compilation, warmup, acquisition, and total duration have independent caller
  supplied hard budgets. The seconds/tens-of-seconds value must be frozen by
  the production protocol and demonstrated on hardware; this proof does not
  invent or validate a universal value.
- Structural validation cannot create a physical profile. `replay` delegates
  only validated evidence to a model-specific pure interpreter. That
  interpreter must define the typed profile and total interpretation over every
  invocation regime. There is no generic curve fitter or coefficient reducer.

`MetalCostFact` is an audited snapshot inside an isolated validation crate. It
is not yet constructionally tied to production: production could currently add
or change a service without compiling this crate. The boundary becomes real
only when the replacement closed physical-fact vocabulary moves into
`estimator/metal`, cost lowering consumes it exhaustively, and both the profile
type and characterization protocol are generated from that same vocabulary.

The host tests prove the current manifest is rejected as unconstructible before
hardware. They are not hardware evidence or estimator accuracy evidence.

## The intended aggregate acquisition

The future native adapter must compile the manifest's entry points into one
library. It then performs one bounded sequence:

1. query immutable device facts and record the complete identity;
2. warm the single production queue under a declared steady-state protocol;
3. run all independent-chain, working-set, occupancy, contention, collective,
   and lifecycle batches;
4. populate every raw fact slot, retaining failures instead of returning early;
5. record timer resolution and independent samples at the GPU start/end
   endpoint;
6. stop at the caller-supplied total budget and mark every unexecuted endpoint
   failed;
7. persist the bundle before profile certification.

No complete candidate is compiled or timed. Held-out candidate measurements
belong to later qualification and cannot enter the raw bundle or profile.

## Direct observation rules

The current four-command residual scheme is excluded. A measured fact must be
the dominant variable of a single directly amplified experiment. A type-changing
operation or Boolean-producing helper is not assigned a coefficient by timing a
forward operation plus feedback and subtracting a separately timed baseline.

Instead, the shared target-closed representation must expose the native
primitive sequence and path structure. Characterization measures reusable
physical primitives and regimes. The analytical estimator composes those facts
through that exact sequence.

- BF16-to-F32 is a 16-bit representation cast, integer left shift, and 32-bit
  representation cast. It uses primitive integer facts; it is not a service.
- F32 comparisons use the visible helper control/bit sequence and its explicit
  finite paths. They are not directly probed as an opaque comparison service.
- Type-changing operations use their visible primitive sequences. A reverse
  conversion inserted only to make a benchmark dependency cycle is never
  charged to the forward conversion.
- Strict helpers with data-dependent paths produce a path-conditioned physical
  model or a sound interval across all reachable paths when runtime data is
  unknown. No implicit input distribution is permitted.

## Current blockers

The result of this proof is a constructibility failure, not a certified
profile. `CharacterizationProtocol::new` currently fails
before acquisition because the audited estimator algebra contains unobservable
facts. A real bundle cannot yet be honestly acquired or interpreted:

1. `ScalarEmissionFamily` still names opaque strict helpers. `render.rs` and
   `softfloat.metal` add 32/64-bit integer operations, rounding, branches,
   state-machine loops, and exceptional paths after the tuning representation.
   Those expansions must become shared target-closed cost programs consumed by
   both renderer and estimator.
2. Register demand is not present before native compilation. An occupancy model
   cannot select a candidate using compiler-reflected register pressure after
   selection. The target-closed cost program must own a pre-native register-live
   bound, or the analytical architecture is not closed.
3. Global memory is currently one additive service. The executable lacks the
   complete cache-resident/spill, reuse-distance, coalescing, and concurrency
   model needed to select a regime.
4. Representation decode/repack is charged as one unit even though the renderer
   emits recipe-specific loads, bit extraction, integer arithmetic, table
   selects, conversions, and strict multiplication.
5. Kernel control-flow costs are currently one generic control unit. Branch and
   repeat counts exist structurally, but strict helper paths depend on runtime
   values and are not exposed as cost paths.
6. The replacement cost-fact vocabulary, exact experiment shapes, and total
   physical derivation rules are not yet frozen. Until E exposes physical facts
   and every exhaustive invocation regime, `v0` remains a boundary proof.

These are completeness failures. More rounds, wider confidence thresholds, or
another opaque service probe cannot resolve them.

## Local validation

```bash
cargo test --manifest-path \
  inference-v4/validation/hardware/metal-characterization/Cargo.toml
```
