---
applies_to:
  - inference-v4/seismic/**
  - inference-v4/seismic-std/**
  - inference-v4/engine/**
  - inference-v4/docs/seismic/**
---

# Seismic numerical precision

Numerical precision is an observable whole-entry compilation contract and a hard selection constraint. It is not a source-level permission, a backend-wide fast-math setting, or a performance penalty.

An explicitly selected top-level native implementation is outside compiler selection and outside these policies. Selecting it asserts that the authored asset implements the attached portable function's semantics; its numerical agreement with the reference is qualified by the consuming application's empirical precision gate, not established by Seismic. It cannot silently fall back.

## Source authority and policies

The first applicable portable body defines reference operation order, casts, accumulation, rounding, and exceptional-value behavior. The interpreter and portable construction use the same versioned scalar recipes. Each declared floating step rounds at its declared output. Source exp_fast and exp name the same computation; an approximate exponential is a separate physical choice. Source authors cannot assert a tolerance or attach evidence.

The caller chooses one policy for the complete result, final writable input state, source failures, and progress:

| Policy | Selection requirement |
| --- | --- |
| Exact | The physical execution preserves the reference contract, including permitted outcomes and representation rules. |
| Bounded | A derived whole-entry relation establishes the declared absolute, relative, ULP, and special-value limits. Exact implementations also satisfy it. |
| Unconstrained | Floating deviation may be unbounded only after discrete values, memory, effects, failures, and progress are established. |

Bounded tolerances do not themselves prove an implementation applicable. The supported alternative analysis has no general nonzero-error bound, so an unresolved bounded alternative stays unselectable. Search effort and diagnostic comparison cannot turn it into an incumbent.

## Construction and applicability

The required source body is implemented by the compiler's source-directed operations, complete value transport, and structured continuation. Its numerical applicability follows those actual constructions; no second whole-program replay is required to authorize the same computation. The compiler must still implement each scalar recipe, tensor operation, store, call, branch, repeat, failure, and native instruction faithfully. A source-body identity is not an assertion that arbitrary emitted code is exact.

Selecting a different body or physical operation requires analysis of its actual relationship to the required computation. An exact helper replacement preserves complete call behavior: results and descriptors, observable writable state, failure cause and prefix, and progress for every admitted input. The parent imports that applicability through the actual call. Local floating tolerances cannot simply be conjoined: later operations or discrete decisions can amplify or change a helper's error. An unsupported relation remains Pending, including under Unconstrained when discrete or effect behavior is unresolved.

One construction-owned numerical applicability determines both solver admission and final retention. Its accepted region and explanation are derived together. An unresolved physical alternative remains in the structural domain but is not selectable. Neither factories, native signatures, mode labels, observations, nor callers may attach an independent approval guard. Precision policy participates in specialization identity; an applicability result is never reused under a different policy without re-evaluation.

## Native numerical contract

Native formation preserves the operation recipes and compiler options assessed before selection. Contraction, reassociation, FTZ/DAZ, approximate math, changed accumulation width, storage rounding, and target intrinsics are distinct physical choices. A typed native Cast is not automatically the source-defined conversion. CPU, Metal, and CUDA emitters must preserve the selected recipe's rounding boundaries, narrow payloads, signed zero, subnormals, infinities, and NaN behavior. A backend unable to do so cannot advertise the required source mapping.

Native artifact and numerical-environment identities protect reuse of an assessed formed implementation. Reflected signatures and matching result types alone do not establish its numerical behavior.

## Observation and comparison

The interpreter and comparator diagnose and test implementations; they do not create selectable scope. A completed reference outcome owns returned values, final input state, and permitted outcome information. Comparison uses the actual completed native observation and checks every subject's geometry and read-only bytes before numeric differences. Integer and Boolean subjects remain exact under every floating policy.

For finite floating elements, diagnostic comparison records absolute error, scale-stabilized relative error, and published-dtype ULP distance. The bounded comparison envelope is abs_error <= atol + rtol * max(abs(reference), relative_floor), together with any ULP and special-value rules. NaN, infinity, signed-zero, and subnormal changes are counted separately. A passing sample, corpus, feedback observation, or benchmark never grants numerical applicability. A definite mismatch from an admitted candidate is a compiler or backend defect, not a reason to widen tolerance or gather more samples.

## Intentional limits and acceptance

Alternative analysis is conservative and may remain Pending. The general required source construction must remain available for every legal entry, subject to real target and resource limits. If an ordinary operation cannot be constructed faithfully, that is an implementation gap to fix, not numerical evidence to seek. Compiler selection has no empirical qualification side channel; empirical qualification applies only to explicitly selected native implementations, outside compiler selection.

- Exact selection never accepts unknown or nonzero deviation.
- Bounded selection accepts an exact implementation or an established sufficient whole-entry bound; otherwise the alternative remains unresolved.
- Unconstrained selection never relaxes discrete values, memory, effect order, failure prefix, or progress requirements.
- Calls import selected child applicability; a Pending child cannot become Exact merely because its parent used the required body.
- Observations may reveal defects but never create or enlarge accepted regions.
- Direct native selection remains explicit and separate from policy-checked compiler selection.
