---
applies_to:
  - inference-v4/seismic/**
  - inference-v4/seismic-std/**
  - inference-v4/engine/**
  - inference-v4/docs/seismic/**
---

# Seismic numerical precision

Numerical precision is an observable compilation contract and a hard selection constraint. It is
not a source-level permission, a backend-wide fast-math mode, or a performance penalty.

## Authority

The first applicable portable function body defines reference operation order, casts,
accumulation, rounding, and exceptional-value behavior. Unary reference transcendental operations
are defined by the versioned recipe in `seismic-lang`: one ordered graph of ordinary f32/u32/i32
primitive steps with rounding after every floating step. The semantic interpreter evaluates that
graph and kernel construction instantiates that same graph; neither owns another formula. “Exact”
means zero deviation from this language-defined recipe. It does not claim the recipe is the
correctly-rounded mathematical real function. `exp_fast` is explicitly approximate and is never
silently treated as reference `exp`. FMA, min, and max retain their direct multi-operand primitive
semantics rather than entering the unary recipe.

Other portable bodies and target implementations are alternatives; their presence asserts
availability, not equivalence.

The compilation caller owns acceptable output deviation. A policy is part of specialization
identity and is either exact, explicitly bounded, or unconstrained. Unconstrained selection is for
exploration and carries no production numerical guarantee.

Kernel authors express computations and ordinary domain facts. They do not assert tolerances or
evidence. Structural partial-value correctness is independent from numerical precision.

## Evidence and selection

Every selected execution has one numerical assessment: exact, proven, qualified, or unknown.

- Exact evidence denotes the reference computation or proved zero deviation.
- Proven evidence is a conservative compiler bound over an identified input domain.
- Qualified evidence is an elementwise comparison of one complete witness with the reference on
  an identified corpus and native numerical environment.
- Unknown is never interpreted as zero and is selectable only by an unconstrained policy.

Numerical admissibility is a hard solver constraint encoded before solving: a predicate over
decisions and invocation symbols that is true exactly when the derived transfer satisfies the
policy analytically or matching evidence exists. It is never checked after selection. Performance
is optimized only within the admissible family. Evidence is keyed by exact implementation identity
and decision assignment, numerical-environment identity, semantic domain predicate, policy, corpus, and
qualification version; it does not transfer across any of these.

Every implementation's transfer is derived from its actual operations, order, data types,
reductions, approximations, and intrinsic semantics; it is never a manually asserted label.
Transfers compose through spliced calls, repeats, reductions, and dtype conversions with one
model, so local and accumulated error share it. When no admissible assignment exists over some
part of the target domain, preparation fails with `NumericalPolicyInfeasible`. Search effort
limits optimization only and cannot turn a feasible program into a no-incumbent failure.

A qualification retains the exact bounded policy used for elementwise checking. It may satisfy a
later policy only when every tolerance and special-value requirement is at least as permissive and
the input-domain facts are identical. Aggregate maxima alone never reconstruct a combined
absolute/relative envelope.

## Metrics

For each finite output element, comparison records absolute error, scale-stabilized relative error,
and published-dtype ULP distance. Acceptance uses the combined envelope
`abs_error <= atol + rtol * max(abs(reference), relative_floor)` plus any ULP limit. NaN, infinity,
signed-zero, and subnormal changes are counted and governed independently.

The same comparison semantics govern qualification, backend sweeps, and engine validation.

## Backend guarantees

Fast math is off unless a concrete implementation operation is admitted by the policy through
its recorded transfer. Reassociation, approximate transcendentals, contraction, storage and
accumulation dtype, publication rounding, flush-to-zero, and backend intrinsics are distinct
numerical effects recorded during kernel construction. Native compilation reproduces exactly the
choices assessed before selection; it cannot enable a relaxation the transfer does not record.
Each ordinary operation produced by a reference recipe is a separate rounding boundary. CPU
workers enter a saved/restored strict IEEE floating environment (nearest-even and gradual
underflow), Cranelift receives distinct strict operations, CUDA emits rounding-qualified PTX
without `.ftz`, and Metal uses precise non-fast-math operations. A backend unable to preserve those
boundaries cannot advertise the reference path; contraction, reassociation, FTZ/DAZ, or native
approximate math must be represented by a different implementation and numerical transfer.

## Runtime and identity

Preparation accepts an immutable evidence catalog; a record participates only through the
admissibility predicate of the implementation it keys. Every executable variant carries its
numerical assessment (exact, proven, qualified with evidence keys, or unknown) and the policy
identity it was prepared under. Runtime never re-evaluates numerical legality.

Changing the policy or available evidence changes the preparation identity.
`NumericalEnvironmentIdentity` is distinct from compatibility and per-open
execution-profile identities. Reprofiling unchanged numerical behavior cannot
invalidate evidence, while any change to emitted numerical mode or the native
kernel's numerical contract does.

## Intentional limits

Static proof coverage is conservative. A numerical effect without a checked transfer rule remains
unknown; the compiler selects the exact reference path or requires matching whole-witness
qualification. It never invents a bound to obtain a faster result.

## Acceptance criteria

- Exact compilation cannot select nonzero or unknown deviation.
- Bounded compilation accepts only exact, sufficient proven, or explicitly permitted matching
  qualified evidence.
- Several local deviations are accepted only through a whole-entry assessment.
- A threshold change may change the selected witness, but cannot widen evidence implicitly.
- Unknown analysis and special-value changes cannot disappear into a performance objective.
- Source syntax has no `admit` escape hatch; partial-value checking is numerical-policy agnostic.
