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
accumulation, rounding, and exceptional-value behavior. Other portable bodies and target
implementations are alternatives; their presence asserts availability, not equivalence.

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

Numerical admissibility is a hard solver constraint. Performance is optimized only within the
admissible family. Evidence is whole-witness evidence and does not transfer to a different program,
entry, specialization, witness, backend environment, corpus, or compiler method.

Transfers compose through nested schedules, repeated loops, reductions, calls, and dtype
conversions; the composed assessment of the selected assignment is checked against the caller's
policy as one constraint. When no admissible assignment exists, the complete planning model is
reported infeasible; the compiler does not infer a cause from a rejected probe assignment. Search
effort limits optimization only and cannot turn a feasible program into a no-incumbent failure.

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

Global fast-math modes remain disabled. Reassociation, approximate transcendentals, contraction,
storage and accumulation dtype, publication rounding, and backend intrinsics are distinct
numerical effects. Realization must reproduce the choices assessed before selection; it cannot
introduce an unrecorded numerical freedom.

## Runtime and identity

Runtime selection accepts an immutable qualification catalog and ignores nonmatching records.
Every accepted witness is structurally audited against the current family and backend before it is
realized. The retained selection includes the policy assessment and qualification identity.

Changing the policy or available evidence invalidates reuse. Device numerical identity is separate
from performance-estimate identity.

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
