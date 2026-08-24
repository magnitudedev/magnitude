---
applies_to:
  - packages/acn/src/local-model-recommendation-policy.ts
  - packages/acn/src/local-model-recommendation-policy.test.ts
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/options.ts
  - cli/src/features/model-setup/chooser.tsx
---

# Model recommendation

ACN derives deterministic recommendation intents from active release-catalog configurations with
completed `Fits` assessments. Every intent applies a different weight vector to the same factors;
there are no intent-specific gates, tiers, relative thresholds, or alternate ranking rules.

## Inputs

A fitting release-catalog candidate is rankable and supplies every factor. Speed uses the expected
sample at 50K occupied context, bounded by the configured context for shorter models.

For normalized capability `C`, speed `S`, fidelity `F`, and memory efficiency `M`:

```text
C = clamp(capability / 100)
F = clamp(fidelity / 100)
M = clamp(1 - loaded memory / stable capacity budget)

r(v) = v / 40                 when 0 <= v <= 40
     = 1 + ln(v / 40)         when 40 < v <= 100

S = r(clamp(generation speed, 0, 100)) / r(100)
```

Curated fidelity is independent of runtime acceleration. Download and disk size do not
participate in any recommendation utility.

## Intents

Every intent maximizes the same weighted geometric utility:

```text
U(C,S,F,M; wc,ws,wf,wm) = C^wc * S^ws * F^wf * M^wm
```

| Intent | Capability | Speed | Fidelity | Memory |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 0.40 | 0.30 | 0.20 | 0.10 |
| Smartest | 0.60 | 0.10 | 0.30 | 0.00 |
| Fastest | 0.30 | 0.60 | 0.05 | 0.05 |
| Lightweight | 0.10 | 0.10 | 0.10 | 0.70 |

Each row sums to one. A weak factor reduces utility proportionally; no secondary heuristic can
override the weighted result.

## Portfolio

Selection is ordered and greedy: Balanced chooses first, followed by Smartest, Fastest, and
Lightweight. Each intent chooses its highest-utility canonical model that has not already been chosen
by an earlier intent. A model appears at most once. An intent is omitted only when no
unselected candidate remains. Canonical model ID is the only tie-breaker. Identical
inputs always produce identical results.

## Conformance

- All four intents use the same normalized factors and geometric utility implementation.
- Intent behavior differs only through the documented weight vectors.
- Download and disk size do not affect any intent.
- No intent has a capability gate, speed floor, memory tier, or improvement threshold.
- Portfolio assignment follows Balanced, Smartest, Fastest, Lightweight and never repeats a
  configuration.
