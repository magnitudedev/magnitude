---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-api/**
  - packages/icn/src/hardware/**
  - packages/acn/src/local-model-assessments.ts
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/local-model-recommendation-policy.ts
  - packages/acn/src/local-models.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/model-setup/**
---

# Model assessment and recommendation

Terms follow [Model-management terminology](./terminology.md). Native mechanics follow
[Hardware calibration and model assessment](../icn/calibration-model-assessment.md).

## Ownership

| Concern | Owner |
|---|---|
| Native compatibility, memory, placement, performance | ICN |
| Profile set, orchestration, recommendation policy | ACN |
| Rendering | Clients |

## Pipeline

```text
recommendable target
  -> exact tensor-storage rejection proof
  -> local profile: 100K (bounded by target maximum)
  -> cached or native ICN assessment with 25K, 50K, 75K, and full-context speed samples
  -> completed configuration candidates
  -> recommendation portfolio
```

The rejection proof compares exact, content-deduplicated tensor storage with aggregate stable
physical capacity. Uncertain targets proceed. File/download size is not rejection evidence.

## Assessment service

ACN exposes one assessment service accepting a batch of exact targets and profiles. It owns the
scoped lifecycle, deadline, ICN batching, result decoding, cardinality checks, and finalization.
One catalog projection assesses release-catalog and discovered installed targets together;
recommendation policy consumes the release-catalog candidates from that projection.
Release-catalog and discovered installed targets use the single 100K product profile, bounded by
the target maximum. A speculative pair uses the lower component maximum.

ICN persists every completed exact profile result, including `DoesNotFit`, and performs
single-flight native work. Repeated reads consume current results and do not trigger native
assessment.

Candidate and recommendation projection is a scoped background consumer. Constructing
ACN services publishes their initial observable state without waiting for assessment; only the
operation owner awaits its bounded request.

## Publication boundary

A catalog candidate exists only for one completed `Fits` configuration. It contains its exact
target, serving configuration, profile, assessment environment, memory, performance, capability,
acquisition, and source evidence.

Candidate performance is an ordered set of samples for the same configuration. Samples above the
configured context are omitted, and the final sample is always the configured context.

Recommendation evidence is present only when the target comes from the recommendable catalog.
Discovered installed targets remain selectable catalog candidates without fabricated intelligence,
fidelity, or quality values.

`DoesNotFit` and `Incompatible` are completed evidence but are not selectable candidates. Missing,
`Assessing`, canceled, or defective work is not published as a successful empty portfolio.
Installed targets remain present independently of assessment and offering publication.

## Assessment lifecycle

ACN exposes one shared assessment query. Its observable state is `Unassessed`, `Assessing`, or
`Assessed`. `Assessing` is an empty marker scoped to the admitted Effect. Finalization clears or
completes it on every exit path, and overlapping owners are serialized so stale completion is
unrepresentable.

An assessment endpoint defect fails the owning operation. ACN retains prior complete publication when
available and never converts the defect into `DoesNotFit`, incompatibility, or an empty catalog.

## Recommendation portfolio

ACN selects at most one configuration for each intent:

| Intent | Objective |
|---|---|
| `balanced` | Overall capability, speed, memory, fidelity, and download utility |
| `best_quality` | Highest useful capability and fidelity within resource guards |
| `fastest` | Highest useful generation speed within capability guards |
| `lightweight` | Highest useful capability within a low-memory tier relative to stable hardware capacity |

Selection is deterministic for identical inputs and uses stable identity as its final tie-breaker.
A recommendation references a candidate; it does not create another model identity.

A candidate is usable only when its full-context expected generation speed is at least 8 tokens per
second. Ranking, Fastest selection, and relative speed comparisons use the sample at 50K context,
bounded by the configured context for shorter models. Clients present the expected-speed range from
the samples at 25K and 75K, likewise bounded by the configured context, without treating those
samples as separate configurations.

The Lightweight tier admits configurations whose complete predicted loaded memory uses at most
20% of each participating physical memory domain's stable post-reserve capacity. Within that tier,
capability precedes memory, fidelity, speed, and download size. The selected configuration must also
use at least 20% less loaded memory than Balanced. If no distinct configuration satisfies both
boundaries, the portfolio omits Lightweight rather than substituting the absolute smallest model.

Lightweight eligibility is independent of the strongest feasible model's capability. Adding a
high-capability heavyweight candidate therefore cannot disqualify or downshift an otherwise
unchanged Lightweight tier.

## Invalidation

Recommendation reuse requires unchanged catalog content, profiles, stable topology and capacity,
native build and backends, hardware calibration, assessment method, and recommendation policy.
Live memory availability does not invalidate assessment or recommendation.

## Loading

Assessment proves compatibility with stable capacity for one sequence. Loading repeats native
planning, determines execution-plan sequence capacity, and applies fresh availability admission.
Cached assessment never authorizes loading.

## Conformance

- Every release-catalog and discovered installed target has one 100K profile, bounded by its exact
  target maximum.
- All missing profiles for one target are submitted together.
- Equivalent concurrent misses perform one native assessment.
- Recommendation generation never replaces a valid portfolio with a defect-derived empty result.
- The usability floor uses full-context performance; ranking and relative comparisons use the
  bounded 50K sample.
- Clients display a bounded 25K-to-75K expected-speed range without context-variant candidates.
- Lightweight is hardware-relative, capability-maximizing within its memory tier, and independent
  of the capability ceiling outside that tier.
- Candidate identity remains the serving-configuration identity.
- Loading never treats cached assessment as admission authority.
- ACN startup and service publication never wait for model assessment.
