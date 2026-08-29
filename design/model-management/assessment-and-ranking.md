---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-api/**
  - packages/icn/src/hardware/**
  - packages/acn/src/local-model-**
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/model-setup/**
  - web/src/components/local-model-onboarding.tsx
  - web/src/components/model-center.tsx
---

# Model assessment and ranking

Terms follow [Model-management terminology](./terminology.md). Native mechanics follow
[Hardware calibration and model assessment](../icn/calibration-model-assessment.md).

## Ownership

| Concern                                                       | Owner   |
| ------------------------------------------------------------- | ------- |
| Native compatibility, memory, placement, performance          | ICN     |
| Native serving-configuration resolution and validation        | ICN     |
| Profile set, demand reconciliation, normalized ranking scores | ACN     |
| Assessment filtering, scheduling, and native concurrency      | ICN     |
| Rendering                                                     | Clients |

## Pipeline

```mermaid
%%{init: { "theme": "dark", "themeVariables": { "background": "#16161d", "primaryColor": "#272733", "primaryTextColor": "#fafafa", "primaryBorderColor": "#a1a1aa", "lineColor": "#e4e4e7", "textColor": "#fafafa", "actorBkg": "#272733", "actorBorder": "#a1a1aa", "actorTextColor": "#fafafa", "actorLineColor": "#a1a1aa", "signalColor": "#e4e4e7", "signalTextColor": "#fafafa", "labelBoxBkgColor": "#272733", "labelBoxBorderColor": "#a1a1aa", "labelTextColor": "#fafafa", "loopTextColor": "#fafafa" } } }%%
sequenceDiagram
    participant Catalog
    participant Inventory
    participant ACN
    participant ICN
    participant Resolver
    participant Ranker

    Catalog->>ACN: Reviewed model identity and configuration (bundle, profile)
    ACN->>ACN: Preserve reviewed profile

    ACN->>ICN: One request containing every distinct pending bundle target
    ICN->>ICN: Resolve and validate bundle and profile
    ICN->>ICN: Assess compatibility, memory, and performance
    ICN-->>ACN: Started(totalTargets)
    ICN-->>ACN: Request-correlated Result as each target finishes
    ICN-->>ACN: Completed(totalTargets)

    ACN->>ACN: Attach evidence to submitted catalog model

    ACN->>Resolver: Publish configuration and assessment
    Resolver->>Resolver: Resolve catalog model configuration
    ACN->>Ranker: Derive normalized scores for Fits catalog configuration
```

ACN supplies the reviewed serving profile for each catalog model. ICN resolves and validates the
exact bundle and profile. Each ACN request carries one or more unique profiles. The response is
correlated first by exact request identity and then by exact profile identity within that request.
ACN attaches returned evidence to the canonical catalog model ID. A serving configuration is a
value, not a separately addressable entity.

ICN owns the rejection proof comparing exact, content-deduplicated tensor storage with aggregate stable
physical capacity. Uncertain bundles proceed. File/download size is not rejection evidence.
GGUF-derived identity and tensor storage share one optimistic inspection cache per immutable
component. Package capabilities compose the inspected primary artifact with exact component roles,
relationships, and native projector inspection. An absent or malformed cache entry reparses the
component and cannot change artifact presence.

## Assessment service

ACN exposes one assessment executor accepting exact bundles and profiles. It owns the scoped demand
lifecycle, result decoding, stream-lifecycle validation, correlation, and publication. It sends the
entire target set once; it owns no fixed batch size, native concurrency, or capacity filter. ICN
owns admission, cache reuse, filtering, scheduling, concurrency, and native deadlines.
Every executor request carries one bundle and one or more unique profiles so ICN can share model
opening across their assessment. Missing, duplicate, or unrequested profile outcomes violate the
assessment operation contract; they fail the operation rather than becoming a model compatibility
or capacity state.
One ACN local-model assessor owns demand for each catalog model's desired configuration and its
installed effective configuration when different. Desired catalog assessment supports ranking
and acquisition; effective catalog assessment supports current serving.
Ranking, provider-offering, and local-model projections consume its state and
never invoke assessment independently. The ranker consumes only release-catalog models. It scores
each desired release configuration and, while an update is incomplete, the distinct installed
effective configuration of that same model. Each configuration uses its reviewed catalog profile
and may publish scores only after its exact assessment is `Fits`. Installed
artifacts that are not attributed to a catalog model remain package inventory; they do not become
callable models with fabricated identities. An installed catalog target remains the effective
configuration of the same canonical catalog model when its desired bundle changes.

ICN persists every completed exact profile result, including `DoesNotFit`, and performs
single-flight native work. Repeated reads consume current results and do not trigger native
assessment.

The assessor is a scoped background owner. Constructing ACN services publishes initial
observable state without waiting for assessment; only the admitted operation owner awaits its
bounded native request.

The coordinator publishes one atomic snapshot containing every current demanded configuration and
exactly one lifecycle state: `Discovering`, `Assessing`, `Ready`, or `Failed`. Consumers never join
assessment entries, progress, and completion from independent reads. Each demand is either being
assessed or carries one terminal result for its current semantic key.

During `Assessing`, `totalTargets` is the number of distinct demanded configurations admitted to
that cycle and `completedTargets` advances as each configuration receives its native result. Native
bundle/profile sharing may complete several demands from one streamed result, but transport batches
and catalog row counts never manufacture progress. `Ready` means every current demand has a
terminal result. `Failed` is the terminal outcome of one exact cycle, not a sticky coordinator
flag; reconciliation that removes or supersedes its failed demands also removes that failure.
Ranking remains pending while assessment runs. Assessment failure is terminal and never appears as
an active ranking step.

## Demand and invalidation

The assessor reconciles current snapshots rather than treating notifications as semantic
changes:

```text
source invalidation
       |
       v
coalesce -> read snapshots -> compute semantic assessment keys
                                      |
                          +-----------+-----------+
                          |                       |
                       unchanged                changed
                          |                       |
                    retain result          assess exact key
                                                  |
                                      publish only while current
```

One private semantic assessment key contains the complete serving configuration, resolved immutable
assessment material, stable topology and capacity identity, native build and enabled backends,
calibration identity, assessment method and policy, and requested performance-depth policy.

Download bytes, transfer speed, model-download or package-attempt identity, package or inventory revision, catalog
presentation, live free memory, provider state, slot state, runtime residency, and client activity
are not assessment inputs. Revisions may require a reread; revision inequality never admits native
assessment by itself.

Reconciliation is serialized and invalidations are coalesced. It assesses only new or changed keys,
preserves terminal results for unchanged keys, and removes state only when catalog and
catalog demand no longer includes the configuration. Completion rechecks the semantic key before publication;
a result for a superseded key is discarded and reconciliation continues without overwriting newer
state.

## Publication boundary

The ranker evaluates private inputs for catalog configurations with completed `Fits`
assessment. The client-facing projection does not publish a parallel candidate entity. It publishes
one `LocalModel` per exact bundle and annotates its assessed serving state with optional
`LocalModelRankingScores` for that exact configuration.

Performance evidence is an ordered set of samples for the same configuration. Samples above the
configured context are omitted, and the final sample is always the configured context.

Ranking scores are present only when the bundle is active in the rankable catalog.
Deprecated catalog configurations remain resolvable and assessable but receive no fabricated
intelligence, fidelity, or ranking scores.

`DoesNotFit` and `Incompatible` are completed evidence but are not selectable. Missing,
`Assessing`, canceled, or defective work is not published as a successful empty portfolio.
Installed packages remain present in the [local-model product projection](./local-model-product-projection.md)
independently of assessment and offering publication.

## Assessment lifecycle

ACN owns one serialized assessment coordinator. It groups current demands by structural bundle,
submits all distinct pending bundle targets in one operation, and combines unique profiles within a
target. While admitted work for the current
semantic key is running, its internal state is `Assessing`. Completion publishes `Failed`, `Fits`,
`DoesNotFit`, or `Incompatible` for that key. Serialized reconciliation plus key validation makes
stale completion unrepresentable. Unchanged configurations retain their terminal state while
another configuration is assessed.

The product projection never publishes an absent assessment as a configuration result. An
installed model without a terminal result for its decided configuration has model-level readiness
`Assessing`; an assessed model contains exactly that configuration and its terminal result.

Each streamed target result is published immediately. If the stream ends without `Completed`,
already received terminal results remain published and only unresolved targets become `Failed`.
An assessment-operation defect publishes `Failed` with its typed failure. It is not converted into
`DoesNotFit` or `Incompatible`, and it cannot replace the local-model collection with an empty
snapshot.

## Ranking scores

Score normalization and client preference utility are defined by
[Local model ranking](./ranking.md). This document owns the assessment evidence and publication
lifecycle consumed by the ranker.

## Invalidation

Ranking-score reuse requires unchanged catalog content, profiles, stable topology and capacity,
native build and backends, hardware calibration, assessment method, and ranking policy. Live memory
availability does not invalidate assessment or ranking scores. Current loadability remains a
separate presentation fact; stable assessment can coexist with insufficient live admission
headroom.

## Loading

Assessment proves compatibility with stable capacity for one sequence. Loading repeats native
planning, determines execution-plan sequence capacity, and applies fresh availability admission.
Cached assessment never authorizes loading.

## Conformance

- Every release-catalog choice publishes one reviewed configuration whose profile does not exceed
  its exact bundle maximum.
- Installed packages without catalog attribution remain artifact inventory and are not assigned a
  callable model identity.
- A package already participating in a catalog bundle is assessed only through that
  bundle. Separate speculative companions are never submitted again as standalone targets.
- All missing profiles for one bundle are submitted together.
- All distinct pending bundles are submitted in one ICN request.
- Assessment progress counts the exact demanded configurations in the active cycle and advances as
  each receives its native result; it is not a transport-batch count or inferred catalog total.
- Assessment entries and lifecycle are one atomic snapshot; no consumer can combine different
  generations of results, progress, or completion.
- Removing or superseding the demands from a failed cycle removes that cycle failure.
- A truncated stream retains every result received before truncation and fails only unresolved targets.
- Equivalent concurrent misses perform one native assessment.
- Download progress and semantically equivalent newer source revisions perform no assessment.
- A changed semantic key reassesses only the affected configurations.
- Provider-offering and local-model projection never invoke assessment.
- Assessment completion cannot publish against a superseded semantic key.
- Assessment results correlate by exact request and profile identity, never by configuration-record comparison.
- Ranking failure never becomes a successful empty result.
- Clients display a bounded 25K-to-75K expected-speed range without context-variant candidates.
- Ranker inputs remain private; client-visible ranking scores preserve the canonical catalog model
  ID and exact configuration.
- Loading never treats cached assessment as admission authority.
- ACN startup and service publication never wait for model assessment.
- Onboarding keeps installed and downloadable model groups at a stable layout height while making
  every presented model reachable by keyboard navigation and pointer scrolling.
- Onboarding ranks downloadable models from connection-scoped controls while keeping installed
  models separate and rendering stable fit independently from current loadability.
