---
applies_to:
  - inference/engine/executor/src/assessment/**
  - inference/engine/src/assessment.rs
  - inference/service/server/src/assessment/**
---

# Generation-performance estimation

## Contract

The engine estimates single-user decode throughput at several occupied-context depths for one exact
assessed model. The ordered estimates are advisory ranking inputs. They never change capacity,
authorize loading, or replace observed runtime timing.

| Layer | Responsibility |
|---|---|
| Engine | Measurement basis, planned decode demand, formula, bounds, confidence |
| Service | Requested depths, assessment lifecycle, identity, caching |
| ACN | Local model ranking only |

Estimation is analytical: the model's planned decode demand multiplied by the environment's
measured generic operation costs. It runs no model, no per-model benchmark and no tuning.

## Scope

Each sample models plain autoregressive decode of one conversation producing one token at its
requested depth. It excludes prompt processing, speculative acceptance and concurrent scheduling.
The serving profile owns capacity and fit; performance samples create no serving configurations.

## Measurement basis

The basis is a fixed, model-free plan (see the measurement basis design): every operation class of
a plain decode step (dense and paired projections, attention, recurrent steps, routed experts, row
ops such as post-norms and per-layer inputs, readout, sampling, conversions, and the host gather and
upload of a host-resident table's rows), plus the launch dependency between consecutive calls and
the fixed per-step submission, timed with shipped default configurations at synthetic sizes. Each
class fits a cost model in the quantities a model's demand supplies: per launch, per byte,
projection cost by output rows, attention history cost by head geometry and depth, and a
weight-format factor per resident representation. The arithmetic parameters that move production
speed most (`INT8`, `PARTS`, `SLICES`) are each varied alone from the default, screened by one
sample, and the fastest is timed. Every repeated sample is kept; fits are clamped to physical values.

## Calculation

- Demand comes from the same allocation-free execution plan a load prepares: the exact resident
  representation, bytes and launches of every term at each depth. History reads grow with depth,
  up to the window of a window-domain layer; a layer sharing another layer's history reads the
  source's.
- The step time sums each class's median cost; the estimate is its reciprocal.
- Bounds come from measured variation only: each class's slowest and fastest sample relative to its
  median scale its time for the lower and upper rates. No fixed band is applied.
- Confidence comes from the relative range `(upper − lower) / estimated` with thresholds fixed once
  in the estimator: high ≤ 5%, moderate ≤ 15%, low otherwise.

Every result has finite positive rates with `lower <= estimated <= upper`, one per requested depth
in ascending order. A demand term outside the basis makes the model `Incompatible`; a missing
measurement or invalid arithmetic fails the assessment. Neither is a partial result.

## Identity and caching

Estimates are cached with their exact assessment under the assessment environment identity, which
includes the basis digest and the engine build. A warm exact-cache hit performs no estimation.

## Conformance

- Estimation reads no tensor payload and runs no model decode.
- Increasing planned traffic cannot improve an otherwise identical estimate.
- Samples are strictly ordered, one per requested depth, ending at the served context.
- Recurrent state is charged once per token, never multiplied by depth.
- Speculative heads do not change the plain-decode estimate.
