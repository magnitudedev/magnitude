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
| Engine | Device bandwidth, planned decode demand, fixed decode costs, formula |
| Service | Requested depths, assessment lifecycle, identity, caching |
| ACN | Local model ranking only |

Estimation is arithmetic: the model's planned decode demand over the selected device's memory
bandwidth. It opens no device and runs no model, benchmark or tuning. Every fitting model has an
estimate.

## Scope

Each sample models plain autoregressive decode of one conversation producing one token at its
requested depth. It excludes prompt processing, speculative acceptance and concurrent scheduling.
The serving profile owns capacity and fit; performance samples create no serving configurations.

## Device bandwidth

Resolution is total and uses only facts device discovery already holds:

1. **Reported.** For a discrete GPU whose driver reports memory clock and bus width without opening
   the device (CUDA), double-data-rate clock × bus width. An integrated device's shared memory has
   no meaningful board clock, so it is never taken from the driver.
2. **Published.** A table of published peak bandwidth keyed by the normalized name the driver
   reports. Chip bins sharing a name are told apart by CPU core count, and boards sharing a name by
   the memory the device allocates from. Where the device's facts cannot tell configurations apart,
   the table holds the lowest.
3. **Assumed.** The low end of the device's class: a GPU with its own memory, a GPU allocating host
   memory, or a CPU.

An unrecognized device therefore errs slow. Order between models on one device depends only on
their demand and is exact whatever the bandwidth.

## Calculation

- Demand comes from the same allocation-free execution plan a load prepares: the bytes one decode
  step streams independent of depth (weights, a routed layer's selected experts only, norms,
  routers, recurrent state, rows, logits and per-layer table uploads), the history it reads per
  token of context, and its entry calls. History reads grow with depth, up to the window of a
  window-domain layer; a layer sharing another layer's history reads the source's. Recurrent state
  is charged once per token.
- One fixed set of decode costs applies to every device and backend: a per-step cost, a
  per-entry-call cost, the share of bandwidth weight streaming reaches, and the share decode
  attention reaches over history. Nothing is fitted per device, and no device is measured.
- The step time is the step cost, plus the entry calls' cost, plus the depth-independent bytes over
  the weight share of bandwidth, plus the history reads over the attention share. The estimate is
  its reciprocal.

Every result has finite positive rates, one per requested depth in ascending order.

## Identity and caching

Estimates are cached with their exact assessment under the assessment environment identity, which
includes the engine build (covering the decode costs and the bandwidth table) and the resolved
bandwidth with its source. A warm exact-cache hit performs no estimation.

## Conformance

- Estimation reads no tensor payload, opens no device and runs no model decode.
- Increasing planned traffic cannot improve an otherwise identical estimate.
- Samples are strictly ordered, one per requested depth, ending at the served context.
- Recurrent state is charged once per token, never multiplied by depth.
- Speculative heads do not change the plain-decode estimate.
- Decode costs are the same on every device; no estimate depends on a per-device fit.
