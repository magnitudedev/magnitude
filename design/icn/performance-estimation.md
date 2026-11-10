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
| Seismic topology/runtime | Hardware graph and measured path rates |
| Engine | Planned decode demand, traffic cost, fixed decode costs, formula |
| Service | Requested depths, assessment lifecycle, identity, caching |
| ACN | Local model ranking only |

Estimation is arithmetic over previously measured paths. Establishing the environment explicitly
measures the requested topology scope, reusing persisted evidence with matching hardware and driver
keys. Per-model estimation opens no device and runs no model, benchmark or tuning.

## Scope

Each sample models plain autoregressive decode of one conversation producing one token at its
requested depth. It excludes prompt processing, speculative acceptance and concurrent scheduling.
The serving profile owns capacity and fit; performance samples create no serving configurations.

## Measured traffic

The engine cost function prices explicitly named path bytes and executor launches using a
`MeasuredTopology`. Missing scoped evidence is an error; no published table, device-name rule,
reported peak or class constant supplies a rate.

Decode demand currently aggregates allocations. Its estimate therefore uses the slowest measured
path in each applicable storage placement. Upload bytes are distinguished from resident bytes;
history uses device storage. This is conservative with respect to allocation placement within the
estimator's formula, not a bound on observed model latency. It never invents a proportional split
between stores or claims that the worst path is the actual allocation. Explicitly attributed
traffic uses the same cost function directly.

## Calculation

- Demand comes from the same allocation-free execution plan a load prepares: the bytes one decode
  step streams independent of depth (weights, a routed layer's selected experts only, norms,
  routers, recurrent state, rows, logits and per-layer table uploads), the history it reads per
  token of context, and its entry calls. History reads grow with depth, up to the window of a
  window-domain layer; a layer sharing another layer's history reads the source's. Recurrent state
  is charged once per token.
- One fixed set of decode costs applies to every device and backend: a per-step cost, a
  per-entry-call cost, the share of bandwidth weight streaming reaches, and the share decode
  attention reaches over history. These fitted estimator terms remain unchanged by the topology
  migration; replacing them is separate estimation work.
- The step time is the step cost, plus the entry calls' cost, plus the depth-independent bytes over
  the weight share of bandwidth, plus the history reads over the attention share. The estimate is
  its reciprocal.

Every result has finite positive rates, one per requested depth in ascending order.

## Streamed models

A model whose routed experts stream through an expert bank is estimated as every other model is:
from its plan's structure and this machine's stored figures. No offline per-model data enters it:
no model starts or needs a measurement, and nothing a model's own loads record enters its estimate
or its identity, so a model never loaded has the same estimate as one loaded many times.

- **What is measured is fixed per backend.** The measurement job's standing requests are the
  executor's cost walk, what its recorded graphs hold, the model store's read line, the device's
  convert at one reference record form, and, where a bank can have a host tier (a discrete device),
  the host memory's read line with every CPU reading and a CPU site's lines at one reference form.
  They are measured once per machine and build, first and in one job: a 10 s budget from the job's
  start (discovery, probes and kernel forming within it), whatever the machine and whatever time
  the job is given, which their points divide. Each point samples for its part of the time left
  beside the fewest passes (making, a lead and a sample) the walk forecasts for the standing points
  after it, or until its median resolves, passing what it does not take on; none starts that would
  end past the budget or leave the required figures after it less than their fewest passes. A
  point the budget cannot hold is recorded unobtainable and priced from those read, as any
  unobtainable point is, so a slower machine gets less precision, never more time; a required
  figure, which nothing prices, is never recorded so: the budget ending before it leaves it
  unrecorded, and the next job resumes from the records stored. Every load, banked or not, plans
  and serves only on the executor's costs, waiting once per machine for the walk's required figures
  (the probe, the launch graph's times and the composition chain), every later figure priced from
  those read until it is. The walk keeps its order: the launch graph's host times wait for every
  compile to end, so they follow the points measured while the kernels form, and the costs complete
  as the formation ends either way. The job walks to the last required figure, then measures what a
  streamed load's bank is priced by (the read line, the convert, a host tier's lines), then the rest
  of the walk, whose later figures are priced closer without them, so what the budget leaves unread
  is what is priced closest. Loads' own requests (CPU sites, shrink moves, their forms' converts,
  tier copies, volume windows) are runtime: the job measures them after the standing requests in
  what is left of its time, and no estimate reads them; a load's point the job's time ends is
  measured by the next job, given longer where a job measured nothing.
- **Storage.** The model store's volume is priced by one read line, `λ + b/D`: the job reads a
  probe file it keeps in the store (256 MiB, written once) past the file cache, timing full reads
  (32 readers of 1 MiB at once: `1/D`) and lone 128 KiB reads (`λ`, what one takes past its bytes
  at `D`). It is keyed by the executor and the volume, of no model.
- **Expert lines.** Each streamed layer's GPU lines come from the executor's costs at its expert's
  formats and bytes, over host storage's route and over device memory; a host tier's record read
  is priced at the host memory's line; a file miss reads its stored bytes along the store's read
  line. With no host tier a file miss then lands through the device's convert, its stored bytes at
  the reference form's rate. A miss of a layer the plan gives a CPU site is served at the reference
  site's lines (the reference convert's record, a decode row's 8 choices): an expert's lines in
  proportion to the record's bytes, and the site's handshake per request. A form's own CPU lines,
  convert and observed landing lines are runtime and do not enter the estimate. The step is the
  resident plan's, so it does not price a bank plan's own launches or its copies, and no read is
  credited to or charged for a hint.
- **Misses.** One generic function of a layer's `E` and `k` gives the share `m̄(s)` of a step's
  distinct routed experts a bank of a share `s` of the layer's experts lacks. Routing is a top-`k`
  competition: a score is what persists over a text and what changes with the token, so an expert of
  persistence `g` is chosen with `p(g) = Φ((√c·g − τ)/√(1−c))`, `τ = Φ⁻¹(1 − k/E)`. The ranking holds
  the highest decayed counts (half-life 64 steps), each a Gamma of mean `p·W₁` and variance
  `p(1−p)·W₂`, past one threshold the bank's size sets, so `m(s; c) = 1 − E_g[p·P]/E_g[p]`. `c` is in
  no header and a text moves it, so `m̄` is the mean over `c` uniform on (0, 1): no fitted constant and
  no model's data. A step of `r` rows reads `D(r) = E·(1 − E_g[(1−p)^r])` distinct experts; the bank
  misses `m̄(slots/E)·D(r)` of them, spread over its steps as evenly as they fall, of which
  `m̄(host/E)·D(r)` are read from the file and the rest from the host tier. A layer's misses of a step
  are one request at its route, a share `q = 1 − (1 − m̄)^k` of a row's steps requesting, and no read
  is credited to prediction. Each `(E, k)`'s table is built once in a process.
- **The step** is the one streamed step model's on the resident plan's step, and the estimate is
  its reciprocal. It is a decode's steady state: it holds no first reads, which a decode's early steps
  pay and its later ones do not.

**A load's own pricing.** A streamed load prices its bank (growth, shrinks, its move budget and a
prompt group's decode wait) through the same step model, on `m̄` counted as one
more step of what its own decode read this session: after `N` counted steps a share prices at
`(prior + N·live)/(N + 1)`, misses over reads and requests over steps, a decode of several rows
counting its requests as single rows' (`1 − (1 − q)^r`). The bank grows no further than the depth
its own reads stop falling at. Nothing a load counts is stored, and nothing it counts enters
assessment.

## Identity and caching

Estimates are cached with their exact assessment under the assessment environment identity, which
includes the engine build, stable graph, and the
records of every figure an estimate reads (the executor's costs, the store's read line, the
reference convert and the host memory's line). Measurement timestamps, live usage and what loads record do not enter identity. A warm exact-cache hit performs no
estimation.

Every record that enters the identity revises the service's assessments as the measurement job
stores it, with no restart. No record prepares a model again: a preparation reads only the
structure, and nothing of it a record prices, so the device's graph memory (priced at its
backend's bound until recorded) enters the environment's identity, not the structure's, and is
applied when each model is finished. Every record finishes each model again over what it was
prepared with: arithmetic over its outlined forms and demand.

## Conformance

- Estimation reads no tensor payload, opens no device and runs no model decode.
- Increasing planned traffic cannot improve an otherwise identical estimate.
- Samples are strictly ordered, one per requested depth, ending at the served context.
- Recurrent state is charged once per token, never multiplied by depth.
- Speculative heads do not change the plain-decode estimate.
- Decode costs are the same on every device; no estimate depends on a per-device fit.
- No estimate reads a record of one model's loads; every fitting model, streamed or resident, is
  estimated once the standing figures are stored.
