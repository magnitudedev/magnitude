---
applies_to:
  - inference/engine/executor/src/assessment/**
  - inference/engine/src/assessment.rs
  - inference/seismic/measure/**
  - inference/service/server/src/assessment/**
  - packages/acn/src/local-model*.ts
  - packages/client-common/src/utils/local-model-radar.ts
---

# Generation-performance estimation

## Contract

The engine estimates single-user decode throughput at several occupied-context depths for one exact
assessed model. The ordered estimates are advisory ranking inputs. They never change capacity,
authorize loading, or replace observed runtime timing.

| Layer | Responsibility |
|---|---|
| Seismic topology/runtime | Hardware graph |
| Seismic measure | Timing points, the stopping rule, the sustained clock, deadlines, records |
| Engine | Per-call decode demand, which points make up the executor costs, the estimate |
| Service | Requested depths, assessment lifecycle, identity, caching |
| ACN | Local model ranking; the measurement state a pending speed waits on |
| Clients | Presentation of each speed state |

Estimation is arithmetic over the selected executor's costs, derived from the records of its cost
points. Only the measurement job measures: a contained worker that opens the executor and
measures its cost points through Seismic's measurer, persisting each record under the executor's
hardware and driver and the kernel's identity; what is stored with matching keys is loaded and
only what is missing is measured. The service never measures: it derives the costs from what is
stored, and until every point has a record a fitting model's performance is `Pending`. A point
that ended two jobs is recorded unobtainable, so the costs complete without it; a model whose calls
need it estimates `Unavailable`. Per-model estimation opens no device and runs no model, benchmark
or tuning.

ACN carries the snapshot's measurement state (`Unmeasured`, `Measuring`, `Measured`,
`Failed { retryAt }`) on every `Pending` speed. Clients show a pending speed as this machine being
measured, or as a failed measurement with its retry time; an `Unavailable` speed shows its failure.
A fitting model without a speed estimate is ranked by intelligence and fidelity alone, after every
model with one, because utilities with and without the speed term are not comparable. It is never
ordered by name.

## Scope

Each sample models plain autoregressive decode of one conversation producing one token at its
requested depth. It excludes prompt processing, speculative acceptance and concurrent scheduling.
The serving profile owns capacity and fit; performance samples create no serving configurations.

## Measurement

Seismic's measurer times generic work and knows nothing of kernels or why a caller asks. A point is
a formed kernel at a configuration with its arguments, or a sealed chain of such calls, on one
executor; its measurement is a timing: the median, the interval reached and the sample count.

- Points are timed at the device's sustained clock, raised once per physical device, so backend
  views of one GPU share it. A CPU's clock is not raised.
- The job's kernels compile on the host while its points are timed: each point waits only for its
  own kernels, and a point timed on the device shares nothing with a compile. A point that touches
  the host (operands in host memory, or host time) is timed only once every compile has ended.
- Samples continue until an anytime-valid confidence interval for the median lies within ±10% of
  it. The estimate's own error is about ±15%, so finer precision would be discarded. Tuning's
  comparisons use the same session and statistics; there is one stopping rule.
- A point stops when its median is resolved; a request has no time budget. Only the service's hard
  job deadline bounds a hung job.
- A record is keyed by stable identities: the executor's physical device, backend and driver, the
  kernel's source identity, its configuration and the point. A matching record is loaded, and only
  the missing points are measured. Records are written atomically through a store the host
  implements and are never evicted.
- The measurer reports each point before it starts, so the host can mark the attempt. A point
  whose kernel cannot be formed is recorded unobtainable at once, since formation is deterministic
  for its key. A point whose timing fails, or whose operands cannot be made now (an allocation can
  fail for a moment's reason), is not recorded. The host records a point unobtainable only after it
  has ended two jobs.

## Executor costs

The executor costs are what the engine's own kernels achieve on the selected executor, derived on
every start from the records of their points; the costs themselves are never stored. Each kernel is
formed at its declaration's default specialization at the backend's resident layout, with bf16
activations and H = 4096. Each sized dimension doubles until one call takes at least 100 launch
costs, bounded by the executor's largest allocation. A streaming kernel then rotates through enough
argument sets to span four times the largest cache that can serve them, so its bytes come from
memory.

The measured formats are fixed per backend: every element a load can hold a weight resident as
(each packed encoding's representation in the backend's resident layout, and each dense dtype),
whatever the models assessed. The points are therefore the same for every model and measured once
per machine; a model's speed is arithmetic over them, and no model ever starts a job or changes
what is measured.

- **Weight rate**, one per measured format along each access path of the executor's device
  placement, and the activation dtype's along each path of its upload placement: the bytes per
  second the format's decode projection streams at one row with its weights on that path. It
  prices the kernel and the path together, so no copy rate is measured. Each format's bytes are
  priced at its own rate.
- **Matrix throughput**, one per measured format: the FLOPs per second of the projection at 512
  rows, the prefill chunk loads run.
- **History cost**, one per stored history layout: the decode attention entry at one sequence and
  two depths, giving a fixed cost per call and the history bytes read per second beyond it.
- **Composition ratio**: one reference chain of decode kernels, timed as one sealed graph against
  the sum of its calls alone.
- **Launch cost**: a sealed graph of 256 dependent trivial calls, its device time over 256.
- **Step cost**: the host time per run beyond its device time, for runs queued back to back (decode
  with lookahead) and for one submit and wait (decode without it).

A figure whose points fail, or that the job recorded unobtainable, is unmeasured with its reason.
What it would price is unavailable; no figure is borrowed from another format, layout or path, and
no published table, device-name rule, reported peak or class constant supplies one. The other
figures stand.

## Calculation

- Demand is one record per call of a plain decode step, from the same allocation-free execution
  plan a load prepares: the weights read once per step, a routed call's experts, the bytes each row
  reads and writes in device and upload storage, each row's matmul FLOPs by weight format, the
  state each sequence reads and writes, the history it reads per token of context with the read's
  window and layout, and its native launches at the step's row count. History reads grow with
  depth, up to the window of a window-domain layer; a layer sharing another layer's history reads
  the source's. Recurrent and state-space state is moved once per sequence, never once per row of
  depth.
- At N sequences, one row each, a call's bytes are its weights, its experts (a routed call reads
  the union of the rows' choices, E · (1 − (1 − k/E)^N) slabs), N rows' bytes and N sequences'
  state, at the slowest weight rate of their storage's placement paths; plus N sequences' history
  at depth d, at its layout's history rate, scaled down where device storage's placement includes a
  slower path than the one the rate was measured on. An indexer's block keys, which no attention
  entry reads, are priced at the weight rate.
- A call's arithmetic is N rows' matmul FLOPs over their format's throughput, plus attention's
  FLOPs over the bf16 matrix throughput.
- A call costs the larger of its bytes' time and its arithmetic's time, plus its layout's fixed
  attention cost and its launches times the launch cost.
- The step costs the sum of its calls, scaled by the composition ratio, plus the step cost of the
  model's resolved lookahead policy. The estimate is its reciprocal, in tokens per second per
  conversation.

Every result has finite positive rates, one per requested depth in ascending order. The expected
error is about ±15% typically, the spread of the composition ratio across models.

## Identity and caching

Estimates are cached with their exact assessment under the assessment environment identity, which
includes the engine build, the topology's normalized structure and the derived executor costs.
Measurement timestamps and live usage do not enter identity. Measurements arriving change the
identity, and every assessed model is estimated again under it and published in one revision. A
warm exact-cache hit performs no estimation.

## Conformance

- Estimation reads no tensor payload, opens no device and runs no model decode.
- Increasing planned traffic cannot improve an otherwise identical estimate.
- Estimated samples are strictly ordered, one per requested depth, ending at the served context;
  a fitting model without measurements has `Pending` or `Unavailable` performance, never samples.
- Recurrent state is charged once per token, never multiplied by depth.
- Speculative heads do not change the plain-decode estimate.
- Every figure an estimate reads was measured on this executor. Nothing is fitted on another
  machine, and an unmeasured figure makes the estimate `Unavailable`, never borrowed.
