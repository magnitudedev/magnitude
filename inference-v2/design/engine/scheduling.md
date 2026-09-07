# Scheduling

**Continuous batching with bounded prefill chunks and time sharing between
prefill and decode.** Scheduling determines eligibility and service allowances;
[batching](batching.md) determines how selected work executes together.

## Request lifecycle

```text
                       admission
Waiting ── capacity available ──► Prefill ── prompt ready ──► Decode
                                   ▲                         ▲  │
                                   └── next chunk            └──┘ next round
                                                              │
                                                        stop / cancel
                                                              ▼
                                               retain reusable prefix;
                                               release remaining state
```

Admission is FIFO and limited by active-request capacity and memory. Memory
must cover state growth and execution workspace, not only existing KV. Retained
prefixes reduce remaining prompt work; reclaimable cached state can make room
for active requests.

A request whose output consumer is blocked receives no new generation work.
Cancellation removes future work immediately; already submitted work finishes
before its resources can be reused.

## Units of service

| Unit | Work granted | Completion |
|---|---|---|
| Prefill service | One aggregate token allowance across compatible pending prompts | Selected chunks have advanced prompt state |
| Decode round | One bounded generation step for every ready request | Each participating request has a publishable result or a retained continuation |

A plain service publishes a bounded number of tokens and may retain one successor
prediction across services. A speculative step is a
[draft–verify–accept round](speculation.md), potentially producing several.
A round can pause at a model-operation boundary and resume later; it need not
hold the device until every participant has published output.

### Dividing prompt work

Start with the oldest admitted unfinished prompt and select compatible peers
in FIFO order. Share the token allowance across these prompts. Equal-width
chunks can execute together; shorter tails form separate groups. An unbatchable
prompt, or a lone prompt, receives the whole allowance.

```text
Aggregate allowance: 1,024 prompt tokens

Only A waiting:      A [──────────── 1,024 ────────────]
Two compatible:      A [── 512 ──]   B [── 512 ──]
                         └──── one physical batch ────┘

The allowance is a total across requests, not 1,024 tokens per request.
```

The allowance also respects available memory. Prefill may use larger chunks
when there are no ready generations to interrupt.

## Selecting the next phase

During contention, prefill incurs **decode debt**: enough decode service must
follow to maintain the chosen share of execution time.

For a desired decode share `s`, where `0 < s < 1`:

```text
Prefill lasting p:      add p × s / (1 − s) to decode debt
Decode lasting d:       subtract d from decode debt
Next prefill:           eligible once the debt is repaid
```

Enter contention with a decode round, then allow prefill. Subsequent phase
selection follows the debt. Repayment overshoot carries forward during
contention, capped at one decode round, so indivisible rounds do not continually
bias the time share toward decode.

```text
Equal shares; 40 ms prefill and 10 ms decode rounds

Time → [D:10][P:40][D:10][D:10][D:10][D:10][P:40] …
             debt:40 → 30 → 20 → 10 → 0
```

If only one phase is eligible, run it without time-share throttling. Reset
contention accounting when contention ends; idle or uncontended execution does
not build credit against future arrivals.

## Interruption bounds and accounting

| Control | Purpose | Trade-off |
|---|---|---|
| Aggregate prompt-token bound | Bound the amount of prompt work selected at once | Smaller chunks repeat execution overhead |
| Decode time share | Limit sustained interference from prefill | More decode protection increases prompt latency |
| Optional prompt-duration target | Limit consecutive prefill interruption | Shorter interruptions can reduce prompt throughput |

Consecutive prefill services share the optional duration budget. Exhausting it
forces a decode round even if time-share accounting would permit more prefill.
Without a duration target, token bounds and measured time sharing still apply.

Charge service elapsed time once, including drafting, verification
and repair in decode service. A batch serving four requests is one service cost,
not four. Emitted token count is not a clock: a speculative round can do useful
work before publishing anything.

These are soft timing bounds. An indivisible operation can overrun a target,
and bounded device lookahead must leave opportunities to reschedule. Yielding
does not itself require a GPU synchronization. Plain service can leave one successor
in flight; subsequent service observes its completion. Prompt-phase transitions and
shared-layout changes complete outstanding consumers before proceeding.

## Consequences

Sharing prompt work brings peers into decode sooner but can delay the oldest
prompt's first token. FIFO admission can leave queued requests behind long-lived
active requests. Time sharing controls contention within this engine; it neither
guarantees a fixed per-request latency nor controls other GPU processes.

## Component models

The [engine component records](components.md) own the contracts, dimensions,
parameter bindings and independent controls. The [service derivations](../performance/derivations/service.md)
supply reusable mathematics; [performance](../performance.md) defines evaluation
and evidence. This document owns the behavior guarantees those models preserve.
