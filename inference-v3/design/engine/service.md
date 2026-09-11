# Service

**One request owner, one physical batch in flight. Prefill and decode time-share;
capacity is negotiated with the model's own prices, never with its layout.**

## Lifecycle

```text
                          admit
Queued ── selected ──► Runnable ──► Completion (batch in flight) ──► Runnable …
                          │                                              │
                          ├── output queue full ──► Output (until drained)│
                          ├── evicted ──► Preempted ──► replay ──────────┘
                          ├── no capacity, peers may change it ──► Capacity (until the epoch moves)
                          └── stop / length / context / cancel / failure ──► Terminal
```

Status is derived from facts, never stored; a request is in exactly one of these
states at any time. The transport drives `step`, waits on the returned ticket, and
drains output. Idle service does nothing until something changes.

## Selecting a phase

Under contention (both prefill and decode candidates exist) decode runs first,
then prefill accrues **decode debt** at the configured share `s`:

```text
prefill lasting p:   debt += p × s / (1 − s)
decode lasting d:    debt −= d
prefill eligible:    debt ≤ 0

equal shares, 40 ms prefill, 10 ms decode rounds:
[D:10][P:40][D:10][D:10][D:10][D:10][P:40] …     debt 40 → 30 → 20 → 10 → 0
```

Without contention the eligible phase runs without throttling, and contention
accounting resets when contention ends: idle time earns no credit against future
arrivals. Within a phase, priority is waiting time plus a locality credit for
requests that are active, resident, or carrying preemption debt, then least
service, so a resident request is not pushed out by a newcomer of equal age.

## Capacity

Preparation claims state and scratch; when it cannot, the service negotiates in a
fixed order, each step cheaper than the next in lost work:

| Step | What it costs |
|---|---|
| Reclaim: retire unborrowed scratch and idle caches | Nothing already paid for |
| Drop the last selected request from the batch | That request waits a step |
| Halve the prefill allowance | Smaller chunks repeat execution overhead |
| Evict victims priced by the model | Their prompts are replayed |
| Block until the service epoch changes | Wait for a peer to finish, publish or cancel |
| Fail with the required and available bytes | Nothing else can change the answer |

A physical shape that already failed is never retried; a soft allowance cannot
split an indivisible input span, so when shrinking would only repeat the shape,
eviction is tried instead. Victims are chosen output-blocked first, then least
preemption debt, then most exclusive bytes per replayed input, then least
service; a victim carries debt that protects it from being chosen again until it
has caught up. Eviction counts only if the budget actually fell.

## Epochs

Every event that could change what is feasible advances the epoch: an admission,
a completion, a publication, a cancellation, a retirement. A request blocked on
capacity is retried only after the epoch moves, so a full engine does not spin.

## Accounting

A batch's elapsed time is one cost, attributed equally to its requests by the
phase they were in. Emitted tokens are not a clock. Preparation, submission and
execution failures are attributed to the requests in the batch and keep their
accepted output; a failed execution owner fails every live request at once.

## Consequences

Decode-first contention keeps generating requests responsive at the cost of a
newcomer's first token. FIFO admission can leave queued requests behind long
residents. Time sharing controls this engine's own contention; it neither promises
a per-request latency nor governs other users of the device.
