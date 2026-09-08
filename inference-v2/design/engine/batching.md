# Batching

**Request progress is independent; physical execution is shared.** Batches form
from ready, compatible model work and change as requests advance. A request does
not belong to a permanent cohort.

## From eligible work to execution

```text
Scheduling                 Ready model work                 Execution

                           A: target, 1 input ──┐
Choose requests ──────────► B: target, 1 input ──┴──────────► target batch
and allowances             C: drafter, 1 input ────────────► draft execution
                           D: target, 5 inputs ────────────► verification
                                                                  │
                           Update each request ◄───────────────────┘
                           and regroup its next ready work
```

[Scheduling](scheduling.md) selects service. Within that service, each request
exposes work whose dependencies are satisfied. Compatible operations run together;
results advance their respective requests and make subsequent work ready.
No extra delay is introduced to await hypothetical arrivals or fill a batch.

Prefill and decode use separate, interleaved executions. Their fusion into a
single execution is outside this design.

## Compatibility

| Property | Grouping rule |
|---|---|
| Loaded model | Share the same executable model and weights; a matching model name alone is insufficient |
| Query width | Group equal numbers of input tokens; different widths form separate groups |
| State and conditioning | Must support the same physical execution, including attention/recurrent state and required conditioning |
| Causal boundary | Must support the same treatment of inputs already known to be committed versus tentative inputs |
| Requested outputs | Combine required logits and features where supported; differing output needs alone do not split execution |

Context lengths may differ if the execution supports independent positions and
validity. Sampling settings, constraints, stopping conditions and eventual
acceptance lengths remain per request. They do not impose a common decision
on the batch.

This separates **query width** from **history length**. Two requests with one
new token each can batch despite different histories; a one-token decode and
a five-input verification use different execution groups.

Compatible batching must preserve each request's declared numerical operation as peers
join, leave or change order. A longer peer or padded capacity cannot redefine that
request's reduction. The [kernel implementation](../kernels.md) owns physical tiling
and weight reuse; batching supplies logical rows, positions and validity. Different
query-width algorithms remain subject to their explicit numerical/state contracts.

## State across steps

Persistent batches reuse useful physical state instead of separating and
reconstructing complete KV histories on every token.

```text
Step 1:      [ A | B | C ]     shared execution, three independent histories
Step 2:      [ A | B | C ]     reuse state; append new positions
                    B finishes; D becomes ready
Step 3:      [ A | C | D ]     change membership; preserve A and C's histories
```

Membership changes may require state movement. Ordinary stable-batch advancement
should require new-token writes and necessary growth, not copying all old tokens.
The physical representation determines how that is achieved:

| State | How independence is represented |
|---|---|
| Dense KV | Each row has its own logical position and valid history; unused capacity is masked |
| Paged / slab-backed KV | Each request has its own visible page/run view; storage placement is separate from batch membership |
| Recurrent state | Each request retains its own recurrent state, even when rows share a physical allocation |

A request leaving the batch cannot invalidate peers' state. Before its physical addresses
are freed, outstanding consumers of the shared arena complete; their logical progress
remains independent. Physical allocations remain charged while any request or
unfinished device work still depends on them.
State growth, batch formation and tentative verification state must fit memory
before execution. If preparation exhausts memory while earlier committed work
still retains resources, complete that work for the affected requests and retry
the same prepared execution once. Retry requires preparation to have rolled back
without submitting model work, and completion to have retired pending work. Capacity
growth may require completing consumers from peers sharing the same arena.
Ordinary advancement adds no completion barrier; execution failures are terminal.
If the group still cannot fit, split its prepared work into smaller groups without
repeating sampling or proposal construction. Charge failed preparation and recovery
to the service that incurred them.

## Divergent progress

One physical execution may advance requests by different amounts. In
[speculation](speculation.md), verification produces a tentative state for every
row; each row commits only its own accepted prefix.

```text
One verification batch:     A [anchor a b c]     B [anchor x y z]
Accepted proposals:                 2                   0
Committed target inputs:    A [anchor a b]       B [anchor]
State reconciliation:       at A's boundary      at B's boundary
```

There is no truncation to the batch's minimum acceptance. A request needing repair
does not erase a peer's completed progress. Subsequent draft, verify and repair
work regroups by readiness and compatibility rather than previous membership.

Completion and cancellation also apply per request. Unsubmitted cancelled work
is removed; submitted shared work retains its resources until safe completion.

## Performance properties

| Workload | Required behavior |
|---|---|
| One ready request | Execute directly, without assembling and dismantling a batch |
| Stable compatible requests | Reuse batched state; orchestration follows active work rather than total history length |
| Membership changes | Pay necessary regrouping costs at the change, not repeatedly on every subsequent token |
| Mixed query widths | Separate execution groups; no extra causal tokens added merely to equalize shapes |

Batching amortizes model execution and weight access across requests. Larger
batches can improve aggregate throughput while increasing per-request latency,
workspace and state traffic. Compatibility and memory determine useful batch
size; batching does not override scheduling's service balance.

## Component models

The [engine component records](components.md) own the contracts, dimensions,
parameter bindings and independent controls. The [service derivations](../performance/derivations/service.md)
supply reusable mathematics; [performance](../performance.md) defines evaluation
and evidence. This document owns the behavior guarantees those models preserve.
