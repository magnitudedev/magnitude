# Generation

**A request owns its logical progress; execution batches own shared computation;
numerical state is borrowed and always reconstructible from the logical record.**

## From proposal to acceptance

```text
request ──ready(allowance)──► proposal: kind, tokens, logits wanted     reserves nothing
                                   │
batch.prepare(proposals) ──► one packed model execution + one selection over its logits
                                   │
                              submit ──► ticket
                                   │
batch.finish() ──► per request: read its selected token, commit its advance, accept
                                   │
                              output queue ◄── take() by the transport
```

| Work | When | Logits |
|---|---|---|
| Prefill | Prompt remains; the next chunk within the allowance and the layout's boundaries | Only on the final chunk |
| Decode | The last sampled token is the pending input | Yes |
| Replay | Accepted history is being re-fed after eviction | No |

A proposal is a pure statement of the next legal step. Nothing is claimed until a
batch is prepared, and a batch is prepared as a whole or unwound as a whole.

## Rules

| Rule | Reason |
|---|---|
| Acceptance is per request | A row that fails to select leaves its peers' progress intact |
| A sample depends on the request's seed and position, never on the batch | The same request produces the same tokens whoever it shares a batch with |
| Output credit is bounded | A transport that does not drain stops the request from generating, not the batch from finishing |
| A stop, the length limit or the context limit is decided at acceptance | The finish is part of the logical record, so it survives eviction |
| Cancellation drops unsubmitted work only | Submitted shared work completes; its resources are released when the ticket is done |
| Failure retains accepted output | What was published stays published; only future work is refused |

## Eviction and reconstruction

Numerical state is borrowed. The logical record (prompt, sampled tokens,
undelivered output, published cursor, finish) is complete at all times, so state
can be discarded under pressure and rebuilt without loss:

```text
processed 900 of prompt 1000, 12 sampled       evict: recovery position = 912
restore with fresh state @ 0
replay  [0..512)   no logits
replay  [512..912) no logits                    caught up
decode  → 13th token                            as if nothing happened
```

Replay costs the prompt again; it costs no output and no sampling. The transport
sees a preempted request, not a restarted one. A checkpoint is the opposite trade:
it keeps state by claiming it, so a fork costs claims rather than replay.

## Sampling

Selection is position-addressed and independent of model and batch: greedy or
categorical, keyed by the request's seed, the sample position and the draw
domain. Distribution shaping (temperature, penalties, truncation, constraints)
belongs to whoever constructs the distribution; selection only selects, and a
distribution that cannot be selected from fails that row explicitly.
