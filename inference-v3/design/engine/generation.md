# Generation

**A request owns its logical progress; execution batches own shared computation;
numerical state is borrowed and always reconstructible from the logical record.**

## From proposal to acceptance

```text
request ──ready(allowance)──► proposal: kind, tokens, logits wanted     reserves nothing
                                   │
batch.prepare(proposals) ──► one packed model execution + one selection over its logits
                                   │
                              submit ──► completion
                                   │
batch.finish() ──► per request: validate proposed output, commit its advance, accept
                                   │
                              output queue ◄── take() by the transport
```

| Work | When | Logits |
|---|---|---|
| Prefill | Prompt remains; the next chunk within the allowance and the layout's boundaries | Final chunk, unless the first output is forced |
| Decode | The last accepted token is the pending input | Unless the next bounded output run is forced |
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
| Cancellation drops unsubmitted work only | Submitted shared work completes; its resources are released when completion is proven |
| Failure retains accepted output | What was published stays published; only future work is refused |

A constrained request owns its symbolic matcher independently of numerical state.
A proposed token sequence is validated on a private candidate before model progress
is committed; only acceptance installs that candidate. Discarded proposals, capacity
retries and numerical replay do not advance the matcher. Checkpoint forks have
independent matcher state. Grammar exhaustion permits EOS; accepting EOS is the
logical terminal transition, and no accepted sequence continues beyond it.

For a forced run, advance the pending token followed by all but the last forced
token; accept the whole run only after completion. The last forced token becomes
the new pending input. Final prompt prefill may accept one forced first output.
Runs exclude EOS and are bounded by remaining output, context space, publication
credit, service allowance and configured quantum. They remain decode work for
fairness accounting. State-only rows request neither output projection nor
selection; a mixed execution reports whatever shared work it actually performs.
Forced outputs occupy ordinary logical positions, preserving later random draw
addresses. Their count survives recovery and forks separately from total output.
Bounded run-length counts and committed state-only input work survive alongside
that history. They distinguish issued numerical work from proposals, retries and
replay; sampled-token counts remain separate from total generated usage.

## Eviction and reconstruction

Numerical state is borrowed. The logical record (prompt, generated tokens,
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

Constrained selection consumes packed vocabulary bits and a per-row mask mapping.
Mask construction is independent of the numerical forward. The model may submit
its selected logits first, construct masks on the owner thread, then perform
selection through Ops. The resulting advance carries one completion covering
the forward, transfer and selection; failures accept no logical output or state.
Unconstrained rows share the batch without a vocabulary mask. Allowed logits retain
their values; prohibited logits have no selectable mass. NaN or positive infinity
in the original distribution remains a row failure even when its token is masked.
Masks never require copying logits to the host. The token/status result is checked
against staged symbolic state before numerical progress and publication are accepted.
