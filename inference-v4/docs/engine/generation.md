# Generation

**Each request owns its logical progress; packed executions share computation.**
Numerical state can be evicted and reconstructed without losing accepted history.

## Request record

| State | Meaning |
| --- | --- |
| Original input | Prompt and retained conditioning required for replay |
| Accepted output | Generated tokens and their logical positions |
| Pending input | Accepted token not yet consumed by the next numerical advance |
| Grammar progress | Request-local accepted matcher state |
| Publication state | Bounded output queue and consumed cursor |
| Terminal outcome | Stop, length/context limit, cancellation, or failure |

## Proposal and acceptance

```text
ready(allowance) → pure proposal
    → batch preparation → submission → completion
    → validate selected output on private grammar state
    → commit model advance → install grammar transition
    → accept tokens and queue output
```

- Proposals reserve no resources and remain retryable before preparation.
- Preparation handles the batch as a whole; acceptance is per request.
- Invalid selection or grammar transition cannot commit numerical progress.
- Failed model commitment leaves grammar progress unchanged.
- Stop, output-length, and context decisions occur at acceptance and survive eviction.
- Output credit bounds generation independently of how quickly a transport drains peers.

## Work kinds

| Work | Numerical input | Readout |
| --- | --- | --- |
| Prefill | Next legal prompt chunk | Needed on the final chunk unless output is forced |
| Decode | Pending accepted token | Next selection, unless a bounded run is forced |
| Replay | Previously accepted input up to the recovery position | None |
| Forced run | Pending token followed by all but the final forced token | None; final forced token becomes pending input |

Forced runs exclude EOS and respect output, context, credit, service, and quantum
bounds. They occupy ordinary output positions and remain decode work for fairness.

## Selection and constraints

- Sampling addresses depend on request seed, position, and draw domain, not batch shape.
- Greedy and categorical selection preserve their declared distribution semantics.
  Unsupported distribution transformations fail rather than being ignored.
- Grammar plans bind to the exact tokenizer, artifact, and initial prefix.
- Matchers are request-local; speculative transitions and checkpoint forks are independent.
- Host mask construction may overlap numerical forward work. Logits remain on device;
  forward, mask transfer, and selection share a completion obligation.
- Row-local invalid distributions produce explicit failures.

Greedy selection breaks ties by the smallest vocabulary index. Categorical
selection uses V3's Philox4x32-10/Gumbel scoring, which samples the softmax
distribution. Masked entries and negative infinity have zero mass. Any source
NaN or positive infinity fails the row, including masked entries; no admitted
finite entry is a distinct empty-distribution failure. Random draws are
counter-addressed by vocabulary ID, request seed, accepted output position,
and selection domain, so retries, replay, vocabulary partitioning, and changes
in batch membership do not consume or shift randomness.

## Recovery and publication

- Eviction retains prompt, accepted tokens, output queue/cursor, terminal state,
  and grammar progress while releasing numerical state.
- Replay neither samples nor emits output nor advances the matcher.
- Checkpoints require reconciled numerical state and retain input continuation.
- Cancellation drops unsubmitted work; shared submitted work completes safely.
- Already accepted output survives failure; subsequent publication or discard follows
  the caller's ownership contract.

[Scheduling](scheduling.md) decides when work runs; [chat](chat.md) interprets published
text; [state](state.md) owns numerical-history transitions.
