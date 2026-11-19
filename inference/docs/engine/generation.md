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
| Front pass | Pending token (exact row), then a front row on each guess of the sequence's chain | Next selection from the exact row only; the front rows' selections feed the next chain |

Forced runs exclude EOS and respect output, context, credit, service, and quantum
bounds. They occupy ordinary output positions and remain decode work for fairness.

Decode, verification and front passes are one shape, `Rows`: one exact row,
the pending token, which always commits, then rows ahead in position order,
each a guess of the token after the row before it. The rows ahead are either
acceptable rows (a verification's proposals), which may extend the commit,
or front rows, which only run ahead and are never accepted. No other pass has
a representation: no pass without an exact row, and no front row beside
acceptable rows.

| Pass | Rows |
| --- | --- |
| Plain decode | `Exact { acceptable: [] }` |
| Verification of `k` proposals | `Exact { acceptable: k proposals }` |
| Front pass of depth `k` | `Front(rows)`: `k` front rows, guessed then idle |

A pass's acceptance is one rule: its acceptable rows commit as far as
acceptance takes them (a stop token is never accepted as a proposal), and the
selection of the last committed row is emitted after them. Every pass also
yields its verdict (`AheadVerdict`): its guesses in row order and the truths
the round selected past its exact row. An acceptable row's guess is checked
against the selection of the row before it; a front row of depth `d` against
the truth at its position, which the round learns only for `d = 1` and a
round `d - 1` passes later learns for the rest (the expert ledger joins them
by position). A guess stands where it is the selection, a stop token
included. The verdict travels with the decision the executor reconciles by
(`RoundAcceptance::decision`, the one constructor of a round's decision,
generation's and the evaluation harness's alike), and the executor hands it to
an expert bank, where the load has one, before the round's state is
published, so reads ahead follow the guesses that stood.

A front row never waits: at each streamed layer it computes with only the
experts its step holds and the host reads the rest ahead. Front row `d` runs
on the `d`-th guess of the sequence's chain and is exact `d` passes later
where every guess up to it stands, so it names that pass's reads. A pass's
outcome holds no front row (front rows' selections are apart from the rows),
so nothing a front row computed is output or accepted state, and its exact
row computes as any decode row does in a launch of that many rows. Every pass
emits, and only from its exact row; a wrong guess costs only the reads its
front rows named, never a pass.

The chain is the device's: the MTP head runs right behind each decode pass
(`guess_chain`) and makes the next pass's chain depth by depth, the pass's
front row's selection at each depth whose tokens are the chain's (a chained
guess), then the head's selection one past them (a head guess). The pass
queued behind the running one reads its tokens from that chain on the device,
and the host reads it from the outcome for a pass it builds, so nothing on
the host computes a guess. A pass queued behind its predecessor runs exactly
as deep as it was queued, idle front rows on the pending token past the
chain; a pass the host builds runs as deep as the chain holds, at most its
grant.

The depth and form are the cost model's (`Regime::pass`), taken afresh for
every pass the host builds: the front or verify pass of the depth whose time
a token emitted is least, at the load's live bank and its own reads and
passes. A launch has guessed rows only where every request it selected
decodes in front passes, so no drafting request's verify rows are held for
them, and it reserves every request, its publication room among them, before
it is priced, so a request turned back is neither priced nor granted guessed
rows. It fronts as many of its requests as its rows (`VERIFY_ROWS`) hold
beside every exact row at that depth, the rest decoding one row. The chain is only a prediction checked against the exact rows'
tokens, so eviction, replay or an interrupted round never makes it unsafe.

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
