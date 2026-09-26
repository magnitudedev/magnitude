# Speculation

**Draft several tokens, verify them together with the target, and commit only
the accepted continuation.** Each request owns its progress; compatible drafting
and verification work can batch independently.

## The generation round

```text
Committed target state + next input (anchor)
                 │
                 ▼
        Draft up to d proposals ◄── drafter state / required target features
                 │
                 ▼
    Target verifies anchor + proposals
                 │
                 ▼
      Accept matching prefix of length k
                 │
                 ▼
   Reconcile target and drafter state
                 │
                 ▼
   Publish k accepted tokens + target token
                 │
                 └── final token becomes the next anchor
```

The **anchor** is the next legal input unit the target must consume: initially the final
prompt unit, normally the last emitted token thereafter. It is part of the
conversation but is not yet represented in the target's consumed state. Plain causal
feedback may instead retain an already-computed successor across service calls: its
target state includes the final published token. That prediction is method-local and
is not part of a reusable checkpoint; checkpoints describe only consumed history.

Proposal width is bounded by the round's output allowance, remaining context
and available memory. An output allowance of `m` permits at most `m − 1`
proposals, reserving room for the target's final token. The allowance bounds
work and output; it does not predict how long verification will take.

## Acceptance: target-sampled prefix matching

For `d` proposals, the target evaluates `d + 1` inputs: the anchor followed by
the proposals. At each position, sample using the request's target distribution
and constraints. Accept consecutive proposals equal to those samples. At the
first mismatch, emit the target sample instead and discard later proposals.
If all proposals match, emit the additional sample after the last proposal.

```text
Target input:          anchor     a     b     c
Target next-token:        a       b     X     …
Proposal to compare:     a       b     c
                         ✓       ✓     ✗

Accept:                a b          (k = 2)
Publish:               a b X        (k + 1 outputs)
Committed target input: anchor a b  (k + 1 consumed inputs)
Next anchor:           X
```

This mechanism needs proposal tokens, not drafter probabilities. Sampling draws
belong to each request and logical output position, so regrouping batches or
discarding proposals does not consume another request's random choices. Numerical
differences in target execution can still change the sampled result.

History-dependent penalties and constraints are evaluated along the candidate
prefix, then committed only along published output. A stop or output limit cuts
off the continuation at its actual boundary; consumed target position is tracked
separately from emitted length.

## State reconciliation

Verification may advance beyond the accepted prefix. Before publication, that
tentative state must resolve to the request's accepted boundary.

| State | Resolution |
|---|---|
| Attention KV | Keep the accepted input prefix; make rejected suffix positions inaccessible |
| Recurrent state | Select an accepted-boundary snapshot, or replay exactly the accepted inputs from a known state |
| Drafter state | Align with accepted target history and required target features; discard rejected proposal state |
| Conversation and constraints | Commit only the permitted emitted continuation |

Attention can hide a rejected suffix through its visibility boundary. A recurrent
state summarizes the inputs it consumed and cannot generally be repaired by
changing a length counter. Required replay is real model work and participates
in scheduling and batching.

One-input recurrent advances need only their initial and final states. Fully
committed inputs likewise need no interior-prefix repair trace. Reserve and
retain that trace only for wider tentative advances; their known input prefix
is uniform across the physical batch, while final acceptance remains per request.
Initial and destination state images remain charged through device completion.

A drafter may lag the target when its method permits deferred catch-up. That
obligation remains explicit and must be satisfied before the drafter's next use.
A reusable prefix therefore includes target state, drafter state or a valid
reconstruction path, and any outstanding alignment obligation. Target KV alone
does not guarantee that drafting can resume.

MTP conditions each consumed token on the preceding target hidden state, including
across prompt chunks. Its checkpoint retains only committed head history, deferred
committed pairs and the final target feature. The successor token is supplied by
the resumed continuation; no token outside the checkpoint prefix is retained as
head state. Head prefill uses the same cooperative model operations as drafting.

For conditioned inputs, the head consumes the actual successor embedding paired
with the preceding target feature. A placeholder token lookup cannot reconstruct
that pair. Target coordinates and head positions remain owned by their respective
model contracts. A final indivisible conditioning unit is consumed completely before
language proposals begin. Suffix matching treats nonlanguage positions as barriers;
history penalties likewise use eligible language tokens. These operations consume
[semantic input contracts](../models/inputs.md), without interpreting image formats.

## Batching without a fixed cohort

```text
A: draft ─ draft ─ verify ─ accept 2 ─ reconcile ─ publish 3
B: draft ─ draft ─ verify ─ accept 0 ─ reconcile ─ publish 1
   └ shared work ┘ └ shared work ┘    independent state boundaries

C: plain target step ─ publish 1

Next round: regroup whichever draft / target / repair work is ready.
```

Drafting batches by drafter compatibility; verification batches by target
compatibility. Different draft depths and proposal widths can produce different
groups. A request ready to verify does not wait for every other request to finish
drafting. A physical batch imposes neither a shared acceptance length nor shared
membership in the next round.

All tentative state remains owned by its request until commitment or discard.
Finishing one row does not release resources still used by another row or by
unfinished device work. Cancelling a request prevents further work and discards
its tentative continuation without changing peers' accepted state.

## Scheduling and stopping boundaries

A speculative round can suspend between model operations, preserving proposals
and unresolved state, so prefill can receive service. Drafting, target verification
and repair all count as decode execution time, once per physical execution.
No published token is required for the round to have made progress.

Yielding alone does not require a GPU fence. Dependent work can stay on the
device; completion is observed when needed for acceptance, publication or safe
resource reuse. Submitted lookahead remains bounded so a long round cannot
eliminate opportunities to reschedule.

| Boundary | Behavior |
|---|---|
| No usable draft seed, or one output slot left | Advance the target without proposals |
| No proposals accepted | Consume the anchor and emit the target's replacement token |
| All proposals accepted | Retain the full verification state and emit the bonus target token |
| Stop inside the candidate continuation | Publish only the permitted prefix and discard unreachable work |
| Known forced tokens | Advance them directly while maintaining required target/drafter state |

## Performance properties

| Active sessions | Plain generation | Speculative generation |
|---|---|---|
| One | Direct target advancement; no proposal or acceptance work | Preserve the draft dependency chain and one target verification per block; no batching wait |
| Multiple | Persistent compatible target batches | Batch draft and target work independently; accept and reconcile per request |

Speculation is useful when accepted output per round compensates for drafting,
wider target execution and repair. High concurrency can reduce that advantage
because ordinary batching already amortizes target weight access. Increasing
proposal width trades potential accepted output against wasted verification,
temporary state and longer service intervals; maximum width is not inherently
optimal.

## Component models

The [engine component records](components.md) own the contracts, dimensions,
parameter bindings and independent controls. The [service derivations](../performance/derivations/service.md)
supply reusable mathematics; [performance](../performance.md) defines evaluation
and evidence. This document owns the behavior guarantees those models preserve.
