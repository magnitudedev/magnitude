# Model executor

**A model advances numerical state for the sequences it owns. Above it, work is
proposed and accepted per request; below it, execution is packed. The contract
exposes work, positions and resource facts, never state layout.**

## Lifecycle

```text
input ──open──► sequence @ position 0
                   │
prepare(requests) ─┴──► one packed execution: commands + logits rows + one advance per request
                            │
                       submit ──► ticket
                            │
       advance.commit() ◄── completion proven      position moves; inputs move to after(position)
       advance.close()  ◄── otherwise               nothing moved; resources released
```

| Rule | Reason |
|---|---|
| Physical execution is packed; acceptance is per request | One forward serves several sequences; one failing row does not roll back a peer |
| Position moves only on proven completion | A committed position is a promise that the state behind it exists |
| An advance not committed is aborted on close | Preparation and submission failures leave the sequence exactly where it was |
| Logits are requested per row: none, last, or all | A prefill chunk that produces no output pays for no readout |
| A checkpoint needs reconciled state | Forking mid-advance would fork a promise not yet kept |
| The executor never interprets an input marker | Media and conditioning are prepared outside; the decoder consumes coordinates and feature slices |

## Inputs

An input layout is a token count plus ordered, disjoint spans of conditioned input.
Each span has an identity, a boundary rule and whether it counts as language
history. Unmarked positions are ordinary text.

```text
prompt:   [ text ][ image: indivisible ][ text ]
chunking: a soft allowance ends before an indivisible span, or consumes the whole span
          when it is the next unit; it never splits one
features: projected conditioning for a span is owned separately and viewed by the
          sequence only while the span is still ahead of its position
```

The executor asks the layout for legal boundaries; it does not know why a span is
indivisible. Conditioning features are the model's operands, not the scheduler's
concern: adding a modality adds spans and features, not service policy.

## Reclamation

The executor answers two questions for the service, and only these:

| Question | Answer |
|---|---|
| What would closing these reconciled sequences free? | Bytes released exclusively by that set: their extents, their state banks, and caches overlapping them |
| Reclaim now | Unborrowed scratch and idle caches are retired; live work and live sequences are untouched |

Pricing is by exclusive ownership, so a sequence sharing a prefix with a
checkpoint is priced at what it alone holds. The service decides victims; the
executor never does.
