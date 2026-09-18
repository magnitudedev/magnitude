# Model inputs

**An input combines token positions with any conditioning needed to interpret them.**
Scheduling consumes legal advancement boundaries without interpreting a modality.

## Preparation boundaries

| Stage | Responsibility |
| --- | --- |
| Host source preparation | Validate bounded media, decode, orient, normalize, and apply artifact-defined processor semantics |
| Model input adapter | Resolve placeholders, expanded token layout, coordinates, and conditioned spans |
| Seismic encoder/projector | Execute numerical feature computation using ordinary programs and resource contracts |
| Decoder input | Bind tokens, positions, feature slices, and visibility for a legal advance |

Processor and preparation identities include every choice that changes numerical
input. Host preprocessing and neural encoding have different owners. Unsupported
media or artifact combinations fail explicitly rather than silently changing input.
Text-only use keeps modality resources lazy.

## Semantic spans

| Property | Meaning |
| --- | --- |
| Extent | Logical decoder positions occupied by the conditioned input |
| Identity | Content and preparation semantics required for compatible reuse |
| Features | Retained resource slices supplying conditioning |
| Boundary rule | Whether a position inside the span is a valid continuation point |
| History contribution | Which positions participate in language history |

- Ordinary text permits token-boundary advancement.
- An indivisible span may be tiled physically but cannot be accepted partially.
- A soft service allowance may expand to finish the first indivisible unit; physical
  capacity limits still apply.
- Physical pages do not define semantic input boundaries.

## Execution and lifetime

```text
prepared input → encoder work → completed feature leases
    → legal decoder advances → remaining input continuation
```

- Encoder work shares the execution owner, service budget, and completion handling.
- Encoder batching is independent of decoder batching. Completion returns control
  to service selection before more work is scheduled.
- Requests, checkpoints, and in-flight work retain their own feature leases.
  A cache is an optional additional owner.
- Cancellation does not publish unfinished conditioning or create accepted decoder state.
- Checkpoints retain the unconsumed input semantics required to resume.
- Replay supplies the same conditioned operands; placeholder token IDs alone are insufficient.
- Fully consumed conditioning need not be encoded again for ordinary text continuation.

[Models](models.md) assemble these inputs; [scheduling](scheduling.md) sees their legal
work boundaries; [state](state.md) retains continuation ownership.
