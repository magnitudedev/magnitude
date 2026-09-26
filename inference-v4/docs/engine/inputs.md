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

Prepared host tensors use immutable binary payloads with explicit names, dtypes,
and shapes. Multibyte values are little-endian; supported host payload dtypes are
float32, int32, int64, and uint8. Geometry must exactly match payload size. Tensor
rank, tensor count, and aggregate prepared bytes are bounded before publication.
Duplicate tensor names fail. Preparation identity includes the processor identity
and every tensor's ordered name, dtype, shape, and bytes. Cloning shares immutable
payloads rather than creating mutable aliases. Host dtype support does not imply
that every model encoder accepts every dtype.

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

For Qwen images, prepared patch tensors must match the bound processor identity,
patch width, finite-value domain, and exact merge-aligned single-frame grids.
Every image owns one matching marked prompt span and its patch slice. Image spans
are causal and excluded from language history. Rotary coordinates advance through
the merged spatial grid; later text continues from its maximum axis extent rather
than from the physical token count. Conditioning identity includes processor, grid,
and patch bytes. Encoder spatial controls preserve merge-group patch order and
align-corners interpolation of the learned position table.

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
