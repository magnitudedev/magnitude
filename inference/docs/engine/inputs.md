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

## Attention visibility

An input's rows attend causally, or, for a readout plan's sequence, in attention
blocks: a row of a block reads every row of the blocks its block reads. Blocks change
only global attention, the layers whose history holds every token; recurrent,
convolution and sliding-window layers stay causal over the whole sequence.

- A boundary is a position no earlier row reads past. Advances, prefix-cache entries
  and resumed states begin and end only at boundaries, and the rows between two of
  them (a unit) share one advance.
- A row reads its block's rows its launch computes fresh and every other row it reads
  as stored history: accepted rows, or rows of its advance published before it
  attends. Over dense history the fresh rows extend to the whole run of positions
  around the row, read alike.
- A launch whose rows read each other apart (around rows they do not read) takes a
  class whose attention publishes every row before reading: a prefill class, or over
  compressed history any class of more than one row.
- A unit, or an indivisible media span, wider than the largest launch is one advance
  alone in its launch, computed layer-major in chunks of the largest launch's rows,
  each in the class of its own rows (the last no wider than a launch of its rows alone),
  but no narrower than the load's least layer-major class: the least row class from
  which every class up to the largest fires, in every graph and under every tuning
  choice, the launches the largest fires (certified with the graphs), so every chunk
  rounds its rows alike and a block reading every chunk reads one computation.
  Between global attention blocks every chunk runs the blocks before the next one,
  recurrent and convolution state passing chunk to chunk through an interim bank
  alternating with the successor;
  every chunk then publishes that block's keys and values (a graph class of the block
  computing only its key and value projections, norms and rotary, appended at the
  rows' destinations) and holds its hidden rows, so the next stretch reads
  the whole unit there. A chunk's rows read the rest of the unit as stored history.
  Its held rows are claimed at reservation; it reads only readout values (logits,
  head, scores, retained rows). Its media are encoded once, and each chunk overlays
  its own rows' part of their features. Only a model whose attention was trained
  non-causal (`<arch>.attention.causal` false) computes it, and only that model's load
  prepares the key and value kernel and graph classes; every other model, or one
  whose blocks read per-layer or n-gram inputs, select history or read media rows
  bidirectionally, refuses it at admission as an invalid request naming why.
- Only such a model reads blocks at all: every other refuses a blocked input at
  admission, so no load prepares for blocks it cannot compute in every width.
- A prefix-cache entry carries the visibility of its rows and serves only a path whose
  rows before it read alike.

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
