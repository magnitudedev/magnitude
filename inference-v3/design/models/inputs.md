# Model inputs

**A model input binds prepared media and tokens into semantic spans and tensor
functions; scheduling consumes only legal advancement boundaries and never
learns a modality.**

## Boundary

```text
Serving and model input adapter                 Ops
───────────────────────────────                 ───────────
bounded source media
  └── decode and model-specific preparation
       └── prepared tensors ──────────────────► encoder/projector tensor function
                                                └── conditioned features
expanded token layout + coordinates ◄─────────────────────┘
  └── semantic spans ─────────────────────────► decoder tensor function
```

Magnitude owns source validation, decoding, preprocessing, placeholder
interpretation, expanded token layout, coordinates and model-specific modality
policy. Neural encoders and projectors are ordinary Ops tensor functions
with the same compilation, resource and completion contracts as language-model
computation.

The public boundary carries typed tensors, span metadata and generic resource
handles. Raw media, processor objects and modality names never enter
Ops. Tensor graphs, kernels and device objects never enter input
preparation or scheduling.

## Semantic spans

A span identifies positions whose token IDs do not fully describe their
computation. It declares content and preparation identity, decoder positions,
conditioned feature slices, visibility and whether an interior boundary is a
complete continuation point.

| Rule | Consequence |
|---|---|
| Ordinary text has no exceptional span | It may advance at every token boundary |
| An indivisible span may be tiled numerically but not published partially | Device tiling cannot redefine continuation semantics |
| A divisible span retains the source and feature slice needed to resume | Chunking does not repeat or lose conditioning |
| Span identity covers every preparation choice that changes computation | Prefix and feature reuse cannot cross incompatible processors or artifacts |
| Physical pages do not define semantic boundaries | State placement remains independent of modality |

The model executor turns spans into row-local tensors, positions, masks and
resource views. The service sees only the next legal amount of work and its
capacity price. Generation, sampling and logical state do not recognize image,
audio or other placeholder tokens.

## Stateless prerequisites

Media encoders and projectors have no autoregressive state. They form ready model
work under the same execution owner and device budget, but their batching is
independent of decoder batching. Differently shaped inputs may encode separately
and later join one compatible decoder invocation.

Completed features have their own content identity, resource lease and
completion. They may be retained independently of decoder history. A cache owns
only an optional lease: eviction cannot release a feature still referenced by a
request, checkpoint or submitted decoder invocation.

## Continuation and replay

A model checkpoint combines accepted decoder state with the unfinished input
semantics required after the same boundary. It retains coordinates, span
identity and any feature slices needed to resume. Replay supplies the same
conditioned operands as the original computation; it never reconstructs them
from placeholder token IDs.

Once a conditioned span has been fully consumed, ordinary generated continuation
needs neither its raw media nor repeated encoder execution unless the model's
declared semantics say otherwise.

## Containment

Adding a modality may add:

- a bounded host preparation adapter;
- model-specific span construction;
- encoder and projector tensor functions;
- semantic tensor operations only when the modality introduces new mathematics.

It does not add a modality branch to service, generation, state ownership,
Ops compilation or TileLang. Unsupported media or conditional model
families fail during binding or preparation, before admission to numerical
execution.
