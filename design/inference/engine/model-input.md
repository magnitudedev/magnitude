---
applies_to:
  - inference/engine/artifacts/**
  - inference/engine/families/**
  - inference/engine/executor/**
  - inference/engine/chat/src/artifacts.rs
  - inference/engine/chat/src/tokenizer.rs
---

# Model input boundary

Artifact preparation owns process-local package identity, payload interpretation,
decoding, and bounded generic media. It may inspect a bounded header bundle
before weight payloads exist; this validates
metadata, tensor geometry, and declared non-overlapping ranges without claiming
that the payloads are present. Executable package admission requires the full
payload ranges. A package component is one GGUF or every shard of a split GGUF
(llama.cpp `gguf-split`): the first shard names the component, the remaining
shards are found beside it by the split naming convention and must declare their
positions, and header and payload opens both present one directory (the first
shard's metadata, every shard's tensors) under one component identity. The header
reader accepts every GGML tensor type a GGUF can store; whether an encoding
executes is decided by planning, where an unsupported encoding is an unsupported
representation, never a malformed artifact. Artifact preparation does not assign model-family coordinates or
position-table interpolation. A family adapter consumes
the model definition, token plan, and generic prepared media to create one closed numerical input
contract. The contract contains the final token coordinates, media spans, patch order, attention
coordinates, attention window ranges, and position-encoding interpolation indices and coefficients
needed by execution. An overview/detail image processor publishes crops and their
source-image slice grids without assigning numerical positions. The MiniCPM-V 4.7
4× adapter places these crops on a shared canvas: the overview spans the full canvas,
detail rows retain their tile offsets, delimiters use the canvas edges, and following
text resumes beyond its largest coordinate. Its SigLIP tower uses learned fractional
position buckets and full attention without rotary; per-patch normalization precedes
2×2 concatenation and a second normalization over each concatenated cell. Intra-tower
16× compression is a different graph and is rejected rather than executed as 4×.
The adapter is configured with tokenizer-derived family marker identities before it prepares a
request; the model definition and media alone cannot identify those tokens.
For text-only input, the same closed contract records the family-supplied coordinates directly;
constructing it performs alignment checks and does not derive coordinate semantics.
Every prompt token is one input row, in order, except each image's placeholder token, which the
adapter expands to its span's rows; the input's layout alone therefore maps a prompt position to its
row.

A family binds every stored tensor of the components it interprets by exact name and shape; a
stored tensor no role binds, or one stored in a shape its role does not read, is an unrecognized
tensor. A family may accept a stored tensor its model provably never reads: it checks the tensor's
exact geometry, and no weight descriptor names it, so planning never imports, places or charges it.
Gemma 4 layers that reuse an earlier layer's history are the case: converters that keep the
checkpoint's key and value projections and key norm for those layers store `attn_k`, `attn_v` and
`attn_k_norm` there, which the layers never compute (llama.cpp loads them as optional for layers
without their own history and never reads them), so such a file interprets to the same definition as
one without them. The header weight-encoding check stays independent of interpretation and still
counts such a tensor's stored type.

Metadata is read strictly only where it can change what the model computes. Every key in the
component's own architecture namespace (`<general.architecture>.*`) must be one its family reads,
and the chat layer reads `tokenizer.*` the same way; any other such key is unrecognized metadata.
A family may likewise accept a key its model provably never uses: a routed Qwen 3.5 header's
`feed_forward_length`, which converters write beside the expert widths (Ollama's as zero), sizes
no sublayer of an all-routed decoder, so it is accepted and ignored whatever it holds.
Every other namespace configures nothing and is ignored whatever it holds: `general.*` describes
the model, and container and uploader namespaces (`split.*`, `quantize.*`, `imatrix.*`,
`mradermacher.*`, `unsloth.*` and the like) record how the file was made. A family may read a named
key outside its namespace, but never refuses one it does not know, and names (a model's or a base
model's) never decide a verdict: a projector binds when its output width
(`clip.vision.projection_dim`) is the target's embedding width, and a header naming no architecture
is unreadable. A projector that does not bind (another width, or any projector given to a text-only
family) is the projector's refusal, never the target's: the target is interpreted without it and
takes text input only. Absent head widths read as llama.cpp reads them, and so does an absent RMS
norm epsilon where llama.cpp takes the layer-norm epsilon in its place (Nemotron-H). The Qwen 3.5 lineage's layer kinds read
as llama.cpp reads them too: `attention.recurrent_layers` flags every stored block, prediction blocks
last and never recurrent, and otherwise one layer in every `full_attention_interval` is attention; a
header carrying both binds only when they agree. The rule is shared by every family,
so a quantizer's provenance stamp never decides a verdict.

An input row's token also names the rows it reads of a gathered table, by the table's row rule. A
token table's row is the token. An n-gram table's rows hash the token with its predecessors: order
`n` mixes the last `n` tokens (exact 64-bit products, XOR-combined) into each of its heads' row
ranges, a predecessor at or before the latest reset token preceding the row, or before the
sequence, reading as the reset token. A media row hashes the token that stands in its place. The
rule is one definition: the device evaluates it from a step's token rows and the token window its
state carries, the host over the tokens it fed, and both name the same rows at every position.

The tokenizer is adapted from the container's tokenizer facts, never from the model family: the
declared scheme selects one implemented profile, and an unimplemented scheme or uninterpreted
tokenizer metadata is an unsupported representation. Implicit sequence-start insertion is such a
fact. A tokenizer that begins every sequence with its BOS applies it exactly once: a prompt whose
rendered text already begins with the BOS (templates that render it) is not given a second one.
End-of-generation and never-generated tokens are likewise tokenizer facts. Declared
additional EOS identities join the stop set after integer and vocabulary-bound
validation; they do not alter ordinary text encoding.

Generic execution validates shapes and bounds against the model definition and consumes the
prepared contract. It does not repeat family-specific spatial derivation or infer semantic meaning
from artifact tensor names. Prepared input and media cross the host/worker boundary as owned,
device-free values; a worker process re-establishes every construction invariant of a decoded
input against its model definition before admitting it. A worker process opens exactly the package
the host admitted (same component files and sizes and tensor directories) and carries the host's package
identity, so both sides name one package. Live device resources remain worker-confined.

A separate block drafter declares block causality independently of history
windows. Its GGUF `dflash.attention.causal` is either BOOL (broadcast to all draft
layers) or ARRAY<BOOL> with exactly one element per layer. No sliding-window or
model-name heuristic supplies omitted semantics. A drafter that omits it is refused
with `MissingMetadata` naming the key, never as built for another target. Interpretation normalizes this
once to per-layer block attention; execution consumes only that normalized model.
History reach is likewise a declared draft fact: a sliding layer reads its
`attention.sliding_window`, and a full layer reads at most `dflash.context_length`
tokens back, the longest sequence the draft was trained on. A full layer is a window
of that length, never a history beyond the draft's training. Injection may skip only
layers narrower than the draft's widest history, which still reads every row they drop.
Draft input embedding scale and final projection softcap are explicit draft facts.
Absent declarations mean raw embeddings and raw logits, independently of target
transforms. A declared positive softcap applies to the vocabulary projection before
any draft-specific additive bias; zero disables it. Nonfinite or invalid scales
are rejected. Converter-inferred values require upstream semantic evidence; target
architecture alone does not establish a drafter transform. An optional `output.weight`
(and its encoding-required scale) declares an owned projection over the full target
vocabulary and hidden width. Absence shares the target projection; equal dimensions
do not establish training compatibility between those projections.

Prediction heads declare their residual form and the conditioning they chain: either
normalized readout features or raw residual streams before the final readout mix.
Every block in a chain preserves that conditioning representation. A residual chain
has the target's width and residual form. Packaging is independent of these semantics:
a head may live beside the target or in a separately pinned draft component while
sharing the target's embedding and vocabulary projection.

The Flash Next shared prediction package (`qwen4exp`) contains one causal head with
its own attention history. It consumes the target's four F32 residual streams before
its exit mix. For each stream it RMS-normalizes that stream with its own coefficients,
concatenates the normalized successor embedding, and applies the shared combining
projection. Its attention and routed feed-forward use the declared hyper-connection
mix/scatter sites. Its exit mix feeds the shared target vocabulary projection; its
unmixed residual streams condition the next draft step. Compression ratio zero means
ordinary causal attention; published dormant indexer weights are validated but never
executed. Header geometry must match the target; the catalog pins the trained pair.

A Gemma 4 assistant (`gemma4-assistant`) is interpreted against its Gemma 4 target as
that target's draft head, stored in the draft component and keeping no history of its
own. Every assistant layer is a prediction layer whose attention reads target history
(`shared_kv_layers` equals `block_count` equals `nextn_predict_layers`): a window layer
the target's second-to-last layer, a full layer its last, as the published checkpoint
was trained. The history domain each layer was trained with (its window, or every row)
must equal the domain the source target layer keeps, and its head count and width and
rotary must equal the owner's; any disagreement rejects the pair rather than reading a
different history. `embedding_length_out` must equal the target's width. The head's
own width comes from `embedding_length`. Its input is the target's token embedding,
scaled as the target scales it, beside the target's normalized output row, projected
into its width; its normalized output reads out through its own full-vocabulary token
table with no soft cap, and its post projection returns it to the target's width as
the next chained step's conditioning row. QAT and non-QAT assistants share shapes, so
shape agreement is necessary but never proof of a matched pair; the catalog pins it.
A Gemma 4 target's last `shared_kv_layers` layers are interpreted as Shared history
domains reading an earlier owning layer of their kind; the definition carries nothing
else about them, and execution derives from those domains alone which blocks need run
only over the rows a step reads (`execution-plan.md`, the stateless tail).

## Acceptance criteria

- Artifact output contains no model-family numerical controls.
- Any valid GGUF header, split or not, parses; a split package plans and loads as one package.
- The family adapter computes every spatial value consumed by the vision lane.
- A prepared numerical input cannot contain mismatched token, span, media, or patch domains.
- A prepared numerical input holds each distinct image once, keyed by content identity: every
  span names a held image and every held image is placed by a span.
- A prepared prompt holds the tokenizer's implicit BOS exactly once, and no BOS otherwise beyond
  what the template renders.
- Execution performs no model-family coordinate or interpolation calculation.
- The device's and the host's n-gram rows equal the reference rule at every position: after
  reset tokens, after a previous context, and across chunk and tape-version boundaries.
