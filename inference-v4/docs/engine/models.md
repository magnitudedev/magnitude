# Models and weights

**A model executor connects artifact-defined architecture and logical sequence
advances to model-owned Seismic programs.**

## Artifact and numerical boundaries

| Concept | Meaning | Owner |
| --- | --- | --- |
| Artifact | Identified source snapshot, metadata, and stored tensors | Loader / format adapter |
| Model description | Architecture geometry, layer topology, weight roles, and input requirements | Model library / adapter |
| Source codec | Container byte encoding before import | Format adapter and typed import contract |
| Numerical representation | Values represented by codes, coefficients, and metadata | Seismic representation semantics |
| Physical layout | Placement and arrangement for an implementation | Seismic compiler/runtime |
| Residency policy | Which weights may be resident or streamed under a budget | Engine |

## Loading and weight ownership

- Validate geometry, role coverage, shapes, encoding, and package ownership before
  admitting numerical execution.
- GGUF and MLX/Safetensors adapters describe source data; they do not select kernels.
- Express imports, decoding, conversion, and relayout through typed Seismic operations.
  Preserve encoded meaning and coefficient precision; account for transient storage.
- Use bounded source reads and explicit transfer lifetimes. Publish complete imported
  resources; failures release unpublished allocations and staging.
- Equal-shaped roles may share compiled code but retain distinct artifact resources.
- Resident and streamed policies describe the same numerical values. Recurring
  streaming costs belong to execution; initial import belongs to preparation.

Device-free artifact interpretation is shared by embedded callers, measurement,
and explicit serving composition. Local MLX/Safetensors directories and GGUF files
produce the same architecture description and numerical loading contract. Format
adapters supply tokenizer and template metadata from the corresponding artifact;
the GGUF package is opened once and shared with its numerical worker. Package and
component identities are process-local identities of those open files, not content
digests. A persistent content identity, if needed, comes from acquisition rather
than a weight scan during engine loading. Host tokenizer and template preparation
may run while the numerical worker prepares, with readiness published only after
both sides succeed.

Serving does not substitute metadata from a different source. Serving validates tokenizer/projection identity,
context and host limits before importing weights, and establishes the execution
owner's storage budget before numerical loading. Device creation and hardware
settings belong to the host composition root; loading does not invent a profile.

Weight import and model compilation require automatic selection settings. Neither
API accepts a fixed candidate or a diagnostic lowering path. Numerical imports
retain logical source plans and select against their bound buffers and scalars
before compiling native code; incomplete selection cannot execute.

Image-tower interpretation is opt-in. For supported MLX Qwen towers, validate
patch-kernel axis order, square position-table geometry, complete rotary head
quarters, decoder output width, layer count, and every affine/norm/merger weight
role before numerical loading. Unsupported activation or deep-stack variants
fail explicitly. Text-only loading does not require image metadata or resources.

## Numerical composition

Model programs own topology and equations. The standard library owns reusable
projection, normalization, attention, recurrence, routing, codec, and sampling
compositions. Seismic owns their physical implementations.

- Preserve accumulation types, stored activation precision, and publication order.
- Vision layer normalization computes centered mean and variance in FP32. Affine
  bias is added to the FP32 projection accumulator before compact publication.
- The Qwen vision stem reorders channel/time/height/width patches to the artifact's
  time/height/width/channel order before BF16 publication and projection. Position
  interpolation preserves table-precision coefficient, product, and ordered-add
  rounding before adding the projected patch.
- Fusion can eliminate storage without removing observable rounding boundaries.
- Qwen vision rotary pairs corresponding channels across the two halves of each
  head, with separate height/width frequency ladders. It preserves head ordering
  and leaves values unrotated. Image attention covers every patch of one image,
  including later patches; decoder causal masking does not apply to that domain.
- Qwen vision blocks use tanh GELU; the spatial merger uses erf GELU. They are
  distinct operations. Residual, normalized, projected, and activated compact
  intermediates retain their publication boundaries throughout the block.
- Packed weights and KV retain their declared decode semantics through computation.
- Qwen draft head blocks may use dense or routed feed-forward weights. Routed
  heads use the same expert and shared-expert equations as routed target blocks;
  they do not require a dense feed-forward width in the artifact header.
- Fresh K/V participates in the current computation before persistent encoding;
  committed history is interpreted through its selected codec.
- Codec identity includes packing, metadata precision, and any rotation/codebook
  definition. A similarly named algorithm does not establish equivalent values.

Selected-vocabulary readout returns final-row logits in caller order, retaining
duplicates and the same normalization, activation publication, and accumulation
semantics as full logits. It projects only selected rows and transfers only their
results to the host. An empty selection performs state advancement without output
projection; out-of-vocabulary selections fail before tentative state reservation.
Sampling remains a separate device-side readout and does not consume host logits.

## Packed execution

```text
requests + tentative state views + conditioned inputs
    → row-local positions, visibility, routes, masks, and output selection
    → prepared Seismic execution
    → per-request outputs + shared completion
    → independent acceptance of sequence advances
```

| Contract | Behavior |
| --- | --- |
| Row isolation | Padding and peer requests never become visible history or reduction operands |
| Specialization | Stable capacity classes bound compiled geometry; actual history and visibility remain dynamic inputs |
| Readout | State-only, last/all logits, selected vocabulary, and sampled output have explicit meanings |
| Selected vocabulary | Ordered vocabulary and validation belong to the request; raw selected logits are not implicitly normalized or sampled |
| Preparation | A batch acquires tentative resources together and unwinds failed preparation without publishing state |
| Completion | Includes forward execution and any deferred control transfer or sampling |
| Acceptance | Advances commit independently after completion and request-level validation |

Batch preparation rejects aliased sequence owners before acquiring any pending
state. All successors remain private until shared completion; a preparation error,
malformed result, or unwind publishes none. Completed selection failures are
request-local and cannot commit numerical successors.

Qwen generation packs embedding and feedforward stages across request rows.
Stateful mixers consume per-request hidden views, positions, visible history,
destinations, and recurrent components. Scratch reused by equal geometries is
not overwritten until its previous use completes. Readout and sampling retain
request-local final rows, masks, seeds, and output positions.

The executor reports reclaimable resources for sets of sequences. It does not choose
victims. Prepared native execution belongs to [Seismic runtime](../seismic/runtime.md);
input continuation belongs to [inputs](inputs.md), and history ownership to [state](state.md).

## Single-session Qwen baseline

The Qwen executor accepts a multi-token proposal with explicit state-only,
final-row logits, or device-sampled readout. State-only execution omits the
vocabulary projection. Sampled execution transfers only the selected token to
the host after forward and selection complete. The baseline measurement path
requests final-row logits. Projections and feedforward computation span all input rows;
attention sees accepted history and the causal prefix of fresh rows. Convolution
preparation reads the accepted window plus earlier projected rows. Delta state
evolves in token order and publishes one final state. The proposal commits or
aborts the entire sequence advance, and ordinary decode continues from that
state. Packed weights are shared across cached row geometries.

This is a numerical baseline with dense activation-precision KV and explicit
ordered history ranges, including fragmented storage. It does not yet supply a
parallel chunked recurrence algorithm or establish optimal prefill performance.

The baseline harness loads an MLX artifact through the ordinary importer and
requires automatic-selection settings from its caller. It measures identical
forced token sequences, separates loading and cold execution, excludes a warm-up,
records three warm trials, and profiles components in a separate forward. Warm
trials that compile new kernels are rejected. All timings include final-logit
readback and sequence acceptance. The report retains the artifact and hardware
identities, tokens, timing samples and logits for cross-engine comparisons.

The harness does not provide a production hardware profile. Until a qualified
profile is supplied and selection completes, there is no native baseline result;
semantic-interpreter checks are numerical evidence only.
