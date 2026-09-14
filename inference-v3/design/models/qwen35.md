# Qwen 3.5

**Qwen defines geometry, weight roles, mixer topology, feedforward topology and
finite-precision equations as ordinary Ops composition; it owns no
kernel, candidate table, scratch arena or compiler path.**

## Model composition

```text
Qwen 3.5
├── Optional prepared media
│   └── vision encoder + projector ──► decoder-width conditioned spans
├── Token embedding with conditioned-span replacement
├── Repeated block, mixer kind from the description
│   ├── Input normalization
│   ├── Attention mixer
│   │   ├── grouped Q/gate, K and V projection
│   │   ├── Q/K normalization and rotary
│   │   ├── versioned KV append and causal attention
│   │   └── output gate and projection
│   ├── Recurrent mixer
│   │   ├── grouped QKV, gate, beta and alpha projection
│   │   ├── convolution, decay and delta-state transition
│   │   └── gated normalization and output projection
│   ├── Residual transition
│   └── Feedforward
│       ├── dense gated feedforward, or
│       └── router + selected experts + shared expert
└── Selected rows ──► output normalization ──► readout
```

The description names these roles and their geometry independently of a
container. GGUF and other formats map their names and stored tensors to those
roles. Artifact identity is checked before the model function is compiled.

## Ops boundary

The model uses semantically strong operations for quantized projections,
attention, recurrence, routing, selected experts, shared experts, state access
and normalization. Those operations preserve the information required for
specialized prefill and decode lowering.

The block itself remains ordinary model composition. Ops may match a
complete attention, recurrence or expert producer-consumer region and emit one
authored fused TileLang schedule. Such a lowering is described by its
mathematics, geometry, representation and effects, never by the Qwen name.

Dense and routed variants use the same tensor system. MoE is not a second model
executor: routing, grouping, expert projection, weighted reduction and shared
expert combination are tensor operations within the same graph, resource plan
and completion.

## Precision

The model declares its observable finite-precision contract. Production Qwen
descriptions currently select FP16 activation and KV storage with FP32 reduction
and recurrent state where declared. Norms, rotary,
projection boundaries, gates, residuals, recurrent state, reductions and logits
retain their required accumulation and storage boundaries through fusion. A
lowering that changes an observable rounding point is not a substitute for the
same model contract.

The routed-expert primitive explicitly carries its hidden storage dtype. Gate and
up contractions accumulate in FP32 and publish to that dtype; its internal gated
product publishes once to that dtype. Each down contraction publishes before
multiplication by FP32 routing scores. Contributions accumulate in route order in
FP32 and the mixture publishes once. In particular, combining lane-local expert
partial sums before each expert's projection publication changes the equation.
Resident grouped, selected decode and streamed bodies must share these boundaries.
The shared expert branch remains an explicitly composed linear/SiLU/multiply
formula and retains its separately observable SiLU publication.

## Specialization

Prefill, decode and verification compile as different tensor specializations
because their optimal algorithms and geometries differ. Packed counts, requested
readout rows, causal read/write geometry and representation constraints are
explicit graph inputs or static specialization facts.

History lengths remain dynamic within bounded capacity classes. A growing
history changes resource metadata, not model topology. A state-only invocation
omits the stateless suffix whose result has no consumer.

## State

Attention layers receive logical KV views backed by Ops resources;
recurrent layers receive versioned recurrent banks. The tensor graph orders
writes and subsequent reads. The Qwen sequence owner decides whether each
tentative advance commits, aborts, forks or becomes a checkpoint.

Packing combines several sequences into one tensor invocation while preserving
row-local positions, visibility, routing, reductions and acceptance. Padding or
peer rows never enter another row's arithmetic.

## Vision

The Qwen input adapter interprets its processor contract, expands media
placeholders and constructs the required coordinates and spans. The vision tower
and projector are stateless Ops tensor functions whose outputs are
decoder-width features. Their resources and batching are independent of decoder
state and decoder batching.

The language function consumes prepared feature slices through the model input
contract; it does not inspect raw images or placeholder token identities. A
partial legal continuation retains the coordinates and remaining feature slice
needed to resume. Fully consumed media does not remain an operand of ordinary
text decode. See [inputs.md](inputs.md).
