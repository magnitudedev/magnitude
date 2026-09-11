# Qwen 3.5 dense

**The architecture names its geometry and weight roles; a container supplies them;
the program composes operations and is captured once per invocation geometry.**

## Assembly

```text
Qwen 3.5 dense
├── Embedding
├── Repeated block (mixer kind per layer from the description)
│   ├── Input norm
│   ├── Attention mixer
│   │   ├── Grouped q/gate, k, v projections
│   │   ├── Rotary coordinates and q/k norms
│   │   ├── KV append ── State: KV pool
│   │   ├── Causal attention over visible history
│   │   ├── Sigmoid gate
│   │   └── Output projection
│   ├── Recurrent mixer
│   │   ├── Grouped qkv, gate, beta, alpha projections
│   │   ├── Convolution, norms, decay and gates ── State: recurrent banks
│   │   ├── Gated delta update
│   │   ├── Norm, SiLU gate
│   │   └── Output projection
│   ├── Residual add
│   └── Feedforward norm → Gated feedforward → Down projection → Residual add
└── Selected rows → Output norm → Readout

Description: geometry + weight roles     ◄── formats: GGUF, MLX (one inspector each)
```

| Boundary | Rule |
|---|---|
| Description ↔ container | The description names roles and geometry with no container in view; a format maps its own names to those roles. This is the one place model meets format, one file per format |
| Description ↔ weights | The description carries the artifact identity; a binding from another artifact fails before any work |
| Program ↔ kernels | The program asks operations; it names no schedule and imports none |

## Rounding points

The program carries one precision. Where the reference implementation rounds
natively, so does the program; where it does not, FP32 is kept across the
expression regardless of preset:

| Boundary | Follows the preset | Always FP32 internal |
|---|---|---|
| Norms, rotary and q/k norms, attention gate, feedforward gate | ● | |
| Residual adds, recurrent SiLU gate | | ● |
| Reductions, delta state, decay, statistics, logits | | ● |

## Invocation geometry

A forward is prepared once for a geometry and replayed for every step that shares
it:

```text
geometry = changing-operand layout · packed counts · requested output rows ·
           KV read and write geometry of the first attention layer
same geometry ──► rebind and launch     different ──► release plan, rebuild, capture
```

Decode at a fixed batch size is one geometry for as long as histories stay in
their capacity classes; a prefill chunk of a new length is another. Layer-by-layer
Python work exists only when the geometry changes.

The arena holds every intermediate: the program's slots sized by row count, and
the regions operations declared, merged by name. A slot survives a plan change;
operation scratch is released with the plan. Growth of either is a geometry change.

## State

| Rule | Reason |
|---|---|
| The KV pool is laid out for the attention layers only | Recurrent layers keep no history; they keep a state |
| Recurrent state lives in reusable banks | A new state needs banks; an idle bank is reused rather than reallocated |
| An advance extends a private KV tail in place, otherwise claims new runs | Sharing with a checkpoint forbids the in-place write |
| An advance anticipates the input horizon | The pool sizes new slabs toward the known prompt length without claiming ahead |
| A state-only prefill chunk stops after the last mixer | Its output has no consumer; the recurrent state is already complete |

Packing is the runtime's: several sequences' tokens, positions and coordinates
become one forward; conditioning features overwrite the embedding rows they
replace; each sequence's advance accepts or aborts on its own.
