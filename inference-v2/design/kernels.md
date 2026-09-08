# Kernel construction

**Numerical operations define behavior. Kernel plans arrange its execution. Reusable
Metal code and generated entry points realize those plans through MLX.**

This defines the intended construction boundary for owned kernels.
[Model composability](models/composability.md) owns architecture and operation contracts;
[optimization](models/optimization.md) owns implementation selection and qualification.
Kernel construction also serves state operations where the same execution mechanisms
apply. It does not own model assembly, request policy or physical allocation.

## Responsibilities

```text
Bound operation + actual tensor/state views
    → numerical building blocks + kernel plan
    → specialized Metal entry point + matching launch configuration
    → MLX arrays and execution dependencies
```

| Owner | Responsibility |
|---|---|
| Architecture | Equations, weight interpretation, connections, positions and required outputs |
| Numerical operation | Arithmetic, rounding, reductions, supported inputs and state effects |
| Kernel plan | Tile ownership, data access/reuse, placement, dependencies and execution partition |
| Generator and Metal library | Realize the plan with reusable arithmetic and target code |
| MLX invocation | Bind array operands and launch arguments; return arrays in the dependency graph |
| State and execution owners | Supply valid views, control commitment and retain resources through completion |

Model-facing operations take and return MLX arrays. An implementation may combine
standard MLX primitives and owned kernels. Custom kernels participate in ordinary MLX
dependencies; they do not create a separate device runtime. Scheduling supplies eligible
work and batching groups compatible operations without inspecting kernel plans.

## Implementation organization

`kernels/core/` owns plan records, source assembly and MLX invocation/caching.
Computational categories own their planning functions and adjacent `.metal` sources:
`contractions/`, `reductions/`, `attention/`, `recurrence/` and `state/` for KV writes.
Encoded operands belong with contractions; architecture loading binds those operands.
Categories depend on the core and explicit numerical dependencies, never model or
scheduler implementations. Model components keep their semantic IDs and call the
category operations.

A plan binds named inputs, output shapes/dtypes, launch geometry, template parameters
and finite scalar specializations to a source dependency graph. The source graph holds
immutable content snapshots. Assembly emits each dependency once; the runtime derives
both the kernel signature and positional MLX arguments from the same named bindings.
Kernel-family source encodes the selected arithmetic and reduction ordering. The core
validates construction; it does not prove arbitrary Metal arithmetic correct.

## Numerical building blocks

Use reusable functions or templates for computations with explicit composition rules:

| Building block | Preserved meaning |
|---|---|
| Encoded access and contraction | Logical weight selection, quantization arithmetic and per-output accumulation |
| Row reduction and finalization | Complete reduction domain, casts, normalization, residuals, gates and activations |
| Attention summary | Visible logical key segment, score/value arithmetic and ordered summary combination |
| Recurrent transition | Ordered state update, outputs and accepted-prefix recovery |

Architecture differences remain explicit compositions of these operations. Weight
addressing may vary between dense, selected-expert and streamed execution without
changing contraction arithmetic. Sharing a weight tile across requests does not require
combining their accumulators. A physical expert slot has no numerical meaning.

Mathematical equivalence over reals does not establish finite-precision equivalence.
Required casts and reduction structure remain explicit even when intermediate tensors
are eliminated. A different accumulation algorithm needs its own qualification; it is
not an incidental consequence of batch padding or a launch-size change.

### Affine contractions

Dense projections and selected experts share encoded-pack arithmetic and a row-tile
contraction. Each output retains the independent-row K traversal, SIMD reduction and
dtype boundaries. Row tiles reuse encoded weights without first dequantizing them into
a different multiply/add expression. Short-query binding covers request batches and
verification positions; unsupported encodings use independent upstream rows, while wide
queries retain the upstream matrix path.

Dense tiles share weights across input rows. Expert schedules use direct fused execution
for sparse assignments and grouped execution when reuse can amortize sorting and the
intermediate down result. Grouped output returns to logical assignment order before
weighted combination. Changing a tile, assignment order or physical bank capacity may
change scheduling, never a row's arithmetic. Gate/up activation and ordered expert
combination retain explicit native-dtype rounding boundaries.

## Kernel plans

A kernel plan is a bounded execution IR for a numerical region, referencing its actual
building blocks and operands. It records:

- **Logical domains and access:** output tiles, valid elements, operand mappings and sharing.
- **Arithmetic dependencies:** reductions, rounding boundaries and producer/consumer order.
- **Physical arrangement:** threadgroup/lane ownership, register or shared-memory values,
  device intermediates and synchronization.
- **Boundary obligations:** outputs, state reads/writes, alias constraints and launch geometry.

Plans describe execution beneath existing components. They do not reconstruct an
architecture from names or introduce a second model graph. Start with deliberate plan
families for supported operations; automatic fusion and schedule search are additional
capabilities, not implied by having an IR.

For a gated projection, a plan can express:

```text
Shared input tile
    ├── Gate contraction → required cast
    └── Up contraction   → required cast
                    ↓
          Architecture's activation
                    ↓
              Output tile
```

The two contractions retain their arithmetic while sharing input access. Their results
can stay local through activation. A subsequent down projection consumes the complete
intermediate dimension; that dependency determines whether further fusion is legal and
useful. The plan does not assume the entire feedforward must be one kernel.

## Metal sources and generation

Reusable arithmetic and tile functions live in ordinary `.metal` source files.
Generation specializes and connects those functions, emits the kernel entry point,
and derives its matching MLX launch signature. It must not maintain a separate operand
order or output layout in an independently written launcher.

Use structured plan data and template specialization instead of scattered source-string
substitutions. MLX may receive source text at its API boundary; that text is a generated
artifact, not the authoritative representation of the operation. Retain inspectable
generated source so failures can be traced to the selected plan and source functions.

Metal helper calls compose inside the generated kernel without creating MLX arrays or
additional launches. Python composition remains ordinary array computation. Fusion
between custom kernels must be implemented explicitly; MLX compilation does not imply
that arbitrary custom kernels will merge.

## Legal composition

A fused region must preserve the numerical operation and every required observation.
Each output or state write has an owner. A consumer executes only when its required
values and reductions are complete in the synchronization scope that contains them.
Local barriers do not establish ordering across independent threadgroups.

Logical reductions are independent of physical traversal. Attention's per-query
visibility and prescribed summary tree cannot depend on a peer's history length or
page-table capacity. Fragmentation changes addresses, not the reduction. A logical
summary boundary may remain inside a threadgroup; it need not materialize in memory.

Plans consume state views supplied by their owner. They cannot infer permission to
mutate an MLX input from apparent exclusive use. State writes, aliasing and scratch
reuse must satisfy the supported array/runtime contract and actual completion
dependencies. Accepted-prefix visibility and physical reclamation remain separate.

Component boundaries do not require launches, synchronization or materialization.
Conversely, a component may need several kernels. Composition must account for the
whole region's dependencies and live values rather than assuming individually fast
tiles remain fast together.

## Specialization and evidence

Specialize on relevant encoding, dtype, geometry, layout, numerical contract and target
capabilities. Dynamic validity and positions remain operands where possible. Kernel
selection must preserve the operation's contract across batch membership and storage
changes. Reject unsupported combinations before submitting device work.

Reuse generated kernels across compatible invocations. Cache validity includes every
code-affecting plan/source dependency and compilation option; operand addresses and
benchmark identity do not define a numerical implementation. Generating a kernel is
not a reason to allocate a new [component ID](components.md#identity-and-provenance).

[Performance capture](performance.md#binding-the-executed-composition) records the actual
specialization and executable dependencies. Formulation and benchmarks remain outside
runtime generation. A selected plan informs implementation costs; its avoidable traffic
or synchronization cannot lower the theoretical ceiling for the component contract.

Validate building blocks independently and generated regions against their complete
contracts, including state and continuation. Qualification includes physical relocation,
row regrouping and permitted query partition changes. Shared generator code cannot
serve as its own independent numerical oracle.
