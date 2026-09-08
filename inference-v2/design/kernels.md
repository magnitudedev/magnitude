# Kernel construction

**Numerical operations define behavior. Kernel plans arrange its execution. Reusable
Metal code and generated entry points realize those plans through MLX.**

This defines the construction boundary for owned kernels.
[Model composability](models/composability.md) owns architecture and operation contracts;
[optimization](models/optimization.md) owns implementation selection and qualification.
Kernel construction also serves state operations where the same execution mechanisms
apply. It does not own model assembly, request policy or physical allocation.

## Responsibilities

```text
Python/MLX computation + actual tensor/state views
    → captured operations + numerical/state requirements
    → automatic implementation selection + kernel plan
    → specialized Metal entry point + matching launch configuration
    → MLX arrays and execution dependencies
```

| Owner | Responsibility |
|---|---|
| Architecture | Equations, weight interpretation, connections, positions and required outputs |
| Numerical operation | Arithmetic, rounding, reductions, supported inputs and state effects |
| Kernel plan | Tile ownership, data access/reuse, placement, dependencies and execution partition |
| Metal implementation | Handwritten algorithm and typed tile/hook interface |
| Planner and generator | Connect compatible implementations and emit entry points |
| MLX invocation | Bind array operands and launch arguments; return arrays in the dependency graph |
| State and execution owners | Supply valid views, control commitment and retain resources through completion |

Model-facing operations take and return MLX arrays. An implementation may combine
standard MLX primitives and owned kernels. Custom kernels participate in ordinary MLX
dependencies; they do not create a separate device runtime. Scheduling supplies eligible
work and batching groups compatible operations without inspecting kernel plans.

## Implementation organization

`kernels/core/` owns capture, typed interfaces, automatic planning, source assembly
and MLX invocation/caching. Computational categories own operation declarations,
implementation bindings and adjacent `.metal` sources:
`contractions/`, `reductions/`, `attention/`, `recurrence/` and `state/` for KV writes.
Encoded operands belong with contractions; architecture loading binds those operands.
Categories depend on the core and explicit numerical dependencies, never model or
scheduler implementations. Model components keep their semantic IDs and call the
category operations.

`@kernels.kernel(source=..., function=...)` binds one Python declaration to a
handwritten Metal function. Its body receives tensor descriptions and static parameters,
validates them and returns one execution description. Calling it with arrays returns
MLX arrays. Output geometry comes from that description; there are no separate numerical
leaf subclasses, registration callbacks, inference methods or author-written launch calls.
A complete opaque dispatch uses the same decorator with a `Dispatch` description.

Ordinary Python functions compose these calls with MLX expressions. `kernels.compile`
accepts MLX's compilation options and returns its native compiled callable. Our pass runs
inside MLX tracing, then hands the transformed graph back to MLX. A small native bridge
inspects graph dependencies and reconnects original primitives; unfamiliar operations
retain their actual MLX implementation and streams. There is no export/replay interpreter
and no second execution dispatcher. Captured input/output state belongs to MLX, and the
user function runs once per native trace. Shapeless compilation retains MLX's behavior
without applying fixed-shape fusion. It does not confer symbolic-shape or derivative
support on kernels that lack it.

`kernels.explain(compiled)` and `kernels.artifact(compiled)` inspect the latest trace
without wrapping the native callable. No Python planning occurs on warm compiled calls.

A binding describes logical tiles, dtype, thread ownership, reduction completion,
participation, scratch and effects at a Metal function or hook boundary. The planner
derives connections and launch boundaries from these interfaces and graph dependencies.
The generated launch binding supplies both the kernel signature and MLX arguments.
Source dependencies are immutable content snapshots, emitted once per assembly.
The core checks declared composition requirements; qualification establishes that
handwritten Metal actually meets them. Current mechanisms cover completed scalar/SIMD
tiles, blocked row fragments, ordered traversal and cooperative row hooks. New collective
or exchange mechanisms need qualification; arbitrary Metal is not parsed or reordered.

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

Dense tiles share weights across input rows. Wide multi-row projections can materialize
native input preparation and pack sums once for reuse across output tiles. Inline and
materialized preparation preserve the same arithmetic; eliminating an intermediate is
not automatically an optimization. Reused packs cache unscaled integer coefficients in
half-width registers: four-bit shifted masks and eight-bit integers are exactly
representable. Scales, affine correction and accumulation remain FP32. Fixed register
loops expand statically; the lane/K traversal and reduction order remain unchanged.
Gate/up shares route grouping and input traversal,
with separate handwritten coefficient/accumulator bodies for each projection.

Expert schedules use direct fused execution
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
architecture from names or introduce a second model graph. Mechanisms validate proposed
regions and return plans carrying their emission decisions. The graph planner selects
among these plans subject to dependency, stream and resource constraints. Emission uses
the selected plan without re-running a separate eligibility or interface dispatch tree. Unsupported connections remain
explicit array boundaries, with retained MLX execution among the eligible alternatives.

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

Numerical loops and statements remain Metal, not Python statement builders. Complete
tile calls expose returned fragments. Typed input/output hooks and ordered drivers
expose internal composition where it enables useful reuse. Generated adapters connect
those hooks; an opaque function does not acquire internal fusion from metadata alone.

Use structured plan data and template specialization instead of scattered source-string
substitutions. MLX may receive source text at its API boundary; that text is a generated
artifact, not the authoritative representation of the operation. Retain inspectable
generated source so failures can be traced to the selected plan and source functions.

Metal helper calls compose inside the generated kernel without creating MLX arrays or
additional launches. Python composition remains ordinary array computation. Complete
attention, recurrence, embedding, routing and KV-write bodies can bind as opaque kernels
through the same invocation boundary; opacity does not imply internal fusion. Our planner
implements automatic custom-region fusion; MLX compilation supplies execution and does
not imply that arbitrary custom kernel bodies will merge. Warm compiled execution
performs no Python capture, plan search or source generation.

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
changes. Reject unsupported combinations before submitting device work. Make required row
contiguity explicit in the MLX graph. Unrecognized operations remain native boundaries;
invalid declarations fail instead of silently changing implementations.

Reuse generated kernels across compatible invocations. Caches are bounded, and
standalone specializations distinguish the MLX device/stream; compiled functions retain
MLX's stream semantics. Declarations snapshot source code;
reloading a changed declaration creates a new identity, while changing files beneath
an already-bound executable does not silently change its behavior. Cache validity includes every
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
