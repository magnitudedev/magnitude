# Tensor system

**Magnitensor is the sole numerical boundary of the engine: model code composes
lazy inference tensors, and Magnitensor turns that composition into optimized,
portable TileLang execution without exposing either its graph or TileLang.**

## Shape of the system

```text
Magnitude model function
        │ ordinary tensor composition
        ▼
Magnitensor
  tensor compiler core
  inference operation library
  portable TileLang kernel library
        │ opaque final compilation units
        ▼
TileLang language, compiler and runtime
```

The compiler core owns tensor values, tracing, the semantic graph, lowering,
fusion, representation choice, materialization, temporary storage, compilation,
binding and completion. The operation library gives that machinery an
inference-specific vocabulary: quantized projection, normalization, attention,
recurrence, routing, experts, embeddings and state access.

Magnitude imports Magnitensor. Magnitensor never imports Magnitude. Magnitensor's
kernel library authors schedules against the public TileLang language, while its
runtime adapter owns compilation and execution. No TileLang or TVM value crosses
Magnitensor's public boundary.

## Tensor functions

A model is an ordinary function over lazy tensors and explicit mutable resources.
Calling it during tracing records computation; it performs no device work. Python
control may depend on static architecture facts, so fixed layer structure is
unrolled naturally. Dynamic tensor control is an explicit tensor operation.

| Kind of input | Meaning during compilation |
|---|---|
| Dynamic tensor | Geometry and dtype constrain compilation; storage is bound per invocation |
| Immutable weight | Numerical representation and value identity are captured; storage is bound once |
| Mutable resource | Access and alias identity are explicit; storage is bound per state view |
| Static value | Controls architecture or specialization and is folded into the graph |

Tensor values are immutable. A mutable resource is versioned: a write consumes
one version and produces the next, so dependencies remain explicit without
hidden Python mutation. Magnitude decides whether a state advance is accepted;
Magnitensor only guarantees physical ordering and lifetime.

## Compilation

```text
trace and verify
      ▼
typed semantic graph
      ▼
normalize without erasing coarse operations
      ▼
enumerate legal operation and region lowerings
      ▼
select regions, representations and layouts together
      ▼
materialize boundaries and assign temporary storage
      ▼
order selected kernel work and choose maximal submission units
      ▼
emit and finalize each TileLang compilation unit once
      ▼
compiled callable with static bindings and dynamic inputs
```

The graph is immutable, typed and first-order. Nodes name semantic operations;
values carry shape, dtype, representation and observable layout. Aliases,
resource reads and resource writes are part of operation contracts, not a second
execution graph.

Compilation normally specializes prefill, decode and verification separately.
Continuously changing quantities remain operands or are grouped into bounded
geometry classes. Compilation identity follows graph structure and declared
static facts, never a model name, request, benchmark or machine hostname.

## Region selection

An implementation candidate covers one semantic operation or a bounded connected
region. It declares its boundary values, accepted representations and layouts,
resource effects, workspace, numerical contract and portable kernel emission.
It does not return a finalized `PrimFunc`. Candidate applicability is a fact;
measured cost chooses among applicable facts.

Straight-line choices are solved together rather than selected independently per
operation. Structured branches use dominance and post-dominance to keep regions
closed: a producer with several consumers can disappear inside a fusion only when
all relevant paths and externally used results are represented by the region.

| Optimization | Representation |
|---|---|
| A different algorithm for one semantic operation | Operation lowering candidate |
| One kernel spanning adjacent semantic operations | Region lowering candidate |
| Pointwise or view work absorbed by an anchor | Generic cheap-operation fusion |
| Several ordered device kernels behind one host submission | Launch sequence, not fusion |

Fusion means one device kernel and no globally materialized interior value.
Materialization and temporary allocation happen only after the selected cover is
known. Live intervals and legal aliases share an aligned arena; persistent weights
and state resources remain externally owned.

## Submission planning

Region selection answers which kernels compute the graph. Submission planning
answers how much ordered kernel work one host entrypoint contains. These are
separate compiler decisions: an operation or layer boundary never creates a
submission boundary by itself.

Selected lowerings contribute Python-authored `T.Kernel` work to a shared
compilation unit. After storage and dependency planning, Magnitensor uses
TileLang's public eager construction facility to declare the dynamically
determined ordered ABI, invokes selected authored schedules in dependency order,
and finalizes one `PrimFunc`. The default unit for a prefill, decode or
verification specialization is the whole tensor function.

A unit is split only by a required host observation, a data-dependent host
decision, an unsupported dependency, or a qualified compiler/runtime limit.
Those causes are recorded. A split is never introduced merely because two
regions were selected independently.

```text
selected graph cover + storage plan
              │ ordered kernel emission
              ▼
maximal TileLang PrimFunc ──► one pre-bound native entrypoint
```

Warm host work is proportional to dynamic bindings and true submission units.
It must not scale with model depth, operation count, weight count or device
kernel count. Device kernels remain distinct unless fused, but their launches
are encoded by the native TileLang entrypoint rather than a Python loop.

## Capability and portability

The hardware capability contract runs from TileLang to Magnitensor. Magnitude is
not a participant and remains hardware-unaware.

TileLang reports behavioral facts that affect a schedule: subgroup and matrix
geometry, supported dtypes, memory scopes and capacities, movement,
synchronization, atomics, alignment and launch limits. Magnitensor may choose
different portable strategies from those facts. Backend-neutral therefore does
not mean schedule-neutral.

```text
capability predicate ──► fragment-tiled / subgroup-streaming /
                         split-reduction / persistent strategy
                                      │
                                      ▼
                         portable TileLang program
```

Strategies are named by mechanism and never by vendor. Magnitensor neither probes
hardware nor maintains a parallel target database. A missing performance fact is
a TileLang target-contract gap; an unexpressible mechanism is a TileLang language
gap; poor native realization is a TileLang lowering or runtime defect. None
permits a backend side channel.

Kernel composition is Python composition. Magnitensor never generates Python or
TileLang source strings, calls `exec`, fabricates Python AST, or manipulates TIR.
TileLang owns the eager IR builder that turns dynamic parameter declarations and
ordinary TileLang macro calls into a valid `PrimFunc`.

## Compiled execution

A compiled callable retains immutable bindings, maximal compiled units, one
planned temporary arena, dynamic binding descriptions and compilation
provenance. Magnitensor decides which operands are static; TileLang realizes
their partial binding against the compiled ABI. Submission binds only dynamic
inputs, invokes each native entrypoint once, and returns output tensors with one
completion obligation. Selection, graph traversal, allocation planning,
compilation, tuning and per-kernel Python dispatch are absent from the invocation
hot path.

Physical resources stay claimed through completion. Logical commit is a separate
Magnitude event: completion proves that bytes exist, while commit decides whether
those bytes become model history.

## Boundaries

| Magnitude owns | Magnitensor owns | TileLang owns |
|---|---|---|
| Model topology, request work, logical state, container parsing, modality policy | Tensor semantics, compiler decisions, representations, buffers, portable kernels, submission | Kernel language, compiler invariants, target capabilities, lowering, code generation and execution adapters |

Multimodality does not alter this boundary. Magnitude prepares and aligns media;
vision, audio, projectors and language computation are ordinary Magnitensor tensor
functions using the same compiler and execution contracts.
