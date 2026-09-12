# Kernel optimization

**Kernel optimization begins by identifying the dominant avoidable cost at the
widest relevant execution scope, then changing the structure that causes it
without crossing the Magnitude–Magnitensor–TileLang boundaries.**

The objective is not a locally faster instruction sequence. It is the best
execution of the required model computation: numerically correct, composable
with adjacent work, portable through TileLang and competitive in enclosing
prefill, time-to-first-token and decode measurements.

This guide defines how to reason about and develop that execution.
[Tensor compilation](../tensor-system.md) defines selection and composition,
[portable kernels](../kernels.md) define the TileLang boundary, and
[performance](../performance.md) defines acceptance evidence.

## Optimize the actual execution

### Begin from obligations

Separate what must happen from how the current implementation happens to do it.
The obligations are outputs, observable precision and rounding, state effects,
aliasing, ordering and completion. The following are normally implementation
choices:

- the algorithm and decomposition;
- intermediate representations and layouts;
- tile shapes, work ownership and traversal order;
- materialization and fusion boundaries;
- the number of device kernels and native submissions;
- specialization and binding time.

Preserving the existing choices is not conservatism when those choices cause
the gap. It prevents the necessary optimization.

Trace each important value from production to final consumption:

| Question | Cost it reveals |
|---|---|
| What creates it, and which inputs actually vary? | Repeated preparation, decoding or binding |
| Which consumers need it, and at what common scope? | Missed reuse or an incorrect fusion boundary |
| Where is it represented and materialized? | Device traffic, conversion and scratch cost |
| Which threads own it and exchange it? | Idle lanes, bank conflicts and synchronization |
| What dependency delays its consumer? | Serialization and unavailable overlap |
| How does its work reach the device? | Python, binding and native submission overhead |

Count costs at the level where they are paid. Requested loads are not necessarily
device-memory traffic; a source-level expression is not necessarily one machine
instruction; several device kernels can be one native submission. A useful cost
model distinguishes arithmetic, each memory level, conversion, synchronization,
occupancy, device launch and host invocation.

### Find the widest dominant cause

Optimization scope follows the cause rather than the file containing the slow
code.

```text
instruction inefficiency
        ▼
kernel schedule or layout
        ▼
region algorithm, fusion or representation
        ▼
whole-function materialization and submission
        ▼
engine phase and state pipeline
```

Start at the enclosing phase and descend until one owner explains most of the
recoverable gap. A slow leaf is irrelevant when it is not a meaningful fraction
of the parent. Conversely, a fast leaf does not compensate for hundreds of host
dispatches or a globally materialized intermediate around it.

## Take the structural leap

Incremental tuning is appropriate only while the current structure can plausibly
reach the target. Once evidence identifies a structural ceiling, further parameter
sweeps are not progress.

| Structural ceiling | Required leap |
|---|---|
| Work is fundamentally unnecessary | Remove it or choose an algorithm that never produces it |
| A producer is written globally only for one consumer | Fuse or preserve it in the consumer's useful representation |
| The same invariant is prepared repeatedly | Prepare or bind it once at the widest safe reuse scope |
| Quantized data is expanded before use | Decode the consumed tile at the point of arithmetic |
| A phase is dominated by Python or launch count | Assemble maximal ordered work behind one native entrypoint |
| Dynamic work launches for the worst case | Use bounded persistent workers or device-side work claiming |
| A primitive graph hides an inference algorithm | Retain the semantic operation and lower it as an algorithm |
| A schedule leaves the target's fast mechanism unreachable | Select a capability-specialized portable schedule or repair TileLang lowering |
| Tuning trades one bottleneck for another at the same magnitude | Change dataflow, ownership, layout or representation together |

A proposed leap must state four things before implementation:

1. the dominant cost it eliminates;
2. the new cost or constraint it introduces;
3. why the numerical and state contract remains valid;
4. the operating regimes in which the argument holds.

Coupled changes should be designed and implemented together when none can win
alone. A fused quantized projection, for example, may require a compatible packed
layout, tile ownership and matrix-instruction schedule. Testing each piece inside
the old dataflow can falsely reject the coherent design.

Do not repeatedly benchmark either of these:

- a path whose cost model cannot reach the target even if its remaining local
  inefficiencies disappear;
- a path that reaches a favorable number by violating the intended architecture,
  numerical contract or production composition.

Such measurements answer no useful acceptance question. First build a candidate
whose structure is both legal and plausibly sufficient.

### Use the cheapest discriminating check

Kernel work advances through an explicit evidence ladder:

1. compile-free whole-model analysis proves graph shape, candidate coverage,
   memory planning and maximal submission structure;
2. focused construction and reference tests prove emitted schedule structure and
   semantics;
3. an isolated schedule benchmark measures the affected region;
4. a tiny whole-model execution proves physical composition;
5. real model startup proves artifact integration;
6. session benchmarks provide final prefill, decode and end-to-end evidence.

A failed level is repaired at that level. Long model loads and session workloads
must not be repeated while a compile-free plan or focused test already identifies
the defect. Conversely, isolated latency does not establish acceptance when the
change alters whole-program composition or state behavior.

## Phase-specific leaps

The same semantic operations require different physical strategies in different
generation phases. Shared tensor composition does not imply one universal
schedule.

### Prefill

Prefill exposes matrix parallelism and is usually governed by arithmetic intensity,
weight traffic and intermediate materialization. The important leaps are:

- use matrix instructions through tiles that keep operands reusable in the
  appropriate fast memory;
- consume packed weights directly and combine decoding, scaling and matrix work
  so expanded weights are neither retained nor reread;
- fuse bandwidth-only producers and consumers around the dominant contraction
  when this removes global traffic without destroying its matrix schedule;
- use streaming or tiled attention algorithms that avoid materializing the score
  matrix and preserve stable online-reduction numerics;
- choose layouts and traversal jointly across adjacent contractions rather than
  paying conversions between independently optimal kernels;
- keep full, tail and long-context regimes explicit so an interior-tile result is
  never mistaken for a complete implementation.

When contraction work dominates, reducing a small launch cost is not the leap.
The leap is a better contraction dataflow, representation or matrix schedule.

### Time to first token

Time to first token includes prefill but is not synonymous with it. It also exposes
admission, preparation, compilation, binding, submission, state publication and
sampling. Kernel work therefore participates in a wider latency chain.

Compilation, autotuning and invariant weight preparation stay outside warm request
execution. Immutable operands are bound once. The complete prefill program is
submitted through the smallest number of genuine native entrypoints, and required
state publication is not hidden outside the measured path.

### Decode

Single-token decode has little matrix parallelism and repeatedly traverses model
depth. Its characteristic deficit is often weight bandwidth plus fixed host and
device-launch overhead. Prefill-shaped kernels with smaller dimensions are not a
decode strategy.

The decisive changes are:

- select schedules for narrow activation geometry and streaming weight access;
- fuse normalization, quantized projection preparation, activation, residual and
  state work where the combined schedule removes traffic profitably;
- avoid materializing values that only bridge adjacent work;
- assemble the whole specialized tensor function, or the maximal legal parts of
  it, as one TileLang program with ordered device kernels behind one native host
  entrypoint;
- partially bind weights and stable resources so warm host work depends only on
  dynamic inputs and true submission units, never layer or weight count.

One native submission does not mean one device kernel. Fusion removes device
materialization; submission assembly removes repeated host dispatch. They solve
different costs and are selected independently.

### Mixture of experts

MoE performance is a pipeline property:

```text
scores → top-k selection → route packing → expert work → weighted combine
```

Optimizing expert GEMM alone is insufficient when routing materializes expanded
token–expert structures, launches a worst-case grid or forces repeated layout
conversion. The coherent design:

- retains routing and expert counts as device data;
- packs work into a representation consumed directly by the expert schedule;
- groups quantized expert work without expanding complete expert weights;
- represents runtime-valid rows while retaining statically optimized physical
  matrix tiles;
- avoids launches proportional to every possible expert when actual occupancy is
  sparse;
- combines results without reconstructing a dense token–expert tensor.

For highly variable expert occupancy, a static maximum grid with inactive blocks
is structurally weak. A bounded persistent grid can claim packed work from a
device queue, preserving a static launch contract while executing only actual
work. This is a different scheduling strategy, not a patch to the maximum-grid
kernel.

### Long-context attention

Long context changes both the algorithmic working set and state traversal. Use an
online reduction that keeps only the running normalization state, lay out KV state
for the traversal actually performed, and partition heads, query rows and history
tiles so reductions remain local as long as possible. Avoid copies introduced
solely to make a generic contraction convenient. State indirection and boundary
handling are charged as part of the attention region.

### Quantized projections

Storage, decode arithmetic, multiplication and accumulation precision are separate
choices. Preserve packed information until consumption, load aligned packets,
decode each packet at the scope where its values are reused and feed the target's
matrix mechanism without a global expanded representation. Prove range, signedness,
scale application and rounding independently from the schedule.

### Reductions and pointwise work

Normalization, activation, scaling, masking and residual arithmetic are usually
memory-bound. Their strongest role is often as absorbed work around an anchor
kernel. A standalone optimization matters only when the operation must materialize
or remains a significant enclosing cost. Fusion is rejected when it lengthens
live ranges, raises register pressure or constrains a dominant matrix schedule
more than the eliminated traffic is worth.

### Multimodal computation

Vision, audio and projectors use the same Magnitensor and TileLang path but retain
their own semantic operations and schedules. Optimize their tensor regions normally;
do not introduce modality branches, media objects or preprocessing policy into
generic kernels. Once projected embeddings enter the language model, their source
modality is irrelevant to language-kernel selection.

## Engineer the resource trade

Every optimization moves pressure between resources.

| Change | Removes | Commonly introduces |
|---|---|---|
| Fusion | Launches and global intermediates | Longer live ranges, shared scheduling and register pressure |
| Larger tiles | Repeated loads and boundary overhead | Lower occupancy and larger tails |
| Persistent work | Empty launches and host decisions | Atomic queue traffic and fairness constraints |
| Shared preparation | Repeated decode or conversion | Retained storage and synchronization |
| Recomputation | Intermediate storage and traffic | Arithmetic and dependency depth |
| Compact representation | Bandwidth and capacity | Decode instructions and packing constraints |
| Split reduction | Serial dependency | Partial storage, atomics or a second phase |

Choose representation, location, lifetime and sharing scope together. Registers
are not merely faster memory: they determine occupancy and instruction scheduling.
Threadgroup memory enables cooperation but adds barriers and bank behavior. Device
memory can be the right explicit boundary when fusion would otherwise destroy
parallelism or increase live state excessively.

Optimize useful throughput rather than occupancy, instruction count or fusion
count in isolation. Each is evidence only when connected to the dominant cost.

## Use TileLang as the kernel boundary

### Write portable schedules, not backend bypasses

All numerical device work is expressed through TileLang. A vendor library, direct
Metal/CUDA/HIP call, framework tensor operation or driver helper is not a valid
performance escape. It breaks composition, moves numerical semantics into the
runtime and lets backend behavior differ outside TileLang's target pipeline.

Before proposing a TileLang change:

1. inspect the public language, existing schedules, lowering passes, target facts
   and execution adapters;
2. confirm the desired mechanism cannot already be expressed idiomatically;
3. identify the exact category of deficiency: correctness bug, missing portable
   contract, missing backend implementation or poor lowering;
4. add only the smallest generic TileLang correction, while Magnitensor retains
   algorithm and selection policy.

Difficulty expressing a schedule does not prove a missing primitive. First verify
the meaning of TileLang's existing tiles, fragments, layouts, macros, regions and
program construction. A backend-specific workaround is frequently evidence that
the schedule is using those abstractions incorrectly.

### Compose Python IR as Python

TileLang's language is already a Python construction interface. Magnitensor
composes authored schedules and macros while TileLang's eager builder is active;
it does not generate source strings, synthesize Python AST, invoke `exec` or splice
private TIR.

An emitter contributes work to a compiler-owned program under construction. It
does not prematurely finalize a `PrimFunc`. This distinction permits independent
region selection while retaining whole-function storage and submission planning.

`T.Kernel` denotes one device execution region. A `PrimFunc` may contain several
ordered `T.Kernel` regions and represents one portable TileLang program. TileLang's
runtime must expose that program through one native host entrypoint. Magnitensor
chooses which work shares the program; TileLang owns correct lowering, ordered
native launch realization and generic argument binding.

### Specialize through capabilities

Portable does not mean one schedule. Magnitensor selects mechanism-named schedule
families from behavioral capabilities: subgroup and matrix geometry, memory
capacity, movement, synchronization, atomics, dtype support, alignment and launch
facilities. Neither model code nor portable kernels branch on `metal`, `cuda`, a
vendor name or a machine identity.

If a fact legitimately changes the optimal schedule, it belongs in TileLang's
target capability contract. If a portable operation exists but produces poor
native code, repair its backend lowering. Do not duplicate target databases or
create backend-specific side channels in Magnitensor.

### Respect physical regions and logical matrix axes

A `BufferRegion` describes physical buffer dimensions. Transposition changes how
those dimensions participate as logical M, N and K; it does not change which
physical offset belongs to a row or column. Carry physical row and column offsets
until the transpose mapping is applied. Naming a physical offset after a logical
axis too early can silently swap addresses in transposed GEMM.

Likewise, views and layouts must express actual storage. Do not encode one vendor's
bank width or matrix atom through arbitrary padding, dtype choices or fabricated
aliases in an otherwise portable schedule. A reusable layout need belongs in the
portable layout contract; backend bank realization belongs in TileLang lowering.

### Separate static geometry from runtime validity

Matrix mechanisms require static physical tile shapes even when useful work has a
dynamic tail. Represent both facts: a statically optimized tile and a runtime-valid
extent. The backend predicates whole instruction rows or safe boundary accesses;
portable kernels do not round counts using a vendor-specific instruction atom.

The full-tile path remains structurally free of dynamic predicates. Predication is
paid only by boundary tiles or partially occupied work. The operation contract
states whether invalid suffix values are preserved, zeroed or unspecified.

### Preserve types through native lowering

Generated C-like languages may apply promotions that are absent from TIR. Bit
reinterpretation must preserve the declared source width before native `as_type`
or equivalent operations. Atomic lowering must preserve dtype, address space,
memory semantics and whether the expression returns the previous value. These are
TileLang backend correctness obligations, not reasons for a kernel to emit native
Metal or CUDA syntax.

### Keep the hot path native

Static weights and stable resources are bound once against the compiled ABI. The
warm call binds only dynamic operands. Resolving functions, creating pipeline
state, traversing graph regions, allocating planned scratch and issuing a Python
call per device kernel are absent from repeated execution.

Native multi-launch and partial binding are execution-adapter capabilities. Their
absence is an explicit TileLang runtime limitation and may require splitting or
rejecting a candidate; it is never silently replaced by a Python loop.

## Select and tune coherently

Candidate applicability proves legality; measurements choose among legal candidates.
Selection considers the entire connected region, including representation changes,
materializations, workspace and downstream compatibility. An individually faster
kernel loses when it forces a more expensive enclosing graph.

Tune only parameters that remain real degrees of freedom after algorithm, dataflow,
layout and representation are sound. Search offline over meaningful schedule
families and geometry, then retain results as deterministic candidate data. Production
startup does not discover basic schedules, and a benchmark identity never enters
compiler policy.

Specialization keys contain graph structure and declared static facts. Continuously
changing token counts, expert occupancy and positions remain runtime operands or
bounded geometry classes; they do not cause unbounded recompilation.

## Verify the mechanism

Validation proceeds from contract to mechanism to enclosing outcome:

1. independently validate values, precision boundaries, tails and state effects;
2. inspect lowered IR and generated source for the mechanism claimed by the cost
   model;
3. measure the complete selected region including materializations and dispatch;
4. measure the production prefill, TTFT, decode and long-context paths that the
   change is intended to improve;
5. test other applicable shapes, dtypes, layouts and capability classes.

Timing alone does not prove reuse, fusion, occupancy, eliminated spills or matrix
instruction selection. When generated execution contradicts the explanation,
revise the explanation and design rather than adding compensating patches.

Source-only tests establish portable construction and lowering on machines without
the target device. Target execution tests establish backend correctness. Magnitensor
tests establish capability-based selection and composition. Whole-model evidence
establishes acceptance. Each belongs to the layer whose guarantee it validates.

Remove scaffolding and uncertain contributors after the mechanism wins. The retained
result should explain why it wins, where it applies, what it costs and what remains
limiting. Measurements and rejected experiments belong in `runs/` and `results/`;
this design retains the transferable decision rules.
