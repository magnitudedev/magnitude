# TileLang IR portability

**Portable IR preserves one operation's values, numerical boundaries and effects
while exposing bounded schedule choices that TileLang can qualify, select and
realize efficiently for each target.**

This is a development guide for authoring and evolving portable kernels.
[Portable kernels](../kernels.md) defines the architectural boundary,
[IR authoring rules](ir-rules.md) covers precise language construction, and
[kernel optimization](kernel-optimization.md) governs optimization scope and
performance evidence.

## Principles

1. **Define semantics before decomposition.** Required values, valid inputs,
   rounding, masks, aliasing and state transitions belong to the operation.
   Tiling, storage, traversal and communication belong to its schedules.
2. **Preserve choices where machines have different costs.** A physical schedule
   that compiles everywhere can still discard the best realization for each
   target. Keep meaningful alternatives under one semantic contract.
3. **Expose structure to TileLang.** Express contractions, reductions, regions,
   ownership and dependencies through the public language. TileLang owns native
   instructions, fragment realization and backend lowering.
4. **Separate resource feasibility, legality and speed.** Resource facts can rule
   out an impossible candidate. Compilation establishes whether TileLang can
   lower it; independent validation establishes correctness; measurement ranks
   qualified candidates. None of these establishes the others.
5. **Treat ownership and lifetime as part of the schedule.** A smaller expression
   can require more exchange or retain more live state. Optimize arithmetic,
   preparation, storage and communication together.
6. **Qualify the boundary that will execute.** A selected schedule is valid for
   its complete workload and composition context. Standalone performance does
   not establish fused performance or enclosing application performance.
7. **Keep calibration explicit and reuse exact.** Search outside normal startup
   and invocation. Persist qualified choices with their complete identity and
   reconstruct static IR before native composition.
8. **Use failures to improve the abstraction.** A failing candidate is evidence
   about its construction, contract or lowering. Diagnose that boundary before
   adding a restriction to every target or cloning an algorithm by backend.

## Concepts and ownership

| Concept | Meaning | Owner |
|---|---|---|
| Semantic operation | Values, effects, numerical obligations and permitted transformations | Ops |
| Workload class | A bounded shape or execution regime with materially different scheduling needs | Ops |
| Semantic template | One authored computation parameterized by static scheduling choices | Ops, through public TileLang |
| Schedule family | A bounded set of alternative decompositions implementing that contract | Ops |
| Candidate | One fully static schedule, including all coupled choices | Ops construction |
| Candidate legality | Whether that concrete program can be lowered for the target | TileLang |
| Selected schedule | A qualified configuration for a complete target and workload identity | TileLang selects; Ops persists and reloads |
| Native realization | Instruction selection, inferred layouts, register representation and ordered execution | TileLang |

Ops supplies the reference obligations and calibration inputs. TileLang's tuning
path validates and measures candidates; Ops persists the returned configuration
with its workload and validation identity, then reconstructs the selected schedule
through its ordinary compilation path. Ops does not own a second autotuner,
instruction catalogue or backend implementation registry.

Target facts describe resources and behavior: subgroup width, memory capacity,
thread limits, supported types and execution facilities. Use them to construct
and prune meaningful candidates. They are not a lookup table for the optimal
schedule. Missing facts remain unknown rather than becoming guessed limits.

Backend, vendor, device and native instruction names must not decide numerical
IR construction. Physical device and target identity belong in selection records
and compiler routing; they do not belong in kernel branches. Generated source is
inspection evidence, never input to a heuristic deciding which IR to author.

## Designing a schedule family

### Fix the contract first

Write down:

- input domain, logical extents, representations and exceptional-value behavior;
- accumulation precision, observable rounding points and numerical tolerance;
- permitted reduction-order changes, reassociation and recomputation;
- masks, empty work, tails and dynamic occupancy;
- resource versions, aliases, mutations and publication order.

Every candidate must satisfy the same obligations. Fusion can remove storage
without removing an observable dtype boundary. Algebraic equivalence over real
numbers does not authorize a different finite-precision computation.

An optimization assumption needs a producer or caller that establishes it. A
common fixture pattern is not a contract: one sequence does not imply equal
visibility starts, ordered counts, full tiles or nonempty rows.

### Choose alternatives with different cost structures

Start with a small family whose members change reuse, live state, parallelism or
communication. Name strategies by what they do: sliced preparation, shared
operand, direct fragment chain, partitioned reduction. Keep strategy changes
explicit rather than hiding them in magic tile dimensions.

Useful axes include output and reduction tiles, collective size, operand
preparation, ownership transfer, buffering and partitioning. Bound their
combinations deliberately. Add a candidate to test a concrete cost hypothesis;
remove redundant choices when qualification shows no distinct useful regime.

Workload classes follow semantic geometry: narrow rows, ordinary matrix tiles,
grouped occupancy and tails, or history footprints that change partitioning.
Continuously changing token positions and counts remain runtime data unless a
small, justified specialization class changes the execution structure.

Prune against the complete resource footprint, including simultaneous operands,
coefficients, exchange buffers and workspace. A fused candidate must fit every
stage and every overlapping lifetime. Do not select geometry and then overwrite
its staging or partition parameters afterward; that constructs a different,
unqualified candidate.

## Portable patterns

Each pattern is an alternative with preconditions and costs. None is a universal
preferred spelling or a promise of identical performance across targets.

### 1. Preserve compatible fragments; exchange when ownership changes

A fragment is a distributed value. Equal logical shapes or indices do not imply
equal lane ownership. A parallel elementwise loop is valid only when one inferred
distribution satisfies every participating access; it is not an implicit shuffle.

| Pattern | Preconditions | Trade-off |
|---|---|---|
| Direct fragment chain | TileLang establishes compatible producer and consumer ownership | Avoids publication and reload, but couples layouts and live ranges |
| Shared exchange | Producer publishes, required participants synchronize, consumer reads a valid shared region | Decouples layouts at the cost of storage, traffic and synchronization |
| Global boundary | Values must cross independently ordered device regions | Adds materialization and dispatch but can release local resource pressure |

Retain both direct and explicit-exchange candidates when they serve different
legal realizations. A layout conflict in one candidate does not justify forcing
shared exchange into every schedule. Successful execution on another target does
not justify removing a required exchange.

The exchange has one owner. If a caller already supplies a shared region in the
required layout, consume that region rather than copying it into another shared
buffer. Establish both publication-before-read and read-before-reuse ordering;
removing a duplicate copy does not remove those dependencies.

### 2. Vary operand preparation while preserving numerical boundaries

Quantized contractions separate storage decoding, operand preparation,
multiplication, accumulation and output publication. These are distinct precision
boundaries even when they appear inside one expression.

| Strategy | Useful mechanism | Cost to examine |
|---|---|---|
| Whole reduction tile | Prepare once and reuse across a contraction | Large live operands and register pressure |
| Static K slices | Prepare and consume bounded slices while retaining the accumulator | More contractions and preparation steps |
| Shared decoded operand | Decode into reusable collective storage | Shared capacity, exchange and barriers |
| Factored affine contraction | Contract exact codes, then apply group coefficients | Additional row sums, coefficient work and changed reduction order |

For coefficients constant within a quantization group, a useful algebraic
candidate is:

```text
sum_k x[k] * (scale * code[k] + bias)
    = scale * sum_k x[k] * code[k] + bias * sum_k x[k]
```

This identity motivates a schedule; it does not prove numerical qualification.
Codes must be exactly representable in the chosen operand type, coefficient
grouping must match the representation, and the contract must permit the changed
arithmetic order. Preserve required coefficient precision and publication casts.
Do not silently round decoded weights to the activation dtype to enable a matrix
mechanism. A declared FP32 accumulator alone does not prove multiplication
precision in the lowered program.

Validate nonexact coefficients, cancellation, supported ranges and all tail axes.
When a reduction feeds a differently distributed contraction, give its result a
valid ownership bridge. A row sum indexed like a matrix row is not automatically
owned by the thread that consumes it.

### 3. Keep physical geometry separate from logical validity

A candidate has static physical tiles. Logical dimensions and runtime occupancy
state which values are meaningful within those tiles. Keep narrow physical tiles
in the family even if another target needs larger ones; TileLang decides their
legality individually.

A valid prefix such as `valid_m` describes active M rows inside an otherwise legal
physical contraction. It does not legalize an unsupported tile, select its size,
mask arbitrary rows, handle K/N tails or guarantee a speedup. Its inactive suffix
is not a value that consumers may assume is zero.

Loads, initialization, collective participation and publication must agree with
the validity domain. Mask data without excluding participants required by a
collective. Initialize every consumed element with the operation's appropriate
neutral value, including partially occupied expert groups and reduction tails.

Do not derive valid work from a convenient physical endpoint. A padded last row
cannot suppress earlier valid rows. Entirely empty work must have defined output
and state behavior without reading invalid addresses or dividing by zero.

### 4. Derive traversal and fast paths from different proofs

Attention illustrates a general distinction: the region worth staging can be
larger than the region every consumer may use without a mask.

For valid per-row intervals `[start, end)`:

- a bounding interval covering the union of nonempty rows defines traversal;
- each row retains its own membership mask, including gaps inside that bound;
- the intersection across valid rows bounds a common unmasked fast path;
- empty rows do not enlarge traversal and still constrain the common fast path;
- physical padding is excluded from logical work and receives safe masking.

```text
row A reads [2, 6)
row B reads [4, 8)

stage within [2, 8)
only [4, 6) is common to both rows
mask the remaining staged positions independently for each row
```

A fast path requires proof for the entire physical chunk it consumes. Apply this
reasoning to histories, current rows and partition tails independently. Deriving
traversal from the first row's start or the last row's count requires an explicit
stronger contract; it cannot be inferred from typical input values.

### 5. Select coupled schedules at the composed boundary

Fusion changes resource lifetimes, layout constraints, staging and intermediate
publication. Child schedules that win separately need not win together.

Declare the coupled choices as one bounded candidate for the complete operation.
Qualify its preparation, contractions, transfers, required casts, state effects
and outputs together. If changing one child changes another child's legal family,
represent that dependency in the joint family rather than silently substituting
a different child during reconstruction.

Reconstruction must preserve semantic ports, alias relationships, state effects
and required publication boundaries. Scratch and internal geometry may vary as
part of the qualified candidate, within the declared resource budget.

Select before finalizing private functions into the native module. Compose
statically authored Python templates through TileLang's public builder; do not
recover or splice finalized compiler IR, generate source strings, or introduce
per-kernel Python dispatch as a substitute for native composition. Revalidate the
selected schedule in the ordinary composed execution path.

### 6. Change decomposition when the execution model changes

A GPU collective and a CPU blocked/vector traversal can implement the same
semantic operation with different schedules. Extending a family to another
execution model may require a genuinely different work decomposition. Renaming
the target of a fixed collective schedule is insufficient.

Use public contractions, reductions and storage contracts in each strategy;
TileLang still chooses native vector and matrix instructions. An unavailable fast
instruction can admit a correct generic lowering. Measure that lowering before
deciding it meets the intended performance contract.

Before declaring an operation unsupported, distinguish an omitted portable
strategy from a missing public primitive or a missing backend implementation.
Each supported target needs a qualified schedule for the declared domain or a
precise, evidenced capability gap. Untested targets remain unqualified.

## Calibration and reuse

```text
semantic template + workload + bounded candidate family + reference fixtures
                                  ↓
                  TileLang construction and compilation
                     reject illegal candidates individually
                                  ↓
                  independent numerical and effect checks
                     exclude unqualified candidates
                                  ↓
                   measure the complete candidate boundary
                                  ↓
                    persist the qualified configuration
                                  ↓
             reconstruct static IR → compose native program → execute
```

Calibration is an explicit development action at the work plan's validation gate.
Normal startup and warm execution do not search, benchmark or retune. A diagnostic
choice is useful for investigation but does not become a qualified selection
merely because it executes successfully.

Persist configuration under an identity covering:

- semantic contract, template construction and relevant transitive dependencies;
- the complete candidate family and coupled composition context;
- operation geometry, workload class, dtypes and storage representations;
- numerical mode and reference/fixture/acceptance-policy identity;
- compiler provenance, complete target, physical device and runtime provenance;
- resource facts that affect construction or legality.

Changing any of these invalidates the old qualification. A partial identity,
corrupt record or nearby workload is a miss. Required coverage must produce an
explicit calibration request; it cannot silently choose an unmeasured substitute.
A separately supported conservative default needs its own declared correctness
and coverage contract and must not be reported as the calibrated winner.

Prove reuse by reconstructing and executing through the production path with
tuning disabled. Configuration persistence and compiled executable caching are
different responsibilities; neither establishes the correctness of the other.

## Development and review practice

Start with the enclosing cost and the semantic contract. Identify which
restriction prevents a useful portable schedule, then change that abstraction
coherently. Parameter sweeps are useful after dataflow and ownership can plausibly
meet the objective; they cannot repair an unsuitable decomposition.

At the declared validation gate, establish these claims separately:

| Claim | Required check |
|---|---|
| Correct semantics | Independent outputs, precision boundaries, aliases and complete state effects |
| Complete domain | Empty/unit work, full tiles, every tail axis, arbitrary permitted masks and occupancy |
| Legal ownership | Inferred layouts or explicit exchange, initialization, participation and reuse ordering |
| Intended lowering | Actual contractions, reductions, guards, copies and synchronization; investigate unexplained scalarization or spills |
| Useful performance | Complete operation and enclosing production path, including preparation and publication |
| Safe reuse | Exact identity, ordinary reconstruction and execution with tuning disabled |
| Cross-target coverage | Independent qualification on each applicable target and workload regime |

Use repeated comparable measurements and a predeclared noise rule. An isolated
kernel win cannot establish application recovery, and a difference inside noise
cannot establish an improvement. Keep raw measurements, compiler provenance,
unsupported cases and investigation notes in run records, outside this guide.

### Classify failures before changing the design

| Observation | Development response |
|---|---|
| One candidate fails to compile | Retain the diagnostic and reject that candidate on that target; preserve other targets' choices |
| Every candidate fails | Recheck semantics, construction and family completeness before attributing a compiler gap |
| A candidate violates precision or effects | Repair or exclude it; do not loosen the shared contract to make it win |
| Direct fragments have incompatible ownership | Test the public layout/copy contracts and an explicit-exchange candidate |
| Correct IR lowers inefficiently | Inspect dataflow, lifetime and generated execution before adding more parameters |
| A standalone winner fails after composition | Qualify the actual coupled boundary and correct its identity |
| One target improves and another regresses | Keep useful alternatives and qualify selection independently |
| A fixture reveals an unstated assumption | Revisit the public contract before restricting the fixture or masking the failure |

A TileLang change requires a demonstrated defect or missing portable contract,
a minimal reproduction, and investigation of existing public alternatives.
Keep the correction general and at the owning layer. Scheduling inconvenience,
a speculative speedup or one failed candidate does not justify a compiler fork
change. Never bypass TileLang with a vendor runtime, library or native call.

A review should be able to explain **what remains invariant, which choices vary,
why each choice exists, how ownership stays valid, and where each claim was
qualified**. Turn discovered contract and lowering failures into focused
regression checks. Remove obsolete restrictions and experimental scaffolding so
the retained family stays understandable and admits further improvements.
