# Seismic execution

**This document defines what an authored structure means when it executes, which
adjacent work may share a realization, and how a witness becomes one concrete
execution.** Source rules are in [Language](language.md); target rules are in
[Backends](backends.md).

## Reference semantics

The reference interpreter executes the structured IR directly and is the semantic
oracle. It accepts any legal partition from its caller: a width per static binder.
A conforming program produces the same results under every partition, up to the
numerical permission of its admitted functions. Every backend execution must agree
with the interpreter, including rounding at each operation's dtype, publication
rounding, accumulation dtype, FMA, and reduction order.

Reduction order has one authored exception. `reduce(t, axis, sum|max|min,
unordered=true)`, legal only inside an `admit fn`, permits a backend to reassociate
that reduction; the interpreter still accumulates in ascending order. The execution IR
carries the permission on the reduction (`ordered = not unordered`), and a backend may
use a reassociating algorithm only there. Two library contracts package the
permission: `sum_any_order` (an ordered and a reassociating body) and
`matmul_any_order` (a contraction whose association, and the distribution of a packed
operand's affine decode over the sum, are unspecified). A caller opts in by calling
them; an overload family whose bodies differ in this way is itself declared `admit`
(`linear`). Agreement with the interpreter is then up to F32 summation rounding on
those paths and exact everywhere else.

## Regions

| Form | Meaning |
| --- | --- |
| `parallel` | Independent visits, one slice per binder per visit. A body cannot mutate enclosing state; it may publish to provably disjoint views and yield a result. |
| `ordered` | Visits in ascending lexicographic order, last binder fastest. Each visit completes before the next begins. A body may update enclosing `var` state. |
| `pipeline` | Ordered visits whose body is one linear stage chain. Each state object has exactly one updating stage. Preparation may run ahead only across stable reads. |
| `merge` clause | Canonical near-equal contiguous partition of the axis; adjacent partials combine level by level, an odd value is forwarded. Empty domain gives the identity; one part gives its partial. |
| Refinement | A region over an enclosing slice partitions that slice. It owns a new site. |
| Rebinding | A region over a region result revisits the producer's pieces. It creates no site and reruns nothing. |

One static binder is one site. Every dynamic instance of the binder uses the same
selected value. A binder applied to several tensors is one joint traversal.

Lexical position fixes production: a binding before an inner region is produced once
per enclosing visit; inside, once per inner visit. Execution never moves,
duplicates, or merges producers.

## Stages and completion

Consecutive `stage` statements are one linear chain; a stage receives the previous
stage's yield positionally. Outside a pipeline, a stage and all work it started
complete before the next stage starts, at the scope of the enclosing owner: the
invocation, one parallel visit, or one ordered visit. A region completes before the
statement after it. Region exit discharges every completion obligation inside it.

Inside a pipeline, stages of one visit run in chain order and carried-state updates
of visit `i` precede those of visit `i+1`.

A helper call is never a launch, materialization, or completion boundary by itself.

## Region results

A region used as an expression yields exactly one value of one schema per visit.
The result keeps the producer's partition: consumers revisit the same pieces and
select the member of the current visit. Results are immutable, may nest, and may
pass through stage ports and helper calls. They cannot be counted, indexed by
number, flattened, or escape the compiled composition. Their cardinality is never
semantic data.

Storage of a result is derived, never declared: one backing per yielded member, with
one leading piece axis per binder of every enclosing producer, sized by the selected
piece counts, alive until the last consumer.

## Partial values

A value yielded from a region, or reduced over a structural axis, depends on the
partition until it is combined. Outside an admitted function it may only be
forwarded, stored in results, combined by a `merge` clause, accumulated into state
by `+`, `max`, or `min` within a traversal of the same result, or passed to an
admitted function. Admission is a trust boundary, not a proof. Ordered
accumulation into carried state across windows preserves the element order of the
whole traversal and needs no admission.

## Execution units

An execution unit is a static portion of one authored block with a prescribed
backend execution. Units partition the block's statements in authored order.

| Unit kind | Statement |
| --- | --- |
| Elementwise | Tile-valued binding or tile state update computed pointwise over identical axes, with scalar broadcast |
| Local | Reduction, scalar work, loop, branch, helper call, or any other non-elementwise computation |
| Call | Call occurrence at a lowering boundary |
| Publish | `publish` |
| Region | Nested region, as statement or bound expression |
| Stage | One stage of a chain |

- A single-consumer pure tile-valued `let` adjacent to its consumer's unit joins that
  unit. One that is not adjacent stays its own unit. A multi-consumer `let` is one
  producer in every grouping.
- Operators within one statement never split.
- A stage outside a pipeline carries a completion after it. A fused interval may
  cross a completion only through a realization that preserves it.
- A block with fewer than two units has no sequence and no grouping decision.
- Bodies of loops, branches, stages, and regions have their own sequences. Dynamic
  visits never create units.

## Contiguous fusion

A fusion candidate is a contiguous interval of one sequence with one prescribed
realization. The backend lists every legal interval, including the singletons that
are separate execution. The solver picks an exact cover. Nothing else fuses.

An interval is legal only when its realization establishes all of:

1. **Order.** Units stay in authored order. A joint traversal interleaves them per
   coordinate only when every dependence, state update, failure, and completion is
   preserved.
2. **Correspondence.** The units iterate provably corresponding coordinates within
   the already granted owner. Equal extents or equal selected widths are not proof.
   Required width equalities are exported as constraints on the sites.
3. **Production and numerics.** Every producer keeps its occurrence, multiplicity,
   snapshot semantics, and conversions. No reassociation, no common-producer
   discovery.
4. **Interfaces.** Values leaving the interval keep their representation and
   lifetime. Only compiler-owned intermediates may disappear. A `publish` is never
   removed.
5. **Completion and participation.** Every port, ordered visit, and collective
   participation rule still holds.
6. **Resources.** Hard capacity limits hold for the combined group.

Legality admits a candidate to the solver. It says nothing about profit. A legal
group with a high estimate stays a candidate.

## Instantiation

Instantiation is a deterministic function of the program, the family, and the
witness. It returns one execution IR or a diagnostic. It never chooses and never
repairs.

| Subject | Rule |
| --- | --- |
| Entry | Parameters become the invocation ABI in declaration order. Tensors and views bind buffers; scalars bind scalar arguments; a bounded index binds a checked runtime scalar. |
| Aliasing | Every written tensor parameter must be disjoint from every other tensor parameter, except that a declared `alias` pair may coincide exactly. The requirement travels with the execution and is checked at invocation. |
| Calls | The selected candidate's body is inlined. Views stay references. |
| Slice | Piece `p` of a binder with lower bound `lo` and width `w` is `[lo + p·w, lo + (p+1)·w)`. |
| Root `parallel` region of the entry | One launch whose work items are the pieces. |
| Every other region | Ordered loops over pieces inside its owner, first binder outermost. |
| Root stages and invocation-scope statements | Consecutive root statements, executed in order with completion between them. |
| Tile computation | One element loop per unit; one loop for a selected elementwise interval. A dependency between its units that is not elementwise is a diagnostic. |
| Selected interval of root `parallel` regions | One launch over the shared pieces; differing binder geometry under the selected widths is a diagnostic. |
| `merge` | The canonical adjacent-pair recurrence over the selected part count. |
| Region result | Local tiles with leading piece axes. |
| Reduction | The execution IR's reduction carries its numerical contract: ordered unless the source said `unordered=true`. |
| Runtime-bounded range | Bounds clamped to the axis; the extent is a runtime value. |
| Data-dependent point index | Runtime bounds check with defined failure. |

The execution IR is verified before it is returned. The backend then realizes it by
its own fixed rules ([Backends](backends.md)).

## Current limitations

- **Divisor widths only.** A width must divide its static extent. Instantiation
  rejects any other width. The language contract for tails stands (a selected body
  must be correct for every valid extent up to its capacity), but no tail piece is
  generated yet.
- **Runtime extents.** A domain with a runtime extent is instantiated only at width
  one. Runtime-length history is authored as a semantic range, not a slice.
- **Region results stay inside one launch.** A result produced in one launch and
  consumed in another has no mapping, nor does a root traversal of a result.
- **Synchronous pipeline only.** One visit prepares, then consumes, on the same
  participant. Ring depth is fixed at one and is not a site.
- **Invocation-scope loops** cannot contain regions or calls. A root region result
  is supported only when a `merge` clause reduces it within its launch.
- **Intervals do not span a helper call.** See [Compiler](compiler.md#source-stability).
- **Intervals do not span a lowering-boundary call.** Every call statement is a `Call`
  unit and the family reports when a callee's root block is solely parallel regions,
  but instantiation realizes a selected interval of root `parallel` *region* units
  only. A composition entry pays one launch per callee region.
- **Runtime-extent tiles are shared by capacity.** Element loops over a tile with a runtime
  extent divide its capacity among the lanes (a short extent leaves some lanes with less
  work), and matrix lowerings whose predicates name a runtime extent are inapplicable.
  Attention over runtime-length history writes its score tile cooperatively into
  threadgroup memory with the ordered `matmul` chain and runs its value product
  history-major; it is not windowed.
- `lanes` loops and `atomic(max|min)` have no structured form; `atomic(add)` is
  checked but cannot be instantiated.

Each limitation is reported as its own diagnosed outcome. None selects a different
execution silently.

## Acceptance

- Reference fixtures agree under at least two different partitions.
- For every supported kernel, the selected Metal execution agrees with the
  interpreter.
- Instantiating the same witness twice yields the same execution IR.
