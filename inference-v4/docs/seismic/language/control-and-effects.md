# Control and effects

Control determines which operations execute, in what order, and which values become available. Effects determine what an execution may observe or change. They are part of the computation, even when an operation returns no value.

## Iteration and branching

`for` iterates in ascending order over the actual bounded range. Loop carries are explicit state transitions: the initial carry enters the first iteration, each iteration constructs the complete next carry, and the final carry becomes the loop result. An empty range returns the initial carry and performs no body effects.

`parallel for` expresses independent logical work under the checked access rules. The compiler chooses its physical assignment. Parallel syntax does not permit overlapping ordinary writes or make every nested value shared. Iteration-local storage remains private; shared writes require exclusive participant ownership or a suitable atomic operation.

A branch executes only its chosen arm. Its join constructs the resulting values, initialization and memory versions. A branch-local result is available outside only through that join. The compiler cannot evaluate a partial expression in an inactive arm merely to simplify a guard or reservation.

Reduction semantics include the contribution domain, empty-domain identity, association, accumulation type and publication. An ordered reference reduction preserves its ordered recipe. An explicitly unordered reduction admits its specified outcome relation; it does not grant unrelated arithmetic transformations.

## Effect ownership

| Effect | Construction must retain |
| --- | --- |
| Read | Correct place, initialized region, captured version and any observable access behavior. |
| Write/store | Actual destination region, write permission, rounding and memory-version transition. |
| Atomic | Combined access and arithmetic semantics, participants, scope, order and permitted outcomes. |
| Check | Actual condition, active control path, failure behavior and ordering relative to effects. |
| Publication | The initialized result and the completion point after which its consumer can use it. |
| Synchronization | Participating operations, visibility, completion and progress requirements. |

Pure recomputation is permitted only when it reproduces the captured computation and versions. It cannot duplicate atomics, skip checks or repeat externally observable effects. Calls compose effects from their selected bodies rather than hiding them behind a function boundary.

## Availability and progress

There is a difference between a value existing in a program and being available at a particular point. An invocation parameter is available before execution. A loop index is available inside its loop. A device-produced scalar is available only after its producer and required completion relation.

Physical construction must enforce these distinctions through scoped operands and region results. Copying a parameter into a device scalar slot does not change its semantic origin. Conversely, a bound on a device result does not make the result itself available early.

Parallel and asynchronous programs must preserve the source's progress contract. Correct output arithmetic is insufficient if a barrier can deadlock or a queue protocol assumes blocks run in a favorable order. Typed target participation and completion operations must compose into a legal protocol.

The [construction contract](../compiler/construction.md) explains how these rules become physical values and regions. [Execution resources](../execution/resources-and-lifetimes.md) covers ownership after submission.

