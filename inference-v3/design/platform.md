# Execution ownership

**One owner per device holds storage, order and completion. Nothing above it can
release a resource, reorder work or learn a backend; nothing below it can allocate
or decide.** Every other component composes with the device through this one.

## Lifetime

```text
allocate ──► tensor ──► views (each a claim on the backing)
                          │
              prepare(kernel, operands) ── takes the operands' claims
                          │
                    submit ──► ticket ── holds every claim
                          │
              completion proven (poll or wait) ──► claims released ──► backing freed
```

| Rule | Reason |
|---|---|
| A view is a claim; backing lives while any claim does | A consumer never has to know who else holds the tensor; the last claim frees it |
| Prepared work takes its operands' claims | Unwinding a failed preparation releases exactly what it took, in reverse |
| A ticket keeps claims until completion is proven | Discarding an output releases nothing early; the device may still be writing it |
| Reclamation happens only through proven completion | Memory pressure is relieved by finishing work, never by guessing the device is done |
| A tensor with submitted consumers has no mutation | An upload is private until it returns; there is no in-place write into live data |
| One thread, one stream, program order | Ordering is never inferred; two commands run in the order submitted, and independence is declared explicitly |

Declared independence is a permission, not a demand: a region whose children do
not conflict may be overlapped by a backend that can, and serial execution is
always a valid realization. Completion of a region joins every child.

## Budget

The budget is an admission limit on charged bytes, not a claim about free
hardware memory. Aliased views count once. Native allocation may still fail
below the budget, and a refused allocation reports the required and available
bytes so the layer above can negotiate: finish work, shrink, evict, or wait.
The owner never negotiates on its own; it has no request to negotiate for.

## Captured execution

A decode step is several hundred commands. Binding them one at a time on the host
would cost more than the device work, so a sequence is captured once and replayed:

```text
capture:   static operands (weights, scratch)     viewed once, retained by the capture
           changing operands (inputs, state)      described by layout: allocation group + offset
invoke:    same layout ──► rebind changing operands only ──► one launch sequence
```

The layout preserves alias identity, so two views of one allocation stay aliased
after rebinding. A different layout is a different capture. The captured sequence
retains its static resources for as long as any invocation is outstanding.

## Specialization

Code is shared by structure, never by identity of what it touches:

| Shared | Not shared |
|---|---|
| Two programs that are structurally equal use one executable | Two weights of equal shape use one executable and two allocations |
| Equal compile-time arguments hit one cache entry | Operands, state and scratch are never part of the key |

Compile-time arguments are the whole specialization: geometry, dtypes, precision,
capability and representation parameters. Anything that changes per invocation
is an operand. Specialization count is a cost, so geometry that varies widely
(a history length) is bounded into classes before it reaches a kernel.

## The driver

The driver is one implementation for every backend. What it may vary per backend
is bounded:

| Per backend, in the driver | Never in the driver |
|---|---|
| Which device to open; which completion event to record; how to drain | A compiler pass, transform or pipeline choice |
| How a byte upload is staged | An execution adapter selection per target |
| How the endpoint is checked against the discovered device | A compile flag or annotation |

Compilation is one call with a target. Target policy lives in the compiler fork.

## Capability

The driver reports one capability when the owner opens: lanes that share a
reduction, threads per group, shared memory, and whether matrix hardware exists.
It is the only backend fact a kernel or a candidate table may read. It is taken
from a compiled pipeline rather than a target table, because target tables carry
placeholder limits; the pipeline is what will actually run. On the host the
capability is one lane and one thread, and a grid is a loop.

## Host

Measurement is exclusive: correctness tests and benchmarks take one machine-wide
lock, so a timing never shares the device with a test. The worker that owns an
execution owner runs on one thread, awakened by control jobs or by native
completion; completion wakes have reserved delivery, so a saturated control queue
can neither block completion nor deadlock shutdown. Idle service has no polling
timer.
