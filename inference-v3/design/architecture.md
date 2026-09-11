# Architecture

**A component is a boundary that contains one kind of complexity behind a
contract. The engine is a composition of such components, and no complexity
crosses a boundary.** Sophistication inside a boundary is what makes the engine
efficient; containment is what makes that sophistication survivable.

## Components

Each component contains a concern that would otherwise spread through the system.
The contract is what it exposes; the leak is what it must never expose, because
the moment it does, every neighbor grows a branch for it.

| Component | Contains | Exposes | Must never expose |
|---|---|---|---|
| Execution owner | Device storage, ordering, completion, the driver, the compiler call | Tensors and leases, prepared work, tickets, a capability | Backend names, compiler configuration, runtime objects |
| Kernel | One numerical schedule and its physical arrangement | A factory over shape, precision, capability, representation | Target functions, dialects, container knowledge |
| Operation | Contract, candidate selection, scratch declaration | Prepared commands for logical tensors; a recorded plan | Which schedule ran, how it partitions, backing |
| Weight residency | Containers, encodings, layouts, conversion | Resident weights in a representation | Container identity, file layout |
| State | Pages, extents, claims, growth, reclamation | Visible runs and stable read windows | Physical addresses, page tables, slab membership |
| Model executor | Architecture equations, state transitions, capture | Sequences, packed batches, independent advances | Kernels, storage layout, request identity |
| Generation | One request's history, output credit, sampling, recovery | Work proposals, acceptance, snapshots | Numerical state, batch membership |
| Service | Admission, phase choice, capacity negotiation | Step, submission, publication | Model internals, transport |
| Serving | Transport, rendering, parsing, framing | Tokens and options in, tokens and snapshots out | Live tokenizers, parsers or device objects across the worker |

## Composition

```text
Serving
└── Service
    ├── Generation × requests
    │   └── Sampling
    └── Model executor
        ├── Description (weight roles) ── Weight residency ── Container format
        ├── State (KV pool, recurrent banks)
        └── Program: Operations
            ├── Projection, Attention, Recurrence, Norm, …  each with a candidate table
            │   └── Kernels (schedules)  ── compiler fork ── device
            └── Arena (declared scratch)
Execution owner: shared by every component that touches the device
Composition: the tree above as data, buildable and digestible without a runtime
```

Components receive their dependencies; none looks up the engine that contains it.
A parent selects implementations of its children by contract, so a child can be
replaced, compared or measured without the parent changing.

## What a boundary is

A boundary is a contract, not an execution barrier. Adjacent components may fuse:
a whole forward is captured into one launch sequence, a gated projection may run
as one kernel, attention's scratch is shared by every layer. Fusion is decided
inside the component that owns the work, at plan time, and never requires a
neighbor to know. Diagnostic access to a boundary must not impose its cost on
normal execution.

| Rule | Reason |
|---|---|
| A decision is made where its inputs live | Kernel selection reads representation, precision, capability and shape, so it lives in the operation; a scheduler that knew kernels would have to know all four |
| A fact crosses a boundary once, as a value | Precision is one value carried everywhere; a flag threaded through fifty call sites is the same fact leaking fifty times |
| Names describe the thing, not its provenance | A schedule is named by how it computes; a name that says where it runs has already leaked the backend |
| Proxies are forbidden as conditions | "Is Metal" stands in for "has 32-lane subgroups"; the proxy admits and excludes wrongly the day a second backend has them |
| Resources are owned until completion is proven | A tensor, an extent or a captured sequence stays alive while any submitted work depends on it; early release by an upper layer is a leak of the device's timing into that layer |
| Unsupported compositions fail at binding | A contract mismatch surfaces before any device work, never as a wrong number |

## Efficiency inside the boundary

Containment is not paid for with speed; it is where the speed comes from.

| Where the cost would be | What contains it |
|---|---|
| Choosing a kernel per step | Selection at plan time; a plan is a fixed tuple of executables |
| Hundreds of launches per decode step | One captured sequence per invocation geometry; static operands viewed once |
| Scratch per operation, resized independently | One arena, regions merged by name, allocated once per geometry |
| Recomputing what a container is on every use | Residency decides a representation once; kernels specialize on it |
| Copying history when batches change | Physical extents with claims; logical visibility per sequence |
| A branch per backend in every kernel | The compiler lowers portable operations per target; kernels do not know |

An optimization that needs a neighbor's internals is a leak in disguise. It is
implemented by moving the decision to the component that owns the inputs, or by
giving the compiler a portable operation, never by opening the boundary.

## What leaking looks like

```text
operation:  if backend == METAL and rows < 8: ...        backend leaked into selection
model:      from kernels.metal_decode import ...         a schedule leaked into the equations
driver:     if backend == LLVM: run_pass(...)            the compiler leaked into the runtime
kernel:     call_extern("simd_sum", ...)                  the target leaked into a schedule
weights:    layout = AFFINE if backend == METAL          the backend leaked into residency
```

Each of these is a decision made where its inputs do not live. The fix is never a
cleaner branch; it is relocating the decision behind the boundary that owns it.

## Documents

| Component | Document |
|---|---|
| Execution owner | [platform.md](platform.md) |
| Kernel | [kernels.md](kernels.md) |
| Operation | [operations.md](operations.md) |
| Weight residency | [weights.md](weights.md) |
| State | [state.md](state.md) |
| Model executor | [models/executor.md](models/executor.md), [models/qwen35.md](models/qwen35.md) |
| Generation | [engine/generation.md](engine/generation.md) |
| Service | [engine/service.md](engine/service.md) |
| Serving | [serving.md](serving.md) |
| Measurement | [performance.md](performance.md), [benchmark-fixtures.md](benchmark-fixtures.md) |

## Outside this design

Speculative generation, retained prefixes across requests, expert routing and
multiple execution owners are not part of this engine. Each would be a component
with its own boundary, added without opening the others.
