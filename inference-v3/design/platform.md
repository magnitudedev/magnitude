# Physical tensor execution

**Magnitensor owns device tensors, allocation, binding, submission and physical
completion; TileLang owns target execution adapters. Magnitude holds only
generic resources and completion obligations.**

## Ownership

```text
Magnitensor device owner
├── physical allocations and aliased tensor views
├── compiled-callable cache and immutable bindings
├── reusable temporary slots derived from selected graphs
├── submission order and outstanding executions
└── TileLang target and runtime adapter
```

One owner governs one device execution domain. Upper layers cannot release a
physical resource, reorder tensor work or observe a backend. TileLang realizes
the target but does not decide model allocation, graph regions or inference
state policy.

## Lifetime

```text
resource ──► tensor views ──► compiled invocation ──► completion
    ▲              each retains          retains every         │
    └── backing reclaimed only after all views and executions release ──┘
```

| Rule | Reason |
|---|---|
| A tensor view retains its allocation | Aliasing never depends on an upper layer knowing all consumers |
| Submission retains every dynamic and static resource it uses | Dropping an output cannot reclaim memory still in device use |
| Reclamation follows proven completion | Queue position or logical commit is not physical completion |
| Mutable resources are accessed through graph versions | Submitted readers cannot observe an unordered in-place mutation |
| Failure unwinds unsubmitted claims completely | A partial bind or allocation cannot leave hidden ownership |

Logical acceptance is separate. Magnitude may abort a candidate advance after
its device work completes; Magnitensor still fulfilled and retired the physical
execution correctly.

## Compiled callables

A compiled callable contains maximal compilation units, immutable constant
bindings, dynamic binding descriptions, reusable temporary slots, dependency order,
and compiler/tuning provenance. Its invocation path only validates and binds
dynamic inputs, invokes each pre-bound native entrypoint once and returns outputs
with one completion obligation.

```text
compile: graph + static facts ──► selected units + storage + bindings
submit:  dynamic resources       ──► outputs + completion
```

No graph traversal, candidate selection, memory planning, compilation, tuning,
static rebinding or per-kernel Python dispatch occurs during warm submission.
Equal graph and static identities share code; equal-shaped weights remain
distinct resources.

## Capacity

The device budget is an admission limit on charged physical bytes. Aliased views
count once; immutable weights, persistent state, temporary slots and outstanding
executions are charged to their actual allocations. A refused allocation reports
required and available capacity. Magnitude decides whether to finish work,
shrink, evict or wait; the tensor owner has no request policy.

Temporary allocation is graph-derived. Interior values of fused regions do not
exist; legal views alias; disjoint remaining live intervals reuse aligned planned
ranges. Physical realization maps each distinct range start to a slot that begins
at ABI offset zero. Slots are fixed for a compiled specialization and retained by
outstanding executions.

## TileLang boundary

Magnitensor gives TileLang an ordered ABI, Python-authored TileLang kernel work,
its static binding choices and a target. TileLang's public eager builder creates
the final portable `PrimFunc`; its runtime returns an opaque pre-bound native
entrypoint and target capability information. Magnitensor does not select
compiler passes, adapter internals, flags or backend pipelines.

Magnitensor's runtime adapter opens the physical execution domain and may use
framework tensors and events strictly as ABI-compatible storage and completion
handles. It owns allocation policy, resource leasing, completion aggregation and
the association of work with a completion. Those handles perform no numerical
computation and encode no backend-specific kernel behavior.

TileLang owns source compilation, ABI validation and binding, stream integration,
native multi-launch command encoding and execution through its existing target
adapters. This boundary does not require—and Magnitensor must not induce—a
TileLang-level Device, Allocation, Completion or Executable object model.
Compilation-unit selection remains a Magnitensor responsibility.

## Capability

TileLang reports behavior: subgroup and matrix geometry, supported dtypes,
memory scopes, asynchronous movement, synchronization, atomics, alignment and
launch limits. Magnitensor uses these facts only during lowering and schedule
selection. Magnitude never receives them.

Capabilities describe what the selected target pipeline actually supports. They
do not encode a vendor name, and Magnitensor does not maintain a second hardware
database or probe native APIs independently.

## Host coordination

The engine worker owns its Magnitensor device owner on one thread. Completion
wakes cannot be starved by control work, and idle service does not poll. Tests
and measurements take exclusive access to a device execution domain so timing
and stateful work never overlap accidentally.
