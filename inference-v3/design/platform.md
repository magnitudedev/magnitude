# Physical tensor execution

**Ops owns device tensors, allocation, binding, submission and physical
completion and physical observations; TileLang owns target execution adapters. Magnitude holds only
generic resources and completion obligations.**

## Ownership

```text
Ops device owner
├── physical allocations and aliased tensor views
├── compiled-callable cache and immutable bindings
├── source I/O, transfer/conversion and reusable planned temporary slots
├── submission order and outstanding executions
└── TileLang target and runtime adapter
```

One owner governs one device execution domain. Upper layers cannot release a
physical resource, reorder tensor work or observe a backend. TileLang realizes
the target but does not decide model allocation, graph regions or inference
state policy.

An operation defines the entire physical sequence, not only its numerical kernels.
The shared runtime realizes and observes those actions. Per-invocation memory
windows report baseline, peak and end reservations against physical constraints;
aliases do not increment reservations. Host API spans and byte counts are not
hardware counters or native device timings. Completed observations require actual
completion, not merely a successful submit. The normative measurement boundary is
in [formula execution](../../design/inference/formula-execution.md).

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
its device work completes; Ops still fulfilled and retired the physical
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

Ops gives TileLang an ordered ABI, Python-authored TileLang kernel work,
its static binding choices and a target. TileLang's public eager builder creates
the final portable single-entry `IRModule`; its runtime returns an opaque pre-bound
native entrypoint and target capability information. Ops does not select
compiler passes, adapter internals, flags or backend pipelines.

Ops's runtime adapter opens the physical execution domain and may use
framework tensors and events strictly as ABI-compatible storage and completion
handles. It owns allocation policy, resource leasing, completion aggregation and
the association of work with a completion. Those handles perform no numerical
computation and encode no backend-specific kernel behavior.

TileLang owns source compilation, ABI validation and binding, stream integration,
native multi-launch command encoding and execution through its existing target
adapters. This boundary does not require—and Ops must not induce—a
TileLang-level Device, Allocation, Completion or Executable object model.
Compilation-unit selection remains a Ops responsibility.

## Capability

TileLang reports behavior: subgroup and matrix geometry, supported dtypes,
memory scopes, asynchronous movement, synchronization, atomics, alignment and
launch limits. Ops uses these facts only during lowering and schedule
selection. Magnitude never receives them.

Capabilities describe what the selected target pipeline actually supports. They
do not encode a vendor name, and Ops does not maintain a second hardware
database or probe native APIs independently.

## Host coordination

The engine worker owns its Ops device owner on one thread. Completion
wakes cannot be starved by control work, and idle service does not poll. Tests
and measurements take exclusive access to a device execution domain so timing
and stateful work never overlap accidentally.
