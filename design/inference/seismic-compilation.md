---
applies_to:
  - inference-v4/seismic/crates/seismic-lang/**
  - inference-v4/seismic/crates/seismic-compiler/**
  - inference-v4/seismic/crates/seismic-realization/**
  - inference-v4/seismic/crates/seismic-metal/**
  - inference-v4/seismic/crates/seismic-cpu/**
  - inference-v4/seismic/crates/seismic-cuda/**
  - inference-v4/seismic/crates/seismic-runtime/**
---

# Seismic compilation

The sole production path is:

```text
CheckedProgram
  -> specialize one entry and its implementation choices
  -> LogicalProgram
  -> construct one PlanFamily<BackendDialect>
  -> solve one global planning model
  -> ResolvedPlan<BackendDialect>
  -> mechanically encode NativeArtifact
  -> validate bindings and execute
```

There is no baseline compiler, fallback scheduler, retry compiler, repair pass,
alternative native compiler, or post-selection resource check. Serial, CPU-worker,
grid, grid-stride, subgroup, matrix, fused, split, and multi-launch executions are
peer alternatives in the one `PlanFamily`; the universally applicable serial and
grid-stride forms are ordinary members of that family, not a side channel.

## Artifacts and authority

1. `CheckedProgram` owns types, ownership, mutation permission, control and
   reference meaning, function families, and capabilities. It decides nothing
   about tasks, launches, memory spaces, or limits.
2. `LogicalProgram` is one target/workload specialization: occurrence-qualified
   implementation choices plus hierarchical SSA task graphs. It owns
   specialization, logical storage and views, structured control, calls,
   reductions, dependencies, safety obligations, and runtime extents. It names
   no participants, geometry, allocation, barriers, or opcodes.
3. `PlanFamily<D>` is solver input, not serialized IR. It owns legal
   algorithms and mappings, fusion and splitting, physical storage and
   transport, synchronization, exact hard resources, bounded native-resource
   contracts, capabilities, numerics, and cost.
4. The planning model owns the one global implementation, strategy, tuning,
   activation, dispatch, storage-placement, capability, safety, and
   numerical-policy assignment.
5. `ResolvedPlan<D>` owns the selected nested schedule, opcodes, runtime
   geometry expressions, offsets, resources, and the numerical assessment.
   The root plan alone owns the public ABI, the global storage table, and the
   internal arena; nested call bodies reference transports in that same
   table and never own a second ABI or arena.
6. `NativeArtifact` is the mechanical encoding of the resolved plan. It
   mirrors the resolved execution tree, collapsing only statically empty or
   singleton structural wrappers; dynamic `If`/`Repeat` control is never
   flattened away.
7. The runtime validates bindings, binds retained IDs, evaluates retained
   execution expressions, submits in retained order, skips zero-work
   launches, and reports status. It makes no compilation or selection
   decision.

Solver state is private planning state and is never executable.

## One schedule authority

Order is expressed once, as a structured execution tree of
`Launch`/`Call`/`If`/`Repeat` steps. Sibling steps complete in order; `If`
evaluates one retained predicate and one branch; `Repeat` evaluates a
retained half-open range and rebinds its scalar binder and carries each
visit. Loops or conditionals consumed wholly by one launch become kernel-local
control; those containing retained calls or multiple launches become the
corresponding structured schedule steps. There are no phase lists and no
predecessor edges parallel to item order.

Calls remain nested: `ResolvedCall` retains the child plan body and its
boundary environment, whose transports resolve directly to caller storage
IDs. Only the root boundary has ABI allocations.

## Family, model, and solve

Each backend elaborates a complete `PlanFamily<D>`: every applicable portable
alternative receives a universal physical alternative, or that is a compiler
defect. Alternatives are built through the consuming family builder; mapping,
fusion, splitting, scheduling, calls, and obligation discharge consume exact
logical IDs, and an alternative is finishable only when its logical and
physical pending sets are empty.

The family supplies only legal choices and their exact constraints; it carries
no selected, default, constructive, or executable assignment. The one global
model includes implementation, physical strategy, tuning, child activation,
dispatch and resources, storage activation, interference and offsets,
capability, safety, and whole-plan numerical policy. Global placement owns
every active device offset and constrains every storage end by device capacity;
interfering lifetimes receive ordering constraints, and sequential sibling
internals may reuse space.

The solver is the sole selection authority. Its feasible assignment contains
every implementation, strategy, tuning, activation, and offset decision.
Internal arena size is the deterministic maximum active storage end, not a
second selected value. The optimization budget limits optimization only:
feasibility is decided first, so a budget can never cause a no-incumbent
production failure. Production returns the best incumbent with an optimality
flag, never a partial plan.

Resolution accepts only feasible assignments. It evaluates solved
expressions, allocates IDs, instantiates boundaries, substitutes
offsets/geometry/opcodes, and recurses; it performs no ordinary legality
check and cannot substitute a family-time or backend-time decision for a
solver assignment. The resulting `ResolvedPlan` is self-contained: physical
storage and scalar-slot references are resolved IDs, input scalar references
carry final ABI byte offsets, result scalar destinations carry final result
field IDs, and storage placement is a closed typed variant. No backend retains
or replays the open family to reconstruct those identities.

## Native emission

The compiler core encodes each resolved launch exactly once and the backend
assembles the already-encoded hierarchy. Encoders may assign native names and
instruction spellings, but cannot introduce algorithms, allocation, geometry,
synchronization, copies, or numerical transformations, and cannot reject a
selected opcode. Emission failures are compiler defects, toolchain failures,
or system failures — never a reason to retry another candidate.

Native compilation is not a planning oracle. Reflected native facts
(telemetry and the selected bounded native-resource contract) evaluate
resolved geometry; a fact outside its declared domain is a compiler defect.

## ABI and invocation

The root ABI is created once from the entry interface and the canonical leaf
traversal: dense tensor leaves have one typed buffer, packed leaves have
registry-ordered planes, scalar and index inputs are typed fields, and input
ranges are adjacent start/end fields. Tensor results are runtime-allocated by
path and plane; scalar, index, and range results decode from a
compiler-owned result scalar block. Internal arena and nested boundaries
never appear in the ABI; capability values are forbidden.

Alias rules are retained in the root ABI (shared-read ranges may overlap;
exclusive/owned ranges are disjoint from other live parameter ranges;
results are distinct). The runtime validates them over actual byte ranges
before submission. Invocation values or aliasing that violate the retained
ABI are invocation errors, rejected before submission.

## Failure boundaries

Production failures are one closed taxonomy:

- invalid semantic program (diagnostics);
- no applicable implementation (an exact capability signature is absent);
- planning infeasible (the complete planning model has no valid assignment);
- compiler bug (a retained invariant is contradicted);
- toolchain failure; and
- system failure.

No failure silently changes backend, search effort, or precision policy.

## Invariants

- Every active logical node, call, dependency, obligation, and result is
  consumed exactly once by the alternative that maps it.
- Parallel work is visible only as logical domains plus explicit participant
  maps; logical rank is not native grid rank.
- Ordered work cannot become parallel without a different semantic
  implementation.
- Calls remain nested through resolution and emission; emitters never
  receive logical call bodies.
- Storage, synchronization, resources, and numerical effects belong to
  selected executable objects; every aggregate is scalar SSA or planned
  storage, and no implicit native local array exists.
- Solver selection resolves once; native emission and runtime never retry
  planning.
- `PlanFamily` contains no assignment, default selection, placement, or
  executable witness; every physical decision consumed by resolution comes
  from the solver assignment.
- Backend encoders accept one resolved launch and cannot see an open plan
  family.
- Native assembly accepts only the resolved plan; template-to-resolved replay
  maps and family retention are forbidden.
- Runtime executes resolved native order and never infers ABI identity from
  names.
