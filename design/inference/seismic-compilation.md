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

Seismic has four authoritative compilation artifacts:

1. `SemanticProgram` is checked source meaning: types, ownership, effects, capabilities and
   numerical semantics.
2. `LogicalProgram` is one target/workload specialization. Each applicable implementation owns a
   scheduling-normal `LogicalTaskGraph` whose tasks have one coherent logical domain and whose
   dependencies make value, effect and ownership order explicit.
3. `ResolvedPlan<D>` is one complete executable refinement. It owns nested plans, phases, launches,
   participant mappings, instructions, storage, binding groups, synchronization, resources,
   numerical effects and the complete selected assignment.
4. `NativeArtifact` is the mechanical backend encoding of that resolved plan.

There is no second physical graph, post-selection linker or native feasibility retry. Solver state
is private planning state and is never executable.

## Semantic to logical

Specialization retains all applicable implementations for one target/workload identity.
Normalization decomposes each implementation into tasks before backend planning:

- independent domains become explicit logical axes;
- ordered iteration and carried state remain ordered axes;
- reductions are explicit task operations;
- calls are typed choice boundaries with explicit input and result operands;
- data, mutation and ownership order become graph dependencies; and
- result storage identities and paths exist before physical planning.

A scalar task body cannot hide another independent scheduling domain. Task-local views contain
only identity, reshape and transpose transforms. Dynamic slicing is an index operation whose
endpoints are task-local scalar expressions, so terminal instructions never retain unresolved
program-wide value identities.

Logical structure names no threads, workgroups, tiles, memory spaces or backend limits.

## Logical to executable

Each backend elaborates a `PlanFamily<D>` of complete schedule alternatives. A private
`ScheduleBuilder` owns the obligations for every task, call, dependency and output. Mapping,
subplan composition, synchronization and publication consume those obligations exactly once;
`finish` is the only way to construct an alternative.

A complete alternative contains:

- explicit maps from logical axes to workgroup, participant, subgroup or serial execution;
- non-empty phases and launches with launch-local admitted instructions;
- exact value transports, including representation planes, tuple structure and zero-channel
  `Void`;
- storage with size, alignment, replication and provenance;
- binding groups and access modes;
- program-order, barrier or launch-boundary placement for every dependency; and
- symbolic cost, resource, capability and numerical facts derived from those same objects.

The common planner builds one finite solver model over this family. It selects complete logical
and physical alternatives, structural integers and an admissible numerical assignment. Target
limits and precision are hard constraints. Greedy and exact differ only in search; both resolve
through the same constructor. Bounded search may return a legal incumbent with `optimal = false`,
but never a partial plan.

Resolution evaluates symbols once and produces a nested `ResolvedPlan<D>`. It cannot add storage,
change mappings or choose another implementation. Invocation alias rules are derived from the
resolved plan and exact backend-retained public ABI storage identities.

## Native emission

The compiler core recursively visits the resolved schedule and calls `encode_launch` exactly once
for every resolved launch. The backend assembles the already-encoded hierarchy. Encoders may
assign native names and instruction spellings, but cannot introduce algorithms, storage, copies,
barriers, mappings or numerical transformations.

Metal emits MSL; CUDA emits PTX with the resolved launch bound; CPU emits Cranelift functions for
resolved phases. Native artifacts preserve exact public ABI order and storage identities. Runtime
execution binds that ABI and executes retained phase/launch order without reconstructing compiler
decisions.

Native compilation is not a planning oracle. A JIT rejection or resource contradiction after
resolution is a backend invariant defect, not a reason to retry another candidate.

## Capabilities and precision

Hardware and driver observations form a target capability profile. Kernel authors require a
backend capability through its intrinsic family; ordinary hardware/toolchain variation remains a
backend concern. An alternative requiring an absent capability is inadmissible.

Every selected instruction carries numerical effects. Approximate native operations therefore
participate in whole-plan precision admissibility before resolution. Evidence is keyed to the
complete logical identity, target/toolchain profile, assignment and precision method; evidence for
one composition cannot justify another.

## Failure boundaries

- Invalid types, ownership, effects or logical control are semantic failures.
- Missing applicable implementations are coverage failures.
- No complete target-legal schedule is planning infeasibility.
- No assignment satisfying requested precision is numerical infeasibility.
- A resolved plan or native artifact contradicting retained facts is a compiler defect.
- Invocation values or aliasing that violate retained ABI are invocation errors.

No failure silently changes backend, search strategy or precision policy.

## Invariants

- Every active logical task, call, dependency and output is consumed exactly once.
- Parallel work is visible only as logical domains plus explicit participant maps.
- Ordered work cannot become parallel without a different semantic implementation.
- Calls become nested resolved plans before emission; emitters never receive logical call bodies.
- Storage, synchronization, resources and numerical effects belong to selected executable objects.
- Solver selection resolves once; native emission never retries planning.
- Backend encoders accept one resolved launch and cannot see an open plan family.
- Runtime executes resolved native order and never infers ABI identity from names.
