---
applies_to:
  - inference-v4/seismic/crates/seismic-lang/**
  - inference-v4/seismic/crates/seismic-compiler/**
  - inference-v4/seismic/crates/seismic-realization/**
  - inference-v4/seismic/crates/seismic-metal/**
  - inference-v4/seismic/crates/seismic-cpu/**
  - inference-v4/seismic/crates/seismic-cuda/**
  - inference-v4/seismic/crates/seismic-runtime/**
  - inference-v4/engine/**
---

# Seismic compilation

The sole production path is:

```text
CheckedProgram
  -> SpecializationDomain (complete exact/bounded entry shape and element bindings)
  -> LogicalProgram
  -> form_plan_space: the core forms one PlanSpace<BackendDialect>
     from the backend's declarative mapping catalog
  -> solve one global planning model
  -> resolve the one complete assignment -> PhysicalPlan<BackendDialect>
  -> mechanically encode every sealed launch
  -> assemble NativeArtifact
  -> prepare one validated invocation and execute
```

There is no baseline compiler, fallback scheduler, retry compiler, repair pass,
alternative native compiler, or post-selection resource check. Serial, CPU-worker,
grid, grid-stride, subgroup, matrix, fused, split, and multi-launch executions are
peer strategies in the one `PlanSpace`; the universally applicable serial and
grid-stride forms are ordinary members of that space, not a side channel.

## Artifacts and authority

1. `CheckedProgram` owns types, ownership, mutation permission, control and
   reference meaning, function families, and capabilities. It decides nothing
   about tasks, launches, memory spaces, or limits.
2. `LogicalProgram` is one target/workload specialization: occurrence-qualified
   implementation choices plus hierarchical SSA task graphs. It owns
   specialization, logical storage and views, structured control, calls,
   reductions, dependencies, safety obligations, and runtime extents. It names
   no participants, geometry, allocation, barriers, or opcodes.
3. `PlanSpace<D>` is solver input, not serialized IR. It owns legal
   algorithms and mappings, fusion and splitting, physical storage and
   transport, synchronization, exact hard resources, bounded native-resource
   contracts, capabilities, numerics, and cost.
   It also retains the complete effective target profile supplied once by the
   backend; planning and resource legality have no second target authority.
4. The planning model owns the one global implementation, strategy, tuning,
   activation, dispatch, storage-placement, capability, safety, and
   numerical-policy assignment.
5. `PhysicalPlan<D>` owns the selected nested schedule, opcodes, runtime
   geometry expressions, offsets, resources, and the numerical assessment.
   The root plan alone owns the public ABI, the global storage table, and the
   internal arena; nested call bodies reference transports in that same
   table and never own a second ABI or arena.
6. `NativeArtifact` is the mechanical encoding of the physical plan. It
   mirrors the physical execution tree, collapsing only statically empty or
   singleton structural wrappers; dynamic `If`/`Repeat` control is never
   flattened away.
   Backend `encode` over one sealed launch is the sole physical-to-native
   launch transition; assembly consumes that encoded launch and never
   re-emits a retained launch through a second path.
7. The runtime validates bindings once at preparation, binds dense indices,
   evaluates retained execution expressions, submits in retained order,
   skips zero-work launches, and reports status. It makes no compilation or
   selection decision; the explicit plan-preparation module is the only
   runtime-crate caller of the compiler.

Solver state is private planning state and is never executable.

## One schedule authority

Order is expressed once, as a structured execution tree of
`Launch`/`Guard`/`Call`/`If`/`Repeat` steps. Sibling steps complete in order; `Guard`
evaluates a retained structural safety predicate before subsequent work; `If`
evaluates one retained predicate and one branch; `Repeat` evaluates a
retained half-open range and rebinds its scalar binder and carries each
visit. Loops or conditionals consumed wholly by one launch become kernel-local
control; those containing retained calls or multiple launches become the
corresponding structured schedule steps. There are no phase lists and no
predecessor edges parallel to item order.

A resolved repeat retains its bound through native encoding. Executors validate
`0 <= start <= end <= bound` before conversion or iteration and report an
invalid range as a typed execution-contract failure.

Retained calls remain nested: a sealed call retains the child plan body and
its boundary environment, whose routes resolve directly to caller storage
IDs. An absorbed call is owned completely by its ancestor physical strategy
and has no call step. Only the root boundary has ABI allocations.

Logical values, inter-step transports, and kernel-local values are distinct.
Transports carry storage, executor scalars, tuples, or boundary values between
schedule components. A kernel-local SSA value exists only inside one kernel
and is never represented as a transport. Kernel inputs, SSA definitions,
iteration axes, and published outputs are typed separately.
Closed kernel places are the sole authority for storage views and transforms;
backend opcodes reference places instead of copying view metadata.

## Family, model, and solve

The compiler core alone forms the complete `PlanSpace<D>`: occurrence
canonicalization, strategy shapes, routes, residences, kernel blocks, and
consequences are core-owned formers over sealed intermediates. A backend
supplies only a declarative mapping catalog (optional rule families that
propose over occurrence facts), its typed intrinsic catalog with exhaustive
encoders, and the native assembler; backends never import logical
construction, plan-space construction, or solver construction. Every
physical strategy owns one complete root logical occurrence and may own
complete descendant call occurrences. A retained call remains a schedule
`Call` and activates a child strategy; an absorbed call is owned and
realized completely by its ancestor strategy. The selected strategies form
a non-overlapping ownership tree rooted at the entry. Cross-call fusion
never copies, imports, remaps, or mutates a logical graph.

View transforms obey the same qualification boundary as values. Templates use
graph-qualified dynamic endpoints, kernel-block formation resolves them to
closed kernel value references, and physical plans and backends never carry
bare graph-local endpoint ids.

Every applicable portable alternative receives a universal physical strategy,
or that is a compiler defect. Strategies are formed by the core's private
formers over the complete owned-occurrence set; there is no public
plan-space, strategy, or kernel-block builder. Mapping, fusion, splitting,
scheduling, calls, and obligation discharge consume canonical
occurrence-qualified identities, and a strategy seals only when every
obligation of every owned graph has been consumed exactly once.

The plan space supplies only legal choices and their exact constraints; it carries
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

Resolution consumes exactly the solver's complete assignment; it is a
substitution, not a second decision. It evaluates solved
expressions, allocates dense IDs, instantiates boundaries, substitutes
offsets/geometry/opcodes, and recurses; it performs no ordinary legality
check and cannot substitute a plan-space-time or backend-time decision for a
solver assignment. The resulting `PhysicalPlan` is self-contained and
sealed: construction is private to the core's seal, every reference is a
dense typed index that is in-bounds by construction, physical
storage and scalar-slot references are resolved IDs, input scalar references
carry final ABI byte offsets, result scalar destinations carry final result
field IDs, and storage placement is a closed typed variant. No backend
retains or replays the plan space to reconstruct those identities.

## Native emission

The compiler core encodes each physical launch exactly once and the backend
assembles the already-encoded hierarchy. Encoders may assign native names and
instruction spellings, but cannot introduce algorithms, allocation, geometry,
synchronization, copies, or numerical transformations, and cannot reject a
selected opcode. Emission failures are compiler defects, toolchain failures,
or system failures — never a reason to retry another candidate.

Native compilation is not a planning oracle. Reflected native facts
(telemetry and the selected bounded native-resource contract) evaluate
physical geometry; a fact outside its declared domain is a compiler defect.

## ABI and invocation

The root ABI is created once from the entry interface and the canonical leaf
traversal: dense tensor leaves have one typed buffer, packed leaves have
registry-ordered planes, scalar and index inputs are typed fields, and input
ranges are adjacent start/end fields. Tensor results are runtime-allocated by
path and plane; scalar, index, and range results decode from a
compiler-owned result scalar block. Internal arena and nested boundaries
never appear in the ABI; capability values are forbidden.

Invocation results expose owned tensor planes separately from typed scalar
results. Scalar results retain canonical result paths; range results retain an
explicit start/end endpoint, never positional pairing by convention.

A range remains one semantic boundary leaf but expands in compiler core to two
explicit physical kernel leaves (`Start` and `End`). All other semantic leaves
expand to one kernel leaf. Backends consume those identities directly and do
not infer aggregate structure from transports.

Packed tensor leaves expand to the representation registry's ordered physical
planes. Each plane has its own storage view, ABI buffer when public, and direct
native binding slot; no layer collapses a packed leaf to its first plane.

Native bindings are direct-only: every storage plane, by-value scalar, and
system block has one core-assigned slot. Unsupported descriptor/argument-table
modes are absent rather than modeled differently from their native encoding.

Alias rules are retained in the root ABI (shared-read ranges may overlap;
exclusive/owned ranges are disjoint from other live parameter ranges;
results are distinct). The runtime validates them over actual byte ranges
before submission. Invocation values or aliasing that violate the retained
ABI are invocation errors, rejected before submission.

Physical strategies cannot allocate or redefine root ABI storage. Every
kernel allocation and synchronization object is declared by the physical
kernel that uses it. Solver resource constraints, the selected physical plan,
and native emission consume those same declarations; emission cannot create an
undeclared local allocation or staging resource.

## Consumer preparation

Compilation is preparation-only for every consumer. The engine compiles each
component — decoder, sampler, conditioned overlays, encoder, and weight
import — through one preparation session before any request; execution and
inference modules hold prepared artifacts and cannot import the compiler. A
request outside a prepared workload envelope is rejected as a caller
diagnostic, never compiled on demand. Prepared artifacts are keyed by
complete specialization identity, so serving the same workload dispatches
among sealed compilations and never recompiles.

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
  consumed exactly once by the strategy that owns its complete occurrence.
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
- Sealed artifacts expose accessors only; no public field or constructor
  bypasses the core's seal, and dense indices are in-bounds by construction.
- Backend catalogs are declarative: a rule either proposes over occurrence
  facts or declines; it never constructs schedule structure.
- `PlanSpace` contains no assignment, default selection, placement, or
  executable witness; every physical decision consumed by resolution comes
  from the solver assignment.
- Backend encoders accept one physical launch and cannot see the plan space.
- Native assembly accepts only the physical plan; template-to-physical replay
  maps and plan-space retention are forbidden.
- Runtime executes physical plan order and never infers ABI identity from
  names.
