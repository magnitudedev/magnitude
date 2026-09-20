# Seismic compiler

The compiler has one production artifact sequence:

`Program → LogicalProgram → PhysicalProgram<Open> → PhysicalProgram<Resolved> → native artifact`

There is no post-selection reconstruction step and no second backend planner.

## Semantic to logical

Checking produces the semantic program: authored function families, ownership,
shapes, effects, control flow, calls, and numerical obligations. Logical
specialization closes one entry for a target capability fingerprint and workload.
It retains every applicable authored implementation as a guarded choice and
turns bodies into typed, choice-local fragments. Results, calls, logical storage,
dynamic extents, and numerical effects are explicit before physical planning.

Logical IR contains meaning, not hardware placement. It does not name thread
stacks, threadgroup memory, registers, native launch bounds, or ABI slots.

## Logical to physical

Each backend elaborates the complete open physical family. Every alternative is
a constructed physical object with its operations, dependencies, schedule,
storage claims, allocation choices, ABI bindings, launch envelope, resource
contributions, capability requirements, terminal mappings, numerical behavior,
and estimated cost. A backend may not defer construction to a callback that runs
after planning.

The common planner exports those finite choices and constraints to the solver.
Resolution consumes one assignment, allocates exact guarded storage, evaluates
exact resources, verifies terminal mapping completeness, and checks the numerical
policy. Allocation or resource conflicts add exclusions over the physical choices
and are resolved inside planning; they never trigger native compilation retries.

The resolved physical program is the authoritative executable plan. Its planning
report contains the assignment, estimated cost, proof status, exact launch
resources, numerical assessment, and evidence identity.

## Physical to native

Emission is mechanical and one-way. It may assign native spelling and serialize
the resolved terminal program, but it may not change operations, storage,
schedules, choices, or resource ownership.

- CPU emits selected Cranelift functions and finalizes them once. Register
  allocation may spill according to the native ABI; it is not a planning retry.
- CUDA emits exact launch bounds and PTX once. Driver resource facts that
  contradict the resolved plan are compiler invariant failures.
- Metal emits one dispatch-parametric pipeline once, then completes the resolved
  launch envelope from that exact pipeline's function-specific maximum. This
  completes a physical fact; it does not select another implementation.

## Capabilities and precision

The target environment is the intersection of backend implementation, hardware,
driver, and toolchain support, identified by a capability fingerprint. Authored
backend intrinsics are admitted only when their intrinsic family is supported.
Other hardware differences remain internal to physical elaboration and native
emission.

Numerical policy is part of planning. Exact and bounded programs admit only
physical assignments whose declared effects satisfy the policy. Qualified
whole-program evidence is keyed to the complete canonical physical assignment;
it cannot be reused for a different composition.

## Inspection

`seismic select` compiles through the unified pipeline and reports the resolved
physical assignment and exact resources. `seismic analyze-search` reports the
constructed physical choice space. `seismic emit` prints the native source
artifact where one exists (MSL or PTX), and the physical report plus terminal
module for CPU native encoding.

## Ownership

| Concern | Owner |
| --- | --- |
| Syntax, checking, semantic program, interpreter, logical specialization | `seismic-lang` |
| Common planning, exact resolution, terminal mapping, pipeline contract | `seismic-compiler` |
| Physical graph and resolution invariants | `seismic-realization` |
| Target facts, physical elaboration, native emission and execution | backend crate |
| Search algorithms and proof semantics | `magnitude-solver` |
