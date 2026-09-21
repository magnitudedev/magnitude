---
applies_to:
  - inference-v4/seismic/crates/seismic-lang/**
  - inference-v4/seismic/crates/seismic-compiler/**
  - inference-v4/seismic/crates/seismic-metal/**
  - inference-v4/seismic/crates/seismic-cpu/**
  - inference-v4/seismic/crates/seismic-cuda/**
  - inference-v4/seismic/crates/seismic-runtime/**
  - inference-v4/seismic/crates/seismic/**
  - inference-v4/seismic/crates/seismic-build/**
  - inference-v4/solver/**
  - inference-v4/engine/**
---

# Seismic compilation

The ordinary compiler artifact progression is:

```text
CheckedModule -> LogicalEntry -> ImplementationDraft<B>
             -> NativeKernelCandidate<B> -> NativeKernel<B>
             -> PlanSpace<B> -> FrozenPlan<B> -> ExecutableVariant<B>
             -> PreparedKernel<B> -> WorkflowDraft<B>
             -> PreparedWorkflow<B> -> AdmittedWorkflowRun<B>
             -> Execution<B> -> Completion<B>
```

Each transition consumes its input. There is no other semantic layer, no
realization subsystem, no independently sealed strategy, dataflow,
placement, kernel, consequence, or occurrence artifact, no reference
fallback, retry compiler, compatibility route, greedy selector, or backup
implementation. Private algorithms may normalize, infer, schedule, or emit,
but they never introduce a public or cross-crate artifact that restates the
program.

An explicitly selected top-level native implementation is a separate terminal route, not another
compiler progression:

```text
CheckedModule -> LogicalEntry -> InvocationContract
             + embedded native source + authored launch
             -> NativeKernel<Entry> -> direct synchronous Metal completion
```

This route reuses the checked entry contract and public tensor runtime but constructs none of
`ImplementationDraft`, compiler kernel IR, `PlanSpace`, `FrozenPlan`, `ExecutableVariant`,
`PreparedKernel`, or workflow artifacts. It has no tuning, solving, duration model, candidate
selection, retry, or fallback. The distinct public handle makes direct-only use structural.

## Principles

1. One fact, one owner, one representation. A fact that affects legality,
   selection, resources, numerics, layout, or execution is owned by exactly
   one artifact; other layers consume it by reference or as a derived value.
2. Invalid compiler states are unrepresentable where the semantic category is
   known: private fields, opaque scoped ids, typed builders, non-empty
   collections, refined enums, consuming transitions. Validators do not
   compensate for open structs.
3. Alternatives are closed: an implementation contains its schedule, kernels,
   transfers, storage topology, constraints, numerical transfer, and modeled duration.
4. Planning uses complete machine truth. Device-wide facts are in the device
   contract, concrete kernel facts are in reflection-reconciled native-kernel
   contracts, and measured performance facts are in the execution profile.
5. Native compilation and reconciliation happen before plan-space admission.
   They consume kernel-affecting choices, do not redesign, and cannot reject a
   later selected plan for a planning fact.
6. Runtime executes; it does not prove. It never discovers an inconsistency
   between compiler artifacts.
7. Errors model reality; panics model bugs. A large family of panic sites is
   itself an architectural defect.

## Artifacts and authority

| Artifact | Owns | Must not own |
|---|---|---|
| `CheckedModule` | source semantics, types, effects, canonical bodies, lowering declarations, top-level native asset references and launch expressions, capability requirements, stable identities | target decisions, schedules, allocations, native code bytes |
| `LogicalEntry` | monomorphized entry semantics, `CallSchema`, `EntryDomain`, canonical operation graph, provenance, the entry's expression arena | placement, algorithm selection, native limits |
| `DeviceContract<B>` | device-wide compatibility, capabilities, hard limits, memory rules, toolchain modes, and numerical environment | kernel-specific limits, measured rates, selected plan |
| `NativeKernel<B>` | one concrete kernel handle, exact ABI, launch domain, reflected resources, numerical mode, and service footprint | unresolved codegen choices, performance observations |
| `ExecutionProfile<B>` | measured service definitions, uncertainty, qualification domains, and per-open performance identity | legality, program semantics, transient availability |
| `PlanSpace<B>` | exactly one universal implementation, zero or more optimized machine-closed implementations, and one exact finite solver model | partial proposals, unreconciled native candidates, fallback |
| `FrozenPlan<B>` | one fixed physical choice with symbolic invocation dimensions, exact guard, layouts, allocations, structured commands, numerical assessment | alternatives, solver objects, logical IR, native mirrors |
| `ExecutableVariant<B>` | one native structured schedule, guard/duration/layout evaluators, binding table, identity, assessment | physical plan, logical program, plan space |
| `PreparedKernel<B>` | call schema, target domain, non-empty covered portfolio, deterministic selector | compilation logic, uncovered domain, inter-call scheduling |
| `PreparedWorkflow<B>` | dependency-closed topology, symbolic access hazards, and reusable submission structure | transient reservations, selected invocation variants |
| `AdmittedWorkflowRun<B>` | bound invocations, selected variants, one atomic reservation set, retained resources, and submission ownership | compiler repair, retry selection |

Only the checker and the validated bundle decoder construct a checked
module. Only the module constructs a logical entry. Only the portfolio
builder constructs a prepared kernel, after proving coverage.

## Identities and expressions

Every semantic identity is an opaque arena index with a crate-private
constructor; region-local identity carries its region. Stable identity is
content-derived and is the only identity that crosses a bundle boundary or
enters a cache key.

One hash-consed typed expression DAG per entry (`Nat`, `Int`, `Bool`,
`Scalar<T>`, `Duration`) drives solver constraints, partial evaluation,
applicability guards, layout, geometry, allocation sizes, numerical bounds,
and modeled duration. Its free symbols are call dimensions, call scalars, target
constants, finite decisions, loop binders, and schedule scalar slots. There
is no string symbol, sentinel, or second formula language. Integer semantics
are mathematical; runtime representability restricts the target domain
rather than wrapping.

The one DAG has two non-interchangeable authority wrappers. `PlanningExpr`
contains only finite decisions, exact Boolean/table/linear structure, and
target constants accepted by the complete solver adapter. `InvocationExpr`
is the total checked evaluation language for call-dependent products,
division, remainder, alignment, folds, guards, geometry, and layout. Planning
expressions embed into invocation expressions; invocation expressions never
enter the solver. Raw solver assignments are private and become
`FeasibleAssignment` only after direct evaluation of every immutable planning
constraint.

## Machine contracts and capabilities

Catalog discovery only enumerates unopened physical devices. Opening a Metal device creates the
production service/queue immediately. Its device contract and fixed primitive execution profile
are acquired together and cached on first use by ordinary compiler preparation or capability
introspection. This preserves exact pairing with the opened service while allowing explicitly
selected direct native calls to avoid profiling entirely. Other backends may still acquire their
complete profile while opening.

The device contract is assembled once from backend revision, hardware identity
and device-wide limits, driver and toolchain versions, dtype support, the
numerical environment, and the static capability registry. It contains no
fact whose truth depends on a particular compiled function or pipeline.

Every numeric performance fact is either derived by a sound
documented rule from architectural/device facts or measured by a backend-owned
primitive-service probe on the exact opened device before planning and
compilation. Measured facts bind their probe/methodology, interval/uncertainty,
and complete target identity into a per-open execution-profile identity. A
separate stable compatibility identity contains only legality/codegen facts and
keys native artifacts; raw timing observations do not invalidate reusable
native code. Prepared selection is never reused under a different execution
profile. Candidate implementations are never
benchmarked to create these facts. There are no calibrated coefficients,
fitted curves, arbitrary weights, guessed defaults, copied values from similar
hardware, nominal-peak shortcuts, or unknowns represented as zero.

Kernel-affecting decisions are fixed before native formation. Compilation
produces an unusable `NativeKernelCandidate`; reconciliation consumes it and
authoritative reflection to construct `NativeKernel`. Its contract records
the actual ABI, launch domain, pipeline/function limits, static local memory,
register and spill usage where exposed, cooperative requirements, numerical
mode, service footprint, and compatibility identity. Unknown legality or
selection facts are not represented as zero and prevent admission of that
native implementation. Native compilation is absent below `PlanSpace`.

Every emitted command, primitive, and intrinsic declares demand over sealed
service classes in the same registration that supplies its lowering. Profile
assembly requires exactly one authoritative query, derivation, or measurement
provider for every referenced class and rejects duplicate/unused providers.
There is no optional/default/catch-all service. A backend that cannot construct
the required execution model does not advertise that target as complete. Capabilities are typed
intrinsic families; a backend advertises a signature only when the same
registration provides its typed lowering, resource rules, and native
emission. Registration is sealed at compiler initialization; an inconsistent
registry is a startup panic. Native compilation is forbidden from returning
an unsupported-capability or resource result for anything the profile
represents.

Service measurements retain distinct batches for dependency, setup, and
capacity observations, including workload units, timer resolution, raw
observations, digest, method, and acquisition duration. Qualification cases
state only held-out production service demands and observations; core computes
their prediction through the same service model used for selection. Service
intervals carry correlation identity and an explicit qualification domain.

## Implementations

Implementation factories, portable and backend-specific, receive a semantic
function, the target profile, the shared arena, the precision policy, and
core-owned builders. A factory may decline before construction; once
construction begins it returns a closed implementation or a real preparation
error. Calls are resolved during construction: every applicable child
implementation is spliced under a finite decision, composing guards,
constraints, lifetimes, transfers, durations, provenance, and effect ordering.
No call survives into a schedule.

Factories use a sealed refinement-rule API. They cannot fabricate raw schedule,
storage, synchronization, numerical-transfer, or demand nodes. Each rule
consumes semantic obligations and produces locally valid executable structure;
only a draft with no remaining value, event, output, lifetime, numerical, or
demand obligations can close.

Kernel IR is typed by value category and representation; branches own their
joins and repeats own their carries with identical typed schemas. Global and
launch-local storage are different types; native launch bindings accept only
global views. Materialization is a compiler operation derived from use, never
source ceremony. Allocation topology (representation, alignment, symbolic
bytes, lifetime, alias facts, reuse decisions) is owned by the implementation
and every resource expression derives from it once.

## Planning and coverage

Every compile-time decision is a finite explicit domain owned by one
implementation. A codegen decision changes emitted instructions, static local
memory, numerical mode, ABI, or native resources and is enumerated before
native formation. A launch decision changes only runtime geometry within one
closed native launch domain and may remain in the planning model. Invocation
dimensions stay symbolic. Target limits and
numerical admissibility are solver constraints, never post-selection checks.
The solver exports Boolean structure exactly, including disjunction,
negation, and reified comparison.

Modeled duration is the result of the target-semantic resource/dependency execution model
over the same structured schedule, typed operations, allocation topology,
geometry, and target profile as execution. Core construction accounts for
exact dynamic launch multiplicity, operation/intrinsic classes, dependency
latency, issue-resource demand, effective concurrency and residency, memory
transactions, overlap, barriers, atomics, and command synchronization. A
factory cannot assign or omit duration. Proxy lexicographic counters, empirical
calibration, arbitrary weights, hard-coded timing guesses, and nominal peak
formulas are forbidden. Missing behavior is a compiler/backend-model bug, not
acceptable estimate error. Structural demand is exact; physical service time
is measured and retains timer/acquisition uncertainty because future wall-clock
time changes with thermals, power, OS scheduling, and contention. The model
propagates that interval and never describes a prediction as physical proof.
Data-dependent control and addressing widen the interval across every possible
path/access class; the compiler never invents branch probabilities, cache-hit
rates, or expected input distributions.

Backend qualification requires relative interval half-width at most 1% for
compute-service facts and 2% for memory, transfer, dispatch, synchronization,
barrier, and atomic facts. Fixed held-out regular compositions must be predicted
within 5% absolute relative error. These compositions validate uncertainty; they
never fit correction coefficients or candidate-specific behavior. Selection
intervals include acquisition, semantic, and composition uncertainty, and an
overlapping difference is not reported as a physical performance win.
Metadata such as constants, views, and allocation
declarations cannot form launch boundaries, and structured control stays
within a launch unless a real execution or synchronization boundary requires
otherwise.

Coverage is constructional. `PlanSpace::new` requires one
`UniversalImplementation` whose type admits no decisions, whose numerical
transfer is exact, and whose legality is total over the independently derived
target domain. Optimized implementations have a different type and cannot
impersonate it. Optional optimization is governed by one preparation budget
covering solver work/memory, optimized assignments, unique native templates,
native compile time and code bytes, executable variants, and metadata bytes.
Budget exhaustion retains the already-closed universal portfolio and reports
non-optimality; it never returns partial coverage. Consumers supply no
envelopes, buckets, classes, or expected shapes.

Selection at invocation validates the call against the schema and target
domain, evaluates guards, and picks the minimum `(upper_duration, identity)`.
Non-overlapping intervals establish measured separation; overlap is recorded
honestly rather than treated as proof that future wall-clock time is ordered. Zero
matches after validation contradicts the private constructor and is a panic.

## Native formation and runtime

Each backend forms and reconciles native kernels before plan-space admission.
A frozen plan selects only closed kernels and binds one native schedule over
the shared structured step type; there are no backend schedule mirrors and no
late physical/native comparison. Native errors are toolchain, malformed
output, device loss, cache, and toolchain resource exhaustion only.

Runtime execution is workflow-based. Closing `WorkflowDraft` derives all
inter-call hazards from semantic event manifests and retains unresolved
may-alias relationships as binding obligations. Admission validates every
call, jointly selects admissible variants, atomically acquires one reservation
set, and constructs `AdmittedWorkflowRun`. Only that owned value can submit;
submission returns an execution handle and completion releases resources.
There is no device-wide lock held across execution and synchronization.
`Kernel::call` is a synchronous one-node workflow convenience; production
model execution prepares at least one complete decoder-step workflow. Runtime
never infers placement, repairs a plan, retries selection after execution, or
interprets the portable body.

## Public integration

`seismic-build` checks sources at build time, emits a versioned checked
bundle, and generates typed bindings (`Args`, `Results`, entry handles).
Consumers import only `seismic` and `seismic-build`, provide tensors and
ordinary parameters, prepare with `for_device`, and `call`. Only genuinely
polymorphic element representations are compile-time bindings.

For the explicit direct-native route, runtime renders the canonical registry descriptors of those
compile-time bindings and of every tensor ABI leaf into the Metal source prefix before compiling
the pipeline. The rendered source therefore changes with representation bindings without adding a
second layout authority or runtime representation branch.

## Failure taxonomy

Source, bundle, target, preparation (`NoApplicableImplementation`,
`NumericalPolicyInfeasible`, `TargetDomainUnrepresentable`,
`SolverResourceExhausted`, `NativeCompilation`), invocation, and execution
errors are the complete typed taxonomy. No variant means two compiler phases
disagreed. Permitted panics are: inconsistent static registry, out-of-arena
private id, violated FFI precondition by Seismic code, unrecoverable poisoned
lock, solver witness contradicting the immutable model, and the prepared
kernel coverage invariant.

## Identity, caching, telemetry

Native cache keys are module semantic hash, entry, backend/compiler version,
stable compatibility identity, native-kernel identity, precision policy identity,
implementation/variant identity, and native toolchain identity. In-process
prepared portfolio keys additionally include the per-open execution-profile
identity. Runtime dimensions never create preparation keys. The cache stores checked bundles and executable variants; decoding
validates version, hash, target identity, and binary integrity. Telemetry is
OpenTelemetry at preparation, native compilation, selection, allocation, and
execution; it carries no legality fact back into planning.

## Acceptance criteria

- No struct literal or public constructor can fabricate a checked module,
  logical entry, implementation, native kernel, frozen plan, prepared kernel,
  prepared workflow, or admitted run.
- Every id is opaque and scoped; no map is keyed by a bare region-local
  number.
- Every runtime and solver formula references a node of the entry arena.
- Only directly evaluated `FeasibleAssignment` values freeze, and freezing
  performs no native compilation or legality decision.
- It is impossible to construct a prepared kernel with an uncovered
  target-domain point.
- An unreconciled native candidate cannot enter planning or execution.
- Device contracts contain no pipeline-specific fact; concrete Metal and CUDA
  resource/launch behavior comes from each native-kernel contract.
- Only an admitted workflow can submit, and its selected variants,
  reservations, buffers, and native objects have one owned lifetime.
- Backend crates contain one native schedule type instance and no plan
  mirror; runtime crates contain no compiler-consistency branch.
- The engine imports only `seismic` and its generated bindings.
- The forbidden symbols of the superseded architecture (proposal, recipe,
  algorithm label, placement enum spanning ABI and local storage, sealed
  value joins, encoded plan mirrors, workload envelopes, capacity classes,
  defect taxonomies) do not exist.
