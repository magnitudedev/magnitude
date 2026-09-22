---
applies_to:
  - inference-v4/seismic/**
  - inference-v4/solver/**
  - inference-v4/engine/**
---

# Seismic compilation

The ordinary compiler artifact progression is:

```text
CheckedModule -> LogicalEntry -> RefinedCandidateFamilies<T>
             -> CandidateDomain<T> + private RealizationRegistry<T, H>
             -> EvaluatedCandidateDomain<T> -> PlannedPolicy<T>
             -> exact materialization -> PreparedKernel<T, H>
             -> WorkflowGraphDraft<T, E> -> BoundWorkflowGraph<T, E>
             -> AdmittedRun<T, E> -> SubmittedRun<T, E> -> Completion
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
`RefinedCandidateFamilies`, compiler kernel IR, `CandidateDomain`, `PlannedPolicy`, `ExecutableVariant`,
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
   transfers, storage topology, constraints, and numerical transfer. Prediction is
   derived from that closed structure; factories do not own timing estimates.
4. Candidate evaluation uses one declared method and returns a complete model
   for the entire sealed domain or fails without a partial result. Planning
   consumes only `EvaluatedCandidateDomain` and cannot observe the method.
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
| `DeviceDescription<T>` | immutable device-wide compatibility, capabilities, hard limits, memory rules, toolchain modes, numerical environment and target facts | compiler registrations, native handles, measured rates, selected plan |
| `CompilerRegistry<T>` | compiler policy: structural factories, lowering registrations, launch rules and emitted-intrinsic coverage | device observations, native contexts, analytical coefficients, executor state |
| `RealizationRegistry<T, H>` | opaque native handles keyed by reconciled implementation and artifact identity, retained privately by preparation until materialization | domain membership, performance models, planning decisions |
| `RefinedCandidateFamilies<T>` | one universal structural family, optional optimized families, finite axes, construction report and one `ClosedExecutableIr<T>` per family | native handles, performance models, solver state |
| `CandidateDomain<T>` | invocation domain, every reconciled family, immutable native descriptions and identities, finite axes, and one authoritative constraint relation | native handles, evaluator identity, partial scores, planner search policy |
| `EvaluatedCandidateDomain<T>` | exactly one objective/model for every family, correlated uncertainty, provenance and evaluator-neutral evaluation identity | gaps, unassessed regions, method-specific planner behavior |
| `PlannedPolicy<T>` | non-empty handle-free portfolio, exact guards, evaluated costs, layouts, structured commands, numerical assessments, coverage and deterministic selection | native handles, evaluator or solver services, uncovered regions |
| `ExecutableVariant<T, H>` | one materialized structured schedule, opaque native kernels, guard/duration/layout evaluators, binding table, identity and assessment | candidate alternatives, logical program, solver state, public kernel enumeration |
| `PreparedKernel<T, H>` | call schema, target domain, non-empty covered portfolio, deterministic selector | compilation logic, uncovered domain, inter-call scheduling |
| `BoundWorkflowGraph<T, E>` | selected variants, closed output descriptors, dependency topology, access hazards, lifetimes and complete symbolic resource requirements | reservations, allocations, submission |
| `AdmittedRun<T, E>` | one whole-graph reservation transaction, physical bindings, persistent leases, access permits and opaque submission ownership | binding, selection, replanning |
| `SubmittedRun<T, E>` | native completion owner plus every retained admission resource | allocation, policy evaluation, early resource release |

Executable kernel, schedule, storage and representation definitions have one
shared IR owner. Its coordinated construction API owns scoped identities and
child import; consumers cannot mutate closed artifacts or rebrand handles.
Refinement constructs those artifacts without timing. The independent estimator
reads them through a backend vocabulary contract that requires no native service.
Native formation/reflection uses a separate service contract and explicit live
context; immutable Metal target facts contain no device handle. The preparation
orchestrator retains the native-before-planning order above.

Only the checker and the validated bundle decoder construct a checked
module. Only the module constructs a logical entry. Only preparation owns the
paired realization registry; planning constructs a handle-free policy after
proving coverage, and exact materialization alone constructs a prepared kernel.

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

Catalog discovery only enumerates unopened physical devices. Opening a Metal
device creates the production service/queue immediately and derives the
device legality description. Hardware characterization is a separate,
analytical-evaluator dependency: a fixed, versioned probe manifest is compiled
once by a narrow native adapter, acquired as one aggregate raw-observation
bundle, and interpreted by a pure certifier. Candidate-domain construction,
generic evaluation, planning, direct native calls, and runtime execution do not
depend on characterization.

The device description is assembled once from backend revision, hardware identity
and device-wide limits, driver and toolchain versions, dtype support, the
numerical environment, and the static capability registry. It contains no
fact whose truth depends on a particular compiled function or pipeline.

Every numeric performance fact is either queried, derived by a sound documented
physical rule, or measured by a fixed primitive probe on the exact opened
device. The target-closed Metal cost program owns the finite fact vocabulary.
The renderer and analytical demand traversal consume that same program; neither
may reconstruct performance-bearing work from a broad semantic class. Measured
facts bind their raw observations, probe and method identity, endpoint,
environment, uncertainty, model version, and complete target identity into the
certified-profile identity. A separate stable compatibility identity contains
only legality/codegen facts and keys native artifacts; timing evidence does not
invalidate reusable native code. Prepared selection is never reused under a
different evaluation identity. Candidate implementations are never benchmarked
to create analytical facts. There are no calibrated corrections, fitted
candidate curves, arbitrary weights, guessed defaults, copied values from
similar hardware, nominal-peak shortcuts, or unknowns represented as zero.

Characterization constructs a complete immutable profile or no profile. Its
successful type has an infallible, exhaustive fact projection and contains no
optional required parameter in the installed service vocabulary. Acquisition
failures are aggregated across the fixed batch; certification is pure and
replayable from the retained raw bundle. `ExecutionProfileParts<T>` owns the
exact `Arc<DeviceDescription<T>>` from which its observations were acquired,
and `AnalyticalEvaluationContext<T>` consumes those bound parts with one model
definition. A concrete analytical evaluator cannot be installed until this
profile exists. The production profile boundary is structurally closed for the
built-in service vocabularies, but its physical formulas and evidence remain
unqualified until the independent estimator and characterization gates pass.

Kernel-affecting decisions are fixed before native formation. Compilation
produces an unusable `NativeKernelCandidate`; reconciliation consumes it and
authoritative reflection to construct `NativeKernel`. Its contract records
the actual ABI, launch domain, pipeline/function limits, static local memory,
register and spill usage where exposed, cooperative requirements, numerical
mode, service footprint, and compatibility identity. Unknown legality or
selection facts are not represented as zero and prevent admission of that
native implementation. Native compilation is complete before
`CandidateDomain` is sealed and is absent from evaluation and planning.

Every emitted command, physical primitive, memory relation, synchronization
operation, and intrinsic is visible in the target-closed cost program or in the
closed workflow lifecycle model. The Metal physical vocabulary includes every
finite execution regime required by its formulas; it has no optional, default,
catch-all, or unassessed branch. A backend that cannot close the program or
construct all of its required facts cannot install the analytical evaluator.
Capabilities are typed
intrinsic families; a backend advertises a signature only when the same
registration provides its typed lowering, resource rules, and native
emission. Registration is sealed at compiler initialization; an inconsistent
registry is a startup panic. Native compilation is forbidden from returning
an unsupported-capability or resource result for anything the profile
represents.

Primitive measurements retain their workload identity, timer resolution, raw
observations, ordering, digest, endpoint, environmental controls, compilation
time, execution time, and total acquisition time. Qualification freezes the
profile before measuring separate held-out candidate families. Qualification
observations never feed back into the profile or evaluator. Accuracy and
ranking criteria are versioned and selected before held-out evaluation from
measurement noise and candidate decision sensitivity; the architecture does
not prescribe fixed percentage thresholds in advance.

## Implementations

Refinement factories, portable and backend-specific, receive a semantic
function, pure target rules, the shared arena, the precision policy, and
core-owned builders. A factory may decline before construction; once
construction begins it returns a closed candidate family or a real preparation
error. Calls are resolved during construction: every applicable child family is
spliced under an explicit finite decision, composing constraints, lifetimes,
transfers, provenance, and effect ordering. No call survives into a schedule.
`RefinedCandidateFamilies` owns one universal family, optional optimized
families, their finite axes, and an honest construction report. It owns neither
performance models nor native handles. Structural identity excludes timing
facts; the later evaluation identity invalidates predictions.

Kernel construction requires only the backend intrinsic vocabulary and its
numerical/resource rules. Native compilation and execution services are separate
requirements. Prediction consumes the closed IR and a read-only execution model;
its service algebra has no compiler or device dependency. Native realization,
prediction, and solver admission remain distinct operations in preparation.
The current native-before-planning lifecycle remains required above.

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

Modeled duration is the result of the target-closed physical execution model
over the same structured schedule, exact operation program, allocation
topology, geometry, memory/address relations, path/cohort facts, native
realization bounds, workflow lifecycle, and certified device profile as
execution. Construction accounts for dynamic launch multiplicity,
dependencies, issue resources, concurrency and residency, cache and memory
transactions, overlap, barriers, atomics, submission, synchronization and
completion. A factory cannot assign or omit duration. Proxy lexicographic
counters, empirical candidate calibration, arbitrary weights, hard-coded
timing guesses, and nominal peak formulas are forbidden. A missing regime
prevents cost-program or profile construction; it cannot become a successful
partial estimate. The model propagates correlated evidence and uncertainty and
never describes a prediction as physical proof. Data-dependent control,
addressing or contention uses a sound all-path relation or rejects closure; the
compiler never invents branch probabilities, cache-hit rates, retry counts, or
expected input distributions.

Qualification criteria are selected and frozen before held-out evaluation.
They must establish useful candidate ordering for the declared domain and
bound the cases where modeled differences cannot justify an ordering. Held-out
measurements validate the model and its uncertainty; they never fit correction
coefficients or candidate-specific behavior.
Metadata such as constants, views, and allocation
declarations cannot form launch boundaries, and structured control stays
within a launch unless a real execution or synchronization boundary requires
otherwise.

Coverage is constructional. `CandidateDomain` requires one universal family
whose type admits no residual choice, whose numerical transfer is admissible,
and whose legality is total over the independently derived invocation domain.
Optimized families have a different type and cannot impersonate it. Refinement,
evaluation, and planning own distinct budgets and typed coverage reports.
Budget exhaustion preserves the complete domain already constructed and the
universal prepared policy, while reporting exactly which search scope was not
exhausted. It never turns a partially evaluated domain into success. Consumers
supply no envelopes, buckets, classes, or expected shapes.

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

Runtime execution is workflow-based. Runtime state is generic only over target
family `T` and native executor `E`. The native compiler is a preparation-local
service constrained by `NativeCompiler<T, Handle = E::Handle>` and does not
enter prepared or runtime types; the analytical model is erased inside
`AnalyticalEvaluationContext<T>`. `WorkflowGraphDraft::bind` derives all
inter-call hazards from semantic event manifests, evaluates prepared policies,
closes output descriptors, and retains unresolved may-alias relationships as
binding obligations. `BoundWorkflowGraph::admit` atomically acquires one
whole-graph reservation set and constructs `AdmittedRun`. Only that owned value
can submit; submission constructs `SubmittedRun`, and terminal completion
releases resources.
`execute_variant` is the sole constructor of `ExecutionEnvironment`; runtime
cannot inspect a variant's schedule, enumerate its native kernels, or construct
an environment. A backend submission receives only the selected native handle
through `ExecutionEnvironment::kernel_handle` while executing a closed command.
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
- Device descriptions contain no pipeline-specific fact; concrete Metal and CUDA
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
