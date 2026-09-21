# Seismic compiler

Seismic has one production artifact progression:

```text
CheckedModule -> LogicalEntry -> ImplementationDraft<B>
              -> NativeKernelCandidate<B> -> NativeKernel<B>
              -> PlanSpace<B> -> FrozenPlan<B> -> ExecutableVariant<B>
              -> PreparedKernel<B> -> PreparedWorkflow<B>
              -> AdmittedWorkflowRun<B> -> Execution<B> -> Completion<B>
```

Each artifact removes freedom from the preceding one. There is no separate realization layer,
independently sealed strategy/dataflow/placement pipeline, native plan mirror, retry compiler, or
runtime fallback.

## Artifact ownership

| Artifact | Authority |
| --- | --- |
| `CheckedModule` | Checked source semantics, types, effects, canonical portable bodies, lowering declarations, capabilities, and stable source identities |
| `LogicalEntry` | One monomorphized entry, its generated call schema, inferred semantic domain, canonical operation graph, provenance, and typed expression arena |
| `DeviceContract<B>` | Device-wide compatibility, capabilities, limits, toolchain, and numerical environment; never function-specific facts or measured rates |
| `NativeKernel<B>` | One reconciled native function with exact ABI, launch domain, reflected resources, numerical mode, and artifact identity |
| `ExecutionProfile<B>` | Measured service definitions, uncertainty, qualification, and per-open performance identity; never legality |
| `PlanSpace<B>` | Exactly one native-closed universal implementation, optional native-closed optimized implementations, and one exact constraint model |
| `FrozenPlan<B>` | One fixed physical choice with symbolic invocation dimensions, guard, layouts, allocation topology, typed kernels, structured schedule, numerical assessment, and duration expression |
| `ExecutableVariant<B>` | The directly emitted native schedule plus guard, duration, layout, binding, identity, and numerical metadata needed for execution |
| `PreparedKernel<B>` | The call schema, target domain, and a non-empty portfolio proven to cover that domain |
| `PreparedWorkflow<B>` | Dependency-closed multi-kernel topology and symbolic access hazards |
| `AdmittedWorkflowRun<B>` | Bound invocations, selected variants, retained resources, and one atomic reservation set |

Fields establishing validity are private. Checked modules come only from source checking or a
validated checked bundle. Implementations close through core-owned typed builders. Prepared
kernels come only from the portfolio builder after exact coverage is proved.

## Construction and planning

A portable function body and every applicable target lowering contribute ordinary implementation
alternatives. Factories first produce structural `ImplementationDraft`s. Before `PlanSpace`, the
compiler closes the universal draft into reconciled native kernels, then admits optional optimized
native templates under one total preparation budget. Each admitted alternative owns the actual
schedule, typed kernels, native contracts, transfers, global and launch-local allocation topology,
finite decisions, hard constraints, numerical transfer, and modeled duration. Calls are resolved
and child alternatives are spliced during construction; executable schedules contain no unresolved
calls.

One hash-consed typed expression DAG drives constraints, partial evaluation, variant guards,
layout, geometry, allocation sizes, numerical bounds, and duration. The solver represents its
Boolean structure exactly. Invocation dimensions remain symbolic while compile-time decisions
have finite explicit domains. Target resource limits and numerical admissibility are hard
constraints, never post-selection checks.

The duration objective is derived from the same schedule, operation classes, dependencies,
allocation topology, geometry, and target services that execute. It accounts for launch
multiplicity, concurrency and residency, transactions, synchronization, and overlap. Physical
service facts are either soundly derived from documented target facts or measured by fixed
backend-owned probes on the exact opened device, with uncertainty and provenance retained. Proxy
operation counts, fitted coefficients, guessed launch costs, and nominal-peak formulas are not
valid objectives.

Portfolio construction freezes feasible assignments whose native kernels are already closed;
executable translation performs no compilation and cannot discover a planning fact. Coverage is
constructional through the required universal member. Optional optimization may stop when the
shared solver/template/compile/code/variant/metadata budget expires while retaining full coverage.
Invocation selection is deterministic by
`(upper_duration, stable_variant_identity)` among matching variants.

## Target, native, workflow, and runtime boundaries

Device discovery only enumerates inexpensive descriptors. Opening a device creates the production
service and queues, queries authoritative facts, runs the fixed primitive probe suite, and binds a
complete `DeviceContract` and `ExecutionProfile` to that service before exposing a usable device. Preparation never pairs
an independently acquired profile with a later-opened service.

Native formation consumes closed kernel templates before `PlanSpace`, creates an unusable
candidate, and reconciles authoritative reflection into `NativeKernel`. Frozen-plan translation
only binds one already-closed native schedule. Unsupported capability, invalid geometry, excessive
local storage, or a missing binding cannot first be discovered below planning; native formation
reports only genuine
toolchain, device, cache, malformed-output, or compiler-resource failures.

Workflow closing derives access hazards from the same checked semantic events used by compilation.
Admission selects variants, reserves all resources atomically, and retains them through completion.
Runtime validates generated call arguments, evaluates expression roots,
allocates and binds global storage, executes typed commands, and reports data-dependent or external
device failures. It does not infer placement, reconcile kinds, repair schedules, clamp copies,
recompile calls, or interpret portable code as fallback.

## Precision and failures

The canonical portable body defines reference numerical behavior. Every implementation derives a
whole-entry numerical transfer from its actual operations and ordering. The requested precision
policy participates in solver admissibility, preparation identity, and the executable assessment;
native emission cannot introduce an unassessed relaxation.

Typed errors describe source, checked-bundle, target, preparation, invocation, and execution
conditions caused by the outside world. A contradiction in a privately constructed compiler
artifact is a bug and may panic only at the small local invariant boundaries defined by the durable
compilation contract. No defect taxonomy or phase-disagreement result is part of the architecture.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `seismic-lang` | Parsing, checking, semantic registry, checked bundles, logical entries, call schemas, semantic domains, and the shared expression DAG |
| `magnitude-solver` | Generic finite exact/neighborhood constraint search; no Seismic or hardware concepts |
| `seismic-compiler` | Typed kernel and schedule construction, native closure, storage topology, implementation families, solver adaptation, numerics, freezing, coverage, and prepared portfolios |
| `seismic-cpu`, `seismic-metal`, `seismic-cuda` | Device/profile acquisition, capabilities, implementation registrations, native compilation, and execution services for one backend |
| `seismic-runtime` | Generic opened-device, tensor, preparation-cache, selection, and execution machinery; no planning |
| `seismic` | Stable consumer-facing API composition |
| `seismic-build` | Build-time source checking, checked-bundle emission, and typed Rust binding generation |

The durable sources of truth are [Seismic compilation](../../../design/inference/seismic-compilation.md),
[language and capabilities](../../../design/inference/seismic-language-and-capabilities.md),
[numerical precision](../../../design/inference/seismic-numerical-precision.md), and
[structured solving](../../../design/inference/solver.md).
