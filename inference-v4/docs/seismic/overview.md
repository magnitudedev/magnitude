# Seismic

Seismic is a general-purpose kernel language, compiler, runtime, and standard library exposed as a
Rust library. Source describes logical computation, ownership, ordering, and semantic operations.
The compiler owns materialization, vectorization, tiling, placement, allocation, launch geometry,
scheduling, target selection, and native emission.

Every callable function has one portable body that defines its meaning. Portable implementations
and applicable backend lowerings are alternatives in one plan space; neither is a fallback. Kernel
source never contains model topology, hardware names, workload buckets, physical storage classes,
or launch policy.

## Public lifecycle

```text
.seismic source
    -> seismic-build: checked bundle + typed Rust bindings
    -> DeviceCatalog::discover
    -> open device and acquire its complete target profile
    -> generated entry for_device(device, precision policy)
    -> PreparedKernel covering the full inferred target domain
    -> generated Args call with tensors and ordinary semantic parameters
    -> validated variant selection and native execution
```

Consumers import the public `seismic` API and generated bindings. Tensors carry their device,
representation, shape, and layout. Consumers do not provide specialization domains, workload
envelopes, shape buckets, expected dimensions, tuning grids, physical buffers, binding indices,
or compiler artifacts.

Opening a device is the profile boundary. Discovery is cheap enumeration; opening creates the real
execution service, queries capabilities and limits, runs the fixed primitive probes required for
that exact target, and returns a usable device only after its profile is complete. A prepared
kernel is bound to that opened target and its precision policy.

## One compiler path

```text
CheckedModule -> LogicalEntry -> ImplementationDraft<B>
              -> NativeKernelCandidate<B> -> NativeKernel<B>
              -> PlanSpace<B> -> FrozenPlan<B> -> ExecutableVariant<B>
              -> PreparedKernel<B> -> PreparedWorkflow<B>
              -> AdmittedWorkflowRun<B> -> Execution<B> -> Completion<B>
```

This is the only production path. Structural drafts are native-compiled and reconciled before
they can enter `PlanSpace`; the required universal implementation closes first and optional
templates consume one total preparation budget. Each admitted alternative contains its executable
schedule, typed kernels, native contracts, allocation topology, resource constraints, numerical
transfer, and profile-derived duration. Exact symbolic planning chooses and freezes alternatives while retaining
runtime dimensions as symbols, then builds a non-empty variant portfolio whose guards provably
cover the full target-representable semantic domain.

Frozen-plan translation consumes already closed native kernels without compiling or redesigning
them. Production model execution composes prepared kernels into a dependency-closed workflow;
admission selects variants and reserves all resources atomically. Runtime validates public
invocations and executes typed native schedules; it does not retry compilation, repair a plan,
choose a fallback, or rediscover compiler legality.

## Language and precision guarantees

Owned tensors move, shared borrows may overlap, and mutable borrows are exclusive. Ordered loops
may carry state; `parallel for` asserts independent logical iterations and admits only disjoint
writes or explicit portable atomics. Materialization required by an implementation is inserted by
the compiler rather than authored as a source workaround.

The first applicable portable body defines reference operation order, casts, rounding, and
exceptional-value behavior. Exact, bounded, and unconstrained precision policies determine which
derived implementation transfers are admissible before selection. Fast math, contraction,
reassociation, approximate operations, reduced precision, and flush-to-zero are never implicit
backend defaults.

## Responsibility boundaries

| Participant | Owns |
| --- | --- |
| Kernel/library author | Logical algorithms, values, borrows, semantic control, portable bodies, and explicit target capability use |
| Checker and semantic registry | Types, shapes, effects, ownership, capabilities, canonical semantics, checked identities, and inferred call schema/domain |
| Compiler | Complete implementation construction, exact constraints, storage and resources, numerical assessment, duration modeling, solving, freezing, and exact portfolio coverage |
| Backend | Exact-device profile acquisition, capability implementations, native emission, and execution service |
| Runtime | Public invocation validation, deterministic variant selection, allocation, binding, submission, completion, and real execution failures |
| Host application | Model or application topology, tensors, semantic parameters, device choice, precision policy, and session orchestration |

Magnitude's inference engine is one host application. Model layers, cache policy, sampling, and
session behavior remain engine responsibilities; they do not enter Seismic's compiler or public
kernel abstractions.

## Outcomes

A valid source entry either prepares a fully covered executable portfolio or returns a typed
source, target, or preparation error before execution. After a generated call passes invocation
validation, execution can still report data-dependent checks, allocation/submission/synchronization
failures, or device loss. It cannot report disagreement between compiler phases.

The durable sources of truth are [Seismic compilation](../../../design/inference/seismic-compilation.md),
[language and capabilities](../../../design/inference/seismic-language-and-capabilities.md),
[numerical precision](../../../design/inference/seismic-numerical-precision.md), and
[structured solving](../../../design/inference/solver.md). The concise
[compiler overview](compiler.md) describes the artifact boundaries in more detail.
