# Execution

Execution binds actual invocations to prepared implementations, derives dependencies across calls, acquires resources and retains them until the native work finishes.

The compiler has already chosen and established the implementation contract. Runtime resolves actual arguments and resource availability; it does not complete missing compiler construction.

## Lifecycle

```text
PreparedKernel + actual arguments
                 |
                 v
        entry-owned binding
                 |
                 v
         complete BoundInvocation
                 |
                 v
  workflow: dependencies, accesses and lifetimes
                 |
                 v
    admission: acquire capacity and access
                 |
                 v
            AdmittedRun
                 |
             native issue
                 v
            SubmittedRun
                 |
         actual terminal completion
                 v
       results / failure + resource release
```

These states represent real ownership changes. A bound invocation owns its argument/executable relationship. An admitted run owns acquired capacity/access. A submitted run owns potentially live native work. They are not labels attached after separate validation.

## Responsibility boundaries

| Stage | Resolves | Must already be fixed |
| --- | --- | --- |
| [Binding](binding-and-workflows.md) | Actual dimensions/scalars, devices, representations, aliases, selected candidate and result destinations. | Entry schema/domain, candidate applicability and native contract. |
| Workflow construction | Inter-call hazards, ordering and combined resource lifetimes. | Each node's complete execution and access contract. |
| [Admission](resources-and-lifetimes.md) | Real capacity, access rights, resource leases and planned execution bindings. | Entire resource plan, including any scheduled acquisition/release actions. |
| [Backend execution](backends.md) | Native commands, completion and external errors. | Operations, control, numerical modes, resource lifecycle and selection. |

Argument errors, device mismatch, unavailable capacity and external native failures are real boundary outcomes. Missing producers, invocation guards requiring late scalars, detached resource tables and unsupported instructions inside an admitted executable are compiler defects. Runtime must not repair them with fallback compilation, empirical qualification or silent domain restrictions.

## Ordinary calls and trials

Feedback trials and ordinary calls use the same entry-owned invocation binding, workflow and execution mechanisms. A trial supplies private state and a measurement endpoint. It does not invent a fake policy or bypass ownership because it is “only testing.”

Device-produced scalars can control operations inside an already selected executable when the physical IR represents their availability. Using one to select a later public invocation requires an explicit completion/read boundary before that invocation is bound.

## Persistence

Several different lifetimes appear here. A loop carry persists between iterations; model state persists between invocations; a runtime pool retains native allocations for reuse; a compiled cache can persist artifacts on disk. None implies the others.

Runtime pools are an allocation-management mechanism. Their leases must still describe actual capacity and exclusive/shared access, and reuse waits for the last native reader or writer. See [resources and lifetimes](resources-and-lifetimes.md).

An explicitly selected authored native implementation uses the separate [direct-native contract](../language/functions-and-capabilities.md#explicit-direct-native-execution). It still requires sound binding and native lifetime ownership, but has no compiler-generated portfolio or numerical applicability guarantee.

