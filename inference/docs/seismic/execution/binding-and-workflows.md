# Binding and workflows

The entry-owned invocation contract binds actual arguments to a chosen executable and produces a complete execution node. Workflow construction composes such nodes into ordered resource use.

| Input | Output |
| --- | --- |
| Prepared policy or applicable trial candidate, actual arguments and entry contract | Bound invocation containing arguments, executable, outputs, accesses, scalar publications and resource requirements |
| Complete bound invocations and requested result relationships | Workflow with dependencies and resource lifetimes |

## One binding operation

The binder derives dimensions and scalar values from the actual arguments, checks their original schema equations and definedness, establishes device and representation compatibility, and checks alias/access rights. It evaluates the prepared selector using the resulting invocation metadata and binds the selected executable through the same contract.

Outputs, mutable-state effects, ABI words, resource expressions and publication destinations are constructed from that actual selection and those arguments. Callers cannot pass separately reconstructed pieces and rely on a later planner to decide whether they belong together.

Device identity comes from the actual resources and prepared target. A requested descriptor cannot overwrite it. Index/range values retain their integer/natural sorts and actual endpoints. Invocation-known scalars keep their semantic expressions even if transported through execution slots.

Ordinary binding consumes a prepared policy; trial binding consumes an already chosen applicable candidate. They converge on the same candidate-to-arguments operation. Trial preparation is not another implementation of dimension inference, alias rules or output allocation.

## Workflow composition

Each node exposes its real reads, writes, logical allocation identities, result availability and resource requirements. Workflow construction derives inter-call hazards and lifetimes from those facts. It can order or overlap complete nodes according to their contracts; it cannot alter the candidate's internal physical decisions.

A host port (`&host tensor`, `&host mut tensor`) is one of those facts: the host reads, or rewrites, the tensor once it sees the launch, so the node is ordered as that access although the device neither reads nor writes it. It takes no argument slot: the kernel never accesses it.

Standalone calls, native batches and standalone measurement points can prepare an awaited entry only where the backend parks in kernels. Otherwise the caller must use a graph with an explicit park binding; standalone preparation does not infer that words were answered already.

A timeline has a role in its type, which says who raises it and who waits on it: a `Gate` the host raises and wait nodes hold device work on, a `Reached` that signal nodes raise and the host waits on, or a load's `Parks`. Each handle has only its role's operations and each node's port binds only its role, so neither side waits on what only it raises; on Metal, CUDA and Vulkan each side is its own event, word or semaphore, and such a wait would never return.

A call that parks (an awaited port, `&await`) follows exactly one park node (`timeline::Park`), and a park node precedes only such a call; the draft refuses either alone. Each run binds the park to the load's park timeline (`Timeline<Parks>`, `set_park`), or as answered already (`set_answered`: the awaited words hold the answer, and the park binds nothing), and the backend realizes it (`BackendName::parks_in_kernels`): where the devices see host writes while work runs (Metal, CUDA, CPU) the call's kernel waits for the host's words itself and the park binds no operation, so the submission is what it would be without it; on Vulkan, whose running work neither sees a host write nor publishes its own before it ends, the park is a submission boundary. There the runtime owns the timeline's values: a run that encodes a launch of the parked call takes the next value `v` when it is queued, which its outputs report (`NativeGraphOutputs::park`); the timeline reaches `v` once the work before it completes, its writes visible to the host, and the call's submission waits for `v + 1`, which the host signals once its answer is written (`Timeline::answer`). A run whose launch conditions leave the call out takes no value and does not stop. A run's launches are decided once, by the host-written condition words read when it is queued. A written grid's words gate no launch (`checked::NativeLaunch::grid`): its launch stays encoded (a park's value is still taken), and an empty grid runs nothing by itself (an indirect dispatch; on CUDA a loop of the device's resident blocks over no blocks).

Views remain connected to their underlying logical storage for hazards. A write advances contents/version without changing that identity. Independent copied values have independent storage. Shared storage reuse follows actual last-use dependencies, not just the order nodes were inserted into a list.

Future tensor results can connect nodes when their shape/representation metadata is already known and dependencies preserve production before use. Future scalar contents cannot silently act as current binding metadata.

```text
device-produced scalar
          |
          +-- use inside selected execution
          |      physical producer/completion dependency
          |
          +-- use as later public invocation metadata
                 explicit completion/read boundary
                            |
                            v
                 bind later invocation with actual value
```

This boundary is semantic availability. A broader device-control workflow can be supported only when represented as such in the physical/execution contract; it cannot be approximated by evaluating an unavailable host guard.

## Failure and ownership

Invalid shapes, representations, scalar domains, aliases or incompatible devices fail binding before submission. Whole-workflow [admission](resources-and-lifetimes.md) handles real capacity and access conflicts. A workflow node cannot omit its resources or result bindings and ask admission to reconstruct them.

Bound arguments remain owned or borrowed with rights that cover execution. An asynchronous submission must retain those rights until terminal completion. Dropping a caller-facing handle is not permission to release storage still reachable by native work.

