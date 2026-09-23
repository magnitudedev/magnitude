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

