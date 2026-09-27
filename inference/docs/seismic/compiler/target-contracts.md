# Target contracts

A target contract describes the native facilities the compiler can actually use. It separates hard legality and numerical behavior from performance prediction.

| Input | Output |
| --- | --- |
| Device/API queries, backend implementation, driver/toolchain facts and explicit deployment limits | Immutable effective target, typed operation signatures, limits, storage/argument ABI and numerical environment |

## Effective support

An operation is available only when supported by hardware, API/driver, toolchain and the Seismic backend. Unknown support must remain unknown until resolved; it is neither successful support nor known absence. A focused compile/pipeline probe may establish an external facility when queries are insufficient.

Typed signatures include every behavior-affecting descriptor: operands/results, shape and layout requirements, participant topology, storage access, numerical modes, synchronization, pending/completed state and progress assumptions. Opaque strings cannot carry unmodeled behavior into a supposedly complete physical contract.

Each exposed signature needs construction, emission, binding and resource handling. It also needs numerical semantics and either cost handling or an explicit analytical-model limitation. Missing cost knowledge may defer an analytical estimate; missing execution semantics cannot be delegated to the cost model.

`PhysicalDialect` is the single pure backend vocabulary for kernel operations and the migrated physical schedule requests. `TargetFamily` adds native ABI/reflection type bindings; native compiler and executor services remain separate. Extend this vocabulary with concrete implemented operations rather than empty extension variants, string commands or a second dialect trait.

A physical launch owns its common geometry once and its typed backend launch descriptor. Its ordinary context comes from the region. Construction resolves semantic participation requirements to that descriptor using target facts; native formation and execution consume the selected request. Reflected limits may restrict applicability but never choose or rewrite the request. The initial descriptors cover ordinary CPU/Metal launches and independent/cooperative CUDA launches.

The general compiler requires complete current-operation boundaries. Additional native optimizations become selectable in complete construction, emission, binding, execution and cost slices. Preserving today's modeled operations is mandatory; a model-limitation outcome accompanies the first actual admitted operation or composition without a model, not a blanket escape from existing cost handling.

## Hard limits and deployment scope

The target distinguishes external ABI limits, per-object native limits, aggregate available capacity and dispatch/participant limits. A single native allocation limit is not the maximum size of every logical internal value. Internal storage can use the common direct/segmented [storage realization](physical-ir.md#logical-storage).

Deployment facts also determine which execution structures belong in the candidate domain: queues, streams, contexts, cooperative facilities, native lifecycle actions and any actual bound on controller/program state. Finite device memory alone does not establish a finite bound on all possible host controllers or program descriptions.

The source's full legal invocation domain is retained within external representability. Lack of a physical recipe for some internal resource configuration is a construction gap, not another external input restriction.

## Artifact facts and performance facts

Native compilation can reveal artifact-specific requirements such as actual argument layout or resource usage. [Native formation](native-emission.md) reconciles these with the fixed physical contract before exposing an executable. They are not device-wide facts and cannot be installed into another candidate by sharing a name or ordinal.

An analytical profile models service rates, latency and contention. It is acquired for analytical evaluation; opening a device, feedback evaluation and direct native execution do not require it. The profile does not authorize an unsupported operation or change numerical semantics.

| Identity | Changes when |
| --- | --- |
| Execution compatibility | Native code, ABI or target requirements cease to be interchangeable. |
| Numerical environment | Rounding, contraction, subnormal or other numerical behavior changes. |
| Performance profile | Characterization or observation conditions change. |

These identities may share components but have different purposes. Reprofiling unchanged hardware behavior can invalidate cost predictions without changing semantic applicability. Hash equality indexes compatible data; it does not establish implementation correctness.
