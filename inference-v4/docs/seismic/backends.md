# Seismic backends and emission

A backend supplies concrete implementations of Seismic operations, their resource
mappings, and emission into its target language or instruction representation. CPU,
CUDA, Metal, and Vulkan satisfy the same [compiler contract](compiler.md) through different
hardware mechanisms.

## Backend boundary

Backend lowering applies Seismic `lower` definitions to produce legal backend
implementations in Lowered IR. It can specialize on data shapes and types and use
explicit hardware queries/capability constraints. Performance choices are exposed to
[tuning](tuning.md), rather than resolved through device-name dispatch or hidden
hardware-shape preferences.

Physical pieces, intrinsic geometry, and hardware queries remain within backend
implementation scope. Each admitted assignment implements the same portable
contract, including its permitted numerical variation. These implementation values
cannot become portable parameters, dimensions, or observable partition identities.

The intrinsic vocabulary and general decomposition facilities must let a structured
Seismic lowering express the desired implementations within the admitted backend
form. Optimized behavior must not depend on recognizing model names or private
emitter patterns. Unsupported mechanisms are explicit coverage gaps; a scalar
fallback does not establish optimized coverage.

Emission accepts Tuned IR and implements its selected execution. Native compilation then
produces the executable under a declared target/toolchain configuration. Neither stage
silently changes the selected implementation or searches alternatives.

## Implementation definitions

Each admitted implementation connects:

- The operation's numerical and effect semantics.
- Operand layouts, representations, shape constraints, and participation requirements.
- The introduced operations, storage, communication, and dependency structure.
- The resource model and hardware parameters needed to describe that structure.
- The emitted template or instruction mapping and permitted downstream changes.

Intrinsic contracts include operand layout and ownership, participating roles,
memory visibility, and progress requirements. Asynchronous operations distinguish
issue from completion, identify retained source/destination storage and define
which event or wait makes results available. Copy engines and compute collectives
remain explicit resource users rather than zero-cost overlap annotations.

These definitions are exhaustive over the admitted vocabulary. Adding a primitive
requires extending checking, resource semantics, and emission together. Behavior and
resource consequences are derived from that same implementation structure. Registration
callbacks, matching names, and companion cost tables cannot substitute for this
relationship.

Representation decoding, conversions, addressing, validity checks, math routines,
barriers, scratch handoffs, and ABI support are part of the implementation when
executed. An opaque library call is not assigned zero cost or treated as a single
hardware instruction. Its contract must cover its implementation and conditions.

## Backend mechanisms

| Backend | Required execution mechanisms and resource relationships |
| --- | --- |
| CPU | Scalar/vector covers and tails, register blocking, stack storage, memory/cache locality, worker assignment, instruction dependencies, ABI and math implementations. |
| CUDA | Thread/warp/block and capability-dependent cooperative mappings, scalar/vector/tensor covers, operand layouts, storage hierarchy, async movement/computation, participant roles, completion protocols and residency. |
| Metal | Lane/SIMD-group/threadgroup mappings, scalar/vector/matrix covers and tails, fragment ownership, private/threadgroup/device storage, staging, communication, synchronization and residency. |
| Vulkan | SPIR-V, subgroup/workgroup ownership, supported cooperative matrix covers, operand layouts, storage classes, visibility/barriers and device capability constraints. |

Feature availability and allocation granularities come from applicable target contracts.
Hardware extent queries are explicit. Backend-family mechanisms are modeled
structurally; hardware profiles supply target-specific parameters.

Backends share execution semantics, not an identical list of implementations.
Capabilities determine which alternatives exist. Hardware-managed CPU caches are
not interchangeable with explicitly allocated GPU shared storage. Synchronization
or persistent work protocols must establish forward progress for their actual
participant scope.

Coverage includes the constructs needed by standard kernels and their compositions:
indexing and views, dense and packed accesses, arithmetic/conversions, reductions,
matrix operations, gathers and routing, scans/state updates, and synchronization. The
exact admitted implementations and numerical domains are explicit. A scalar baseline
alone does not satisfy the required optimized backend coverage.

## Emission invariants

1. Every selected operation maps through its declared implementation.
2. Layouts, allocations, lifetimes, mappings, masks, synchronization, and launches
   agree with Tuned IR and the derived account.
3. Emission introduces no independent tiling, storage, reduction, fusion, or grouping policy.
4. Necessary compiler-generated work is represented and accounted before emission.
5. Source/operation origins remain traceable without asserting an unsupported
   one-to-one relationship between IR and final machine instructions.

Backend syntax details may be chosen mechanically when they do not change the modeled
execution. Any choice that changes relevant work or resources belongs to the admitted
implementation or the tuner's legal space.

## Native compiler behavior

PTX, Metal source, and SPIR-V are not final machine instruction streams. CPU code-generation
input also permits downstream lowering and optimization. Native compilation may fuse,
eliminate, introduce, schedule, allocate, or spill work.

A mapping contract must cover those behaviors through a justified exact mapping or an
explicitly modeled admissible envelope. Logical temporaries cannot be relabeled as
physical registers. A compiler flag or nominal instruction count does not establish
native resource behavior by itself.

Separate compiler-controlled ordering from hardware scheduling and cache behavior.
An ideal block placement or instruction interleaving used by analysis is not an
emitted decision unless a supported implementation can enforce it. The model and
qualification must retain that distinction. A large exact search over virtual
instructions does not qualify native spills, instruction selection or memory service.

Where the necessary correspondence cannot be established, the backend must change its
implementation, emission strategy, or control of downstream compilation. The compiler
cannot claim qualification by ignoring a knowable discrepancy. Restricting a diagnostic
form must not silently reduce required production coverage.

Toolchain version, target features, math/optimization settings, and relevant native
mapping assumptions participate in artifact and qualification identities.

## Qualification outside selection

Qualification establishes and challenges the implementation mappings and hardware models
using independently checked semantics, native inspection, architecture contracts, and
correctly conditioned measurements.

It includes:

- Numerical/effect agreement, including precision boundaries and exceptional cases.
- Memory safety, aliasing, partial domains, and collective participation.
- Accounting/emission agreement for storage, communication, and execution structure.
- Register/spill, instruction, and synchronization behavior within the mapping contract.
- Timing predictions and bottleneck explanations on representative and held-out compositions.

Qualification records actual device, toolchain, and operating conditions. Machine
nicknames are execution locations, not hardware identities or selection rules.
Supported platform composition follows [distribution](../distribution.md).

Compile only the selected realization in the normal compilation path. Artifact
inspection can establish conformance or withhold qualification. It does not feed a
hidden compile-and-try loop. Development-time findings revise contracts and invalidate
affected evidence explicitly.

Successful samples do not prove universal native mapping soundness. Physical bounds
covering an entire form require sound rules across that form, not merely inspection of
its selected winner. See [Accounting](accounting.md) for the distinction between model
results and applicable physical assumptions.

## Diagnostics

Tooling must connect suspicious performance or invalid behavior to the originating
operation, decision, mapping, and resource constraint. It should expose selected
layouts, communication, checks, launches, and native evidence where available. Users
must not reverse-engineer hidden emitter policies to write natural kernels.
