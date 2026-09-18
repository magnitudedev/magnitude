# Seismic backends and emission

A backend supplies concrete implementations of Seismic operations, their resource
mappings, and emission into its target language or instruction representation.
CPU, CUDA, and Metal satisfy the same [compiler contract](compiler.md) through
different hardware mechanisms.

## Backend boundary

Backend lowering applies Seismic `lower` definitions to produce legal backend
implementations in Lowered IR. It can specialize on data shapes and types and use
explicit hardware queries/capability constraints. Performance choices are exposed
to [tuning](tuning.md), rather than resolved through device-name dispatch or hidden
hardware-shape preferences.

Emission accepts Tuned IR and implements its selected execution. Native compilation
then produces the executable under a declared target/toolchain configuration.
Neither stage silently changes the selected implementation or searches alternatives.

## Implementation definitions

Each admitted implementation connects:

- The operation's numerical and effect semantics.
- Operand layouts, representations, shape constraints, and participation requirements.
- The introduced operations, storage, communication, and dependency structure.
- The resource model and hardware parameters needed to describe that structure.
- The emitted template or instruction mapping and permitted downstream changes.

These definitions are exhaustive over the admitted vocabulary. Adding a primitive
requires extending checking, resource semantics, and emission together. Versioned
trusted rules establish any new proof authority; registration callbacks or matching
names cannot establish it.

Representation decoding, conversions, addressing, validity checks, math routines,
barriers, scratch handoffs, and ABI support are part of the implementation when
executed. An opaque library call is not assigned zero cost or treated as a single
hardware instruction. Its contract must cover its implementation and conditions.

## Backend mechanisms

| Backend | Required execution mechanisms and resource relationships |
| --- | --- |
| CPU | Scalar/vector instructions, register and stack storage, instruction dependencies, memory hierarchy, worker parallelism, ABI, and required math implementations. |
| CUDA | Thread/warp/block mappings, scalar/vector/tensor instructions where admitted, register/local/shared/device storage, transfers, barriers, launch dependencies, and residency limits. |
| Metal | Lane/SIMD-group/threadgroup mappings, scalar/vector/matrix operations where admitted, private/threadgroup/device storage, communication, barriers, launch dependencies, and residency limits. |

Feature availability and allocation granularities come from applicable target
contracts. Hardware extent queries are explicit. Backend-family mechanisms are
modeled structurally; hardware profiles supply target-specific parameters.

Coverage includes the constructs needed by standard kernels and their compositions:
indexing and views, dense and packed accesses, arithmetic/conversions, reductions,
matrix operations, gathers and routing, scans/state updates, and synchronization.
The exact admitted implementations and numerical domains are explicit. A scalar
baseline alone does not satisfy the required optimized backend coverage.

## Emission invariants

1. Every selected operation maps through its declared implementation.
2. Layouts, allocations, lifetimes, mappings, masks, synchronization, and launches
   agree with Tuned IR and the derived account.
3. Emission introduces no independent tiling, storage, reduction, fusion, or grouping policy.
4. Necessary compiler-generated work is represented and accounted before emission.
5. Source/operation origins remain traceable without asserting an unsupported
   one-to-one relationship between IR and final machine instructions.

Backend syntax details may be chosen mechanically when they do not change the
modeled execution. Any choice that changes relevant work or resources belongs to
the admitted implementation or the tuner's legal space.

## Native compiler behavior

PTX and Metal source are not final machine instruction streams. CPU code-generation
input also permits downstream lowering and optimization. Native compilation may
fuse, eliminate, introduce, schedule, allocate, or spill work.

A mapping contract must cover those behaviors through a justified exact mapping or
an explicitly modeled admissible envelope. Logical temporaries cannot be relabeled
as physical registers. A compiler flag or nominal instruction count does not establish
native resource behavior by itself.

Where the necessary correspondence cannot be established, the backend must change
its implementation, emission strategy, or control of downstream compilation. The
compiler cannot claim qualification by ignoring a knowable discrepancy. Restricting
a diagnostic form must not silently reduce required production coverage.

Toolchain version, target features, math/optimization settings, and relevant native
mapping assumptions participate in artifact and qualification identities.

## Qualification outside selection

Qualification establishes and challenges the implementation mappings and hardware
models using independently checked semantics, native inspection, architecture
contracts, and correctly conditioned measurements.

It includes:

- Numerical/effect agreement, including precision boundaries and exceptional cases.
- Memory safety, aliasing, partial domains, and collective participation.
- Accounting/emission agreement for storage, communication, and execution structure.
- Register/spill, instruction, and synchronization behavior within the mapping contract.
- Timing predictions and bottleneck explanations on representative and held-out compositions.

Qualify CPU and applicable GPU paths on the supported local targets and on Sparky,
M4 Pro 01, and M4 Pro 02. A machine nickname is an execution location, not a semantic
hardware identity or tuning rule. Record actual device, toolchain, and operating
conditions.

Compile only the selected realization in the normal compilation path. Artifact
inspection can establish conformance or withhold qualification. It does not feed a
hidden compile-and-try loop. Development-time findings revise contracts and
invalidate affected evidence explicitly.

Successful samples do not prove universal native mapping soundness. Physical bounds
covering an entire form require sound rules across that form, not merely inspection
of its selected winner. See [Accounting](accounting.md) for the distinction between
model evidence and admitted physical axioms.

## Diagnostics and conformance

Tooling must connect suspicious performance or invalid behavior to the originating
operation, decision, mapping, and resource constraint. It should expose selected
layouts, communication, checks, launches, and native evidence where available.
Users must not reverse-engineer hidden emitter policies to write natural kernels.

A backend conforms when its required mechanisms compose correctly, its emission
preserves Tuned IR, its mappings and hardware assumptions are qualified, and real
executions support the declared performance model. Faster isolated kernels do not
compensate for incorrect semantics or unmodeled enclosing costs.
