# TileLang IR portability

**Portable kernel IR expresses one backend-independent computation as legal,
capability-selected schedules that preserve explicit ownership and communication,
while TileLang realizes each schedule efficiently on Metal, CUDA, HIP and LLVM.**

This guide defines the portability discipline for kernel authors. It complements
[portable kernels](../kernels.md), which owns the architectural boundary,
[IR authoring rules](ir-rules.md), which governs precise construction, and
[kernel optimization](kernel-optimization.md), which governs performance evidence.

## What portability means

Portability is not one physical schedule for every machine. It is one semantic
operation with a bounded family of schedules expressed entirely through public
TileLang constructs. A target may select different physical tiles, thread counts,
reduction steps, staging, layouts and pipeline depths without changing the
operation's values, effects or numerical contract.

Performance portability means that every target can reach its appropriate fast
mechanism without embedding backend identity or native instructions in Ops. It
does not require CUDA warps, Metal SIMD-groups, AMD wavefronts and LLVM CPU vectors
to execute the same physical decomposition.

Use these terms consistently:

| Term | Meaning | Owner |
|---|---|---|
| Logical extent | Values the operation promises to produce or consume | Ops |
| Physical tile | Statically shaped storage and work presented to TileLang | Schedule in Ops |
| Valid extent | Runtime subset of a physical tile whose result is meaningful | Operation and primitive contract |
| Fragment layout | Distribution of logical tile elements among lanes and registers | TileLang inference and lowering |
| Schedule | Legal choices of tiling, ownership, movement, concurrency and staging | Ops, selected from target facts |
| Lowering | Target instructions, register representation and native code | TileLang |

A kernel is portable only when its correctness follows from these public
contracts. Successful compilation or execution on one backend is insufficient:
a permissive lowering can conceal an invalid ownership assumption that another
backend correctly rejects.

## The ownership boundary

```text
operation semantics and numerical obligations
                     ↓
portable schedule family and explicit communication
                     ↓
TileLang layout inference and legality checks
                     ↓
Metal / CUDA / HIP / LLVM lowering and instruction selection
```

Magnitude Ops owns the algorithm, tensor regions, schedule family, physical work
geometry, storage used for semantic communication, tail behavior and numerical
boundaries. Runtime-provided target facts may describe behavioral resources such
as subgroup geometry, available memory, supported dtypes, synchronization and
launch facilities.

TileLang owns the meaning of language primitives, fragment-layout inference,
legal instruction selection, register realization, backend lowering, generic
fallbacks and precise unsupported-target diagnostics. A missing implementation
of a portable primitive on one backend is a TileLang capability gap; it is not a
reason for Ops to reproduce backend lowering.

Neither kernels nor model code may branch on backend, vendor, device name or
native instruction identity. Ops must not maintain an instruction capability
table, inspect generated source to decide which IR to construct, emit backend
dialects, or depend on undocumented lowering accidents. If a schedule-relevant
fact is genuinely portable, it belongs in TileLang's behavioral target contract.

## Authoring rules

### Begin with a semantic contract

Define required outputs, valid inputs, state effects, aliasing, rounding points,
allowed error and tail behavior before selecting tiles. State whether reduction
order, reassociation, redundant work and intermediate materialization may change.
Every schedule in the family must implement this same contract.

Keep a backend-independent reference for correctness. A schedule must not acquire
different semantics merely because one target exposes a more convenient matrix
instruction, vector width or memory scope.

### Separate logical validity from physical geometry

Matrix and vector mechanisms commonly require static physical shapes while the
logical work has a runtime tail. Preserve both facts rather than shrinking an
instruction tile dynamically or rounding logical work to a vendor atom.

For a runtime-valid matrix prefix such as `valid_m`:

- the physical operand and accumulator tiles remain static and fully addressable;
- the valid extent is uniform across the participating collective and lies within
  the physical M extent;
- only the valid prefix is semantically produced;
- the inactive suffix is not consumed unless the contract initializes and
  preserves it independently; and
- loads, initialization and publication obey the same validity domain.

`valid_m` is not a general mask and does not describe arbitrary rows, K tails or
N tails. If a backend lacks its required lowering, implement the same primitive
contract in TileLang for that backend or reject it precisely. Do not reinterpret
the extent in the kernel.

Physical shapes must be legal schedule choices. For example, a narrow logical M
may use a larger physical matrix tile plus a valid prefix on one target and a
smaller physical tile on another. Those are two schedules for one operation, not
two backend-specific semantic kernels.

### Treat fragments as distributed values

A fragment is a logical tile distributed across participating threads and their
registers. It is not an ordinary array independently replicated in every thread.
The same logical index in two fragments does not imply that the same thread owns
both values.

`T.Parallel` is legal across multiple fragments only when one inferred loop
distribution can satisfy every access in the loop. Reads must be covered by the
participating ownership; fragment writes must agree exactly with the destination
owner. Do not use a parallel elementwise loop as an implicit shuffle between
incompatible producer and consumer layouts.

When ownership changes require values to cross threads, express a real
communication boundary. A shared-memory publish, synchronization and reload is
the general portable exchange. Use it only when communication is semantically
required or measurements justify that schedule: compatible layouts should retain
values in fragments and avoid the exchange. TileLang may lower a declared copy or
exchange to a cheaper target mechanism when its contract permits that realization.

### Use matrix primitives through their portable contract

The normal portable contraction keeps statically shaped operands in their
supported storage and accumulates into a fragment initialized according to the
operation. Retain that accumulator across the complete reduction and publish it
only when a consumer or program boundary requires publication.

Do not infer portability from a target extension that accepts a different result
scope or instruction geometry. Shared destinations, target-specific accumulator
forms and native matrix atoms are TileLang lowering concerns unless the public
language explicitly gives them backend-independent semantics.

All matrix dimensions required by the primitive remain construction-time facts.
Runtime occupancy is represented through the primitive's declared validity
mechanism or through an explicit legal tail schedule, never by dynamically
changing the physical fragment shape.

### Express data movement, not hoped-for representation

Use regions, layouts, `T.copy`, fragments and shared storage according to their
semantic contracts. A copy describes movement between storage domains and may be
layout-aware; it is not a promise that arbitrary incompatible fragments can be
redistributed without communication, nor that every backend uses a particular
asynchronous instruction.

Allocate storage by communication requirement:

| Requirement | Portable representation |
|---|---|
| One thread owns mutable scalar state | Local scalar or local array |
| A collective owns a distributed tile | Fragment |
| Threads exchange or jointly reuse values | Shared storage with required synchronization |
| Values cross device-kernel boundaries | Explicit global materialization |

Do not introduce shared staging merely to satisfy one backend's current lowering.
Conversely, do not remove a required exchange because another backend happened to
legalize an ownership mismatch. Storage scope follows the algorithm's ownership
and communication; its native realization follows the backend.

### Vary schedules through behavioral facts

A schedule family may vary:

- physical tile and reduction dimensions;
- threads and subgroup decomposition;
- vector widths and work per lane;
- direct fragment consumption versus explicit exchange;
- shared-memory footprint, buffering and pipeline depth; and
- full-tile, boundary and occupancy regimes.

Selection uses facts that describe the mechanism, not labels that identify the
machine. Schedule names likewise describe computation or data movement. The set
must remain small and explainable; it is not a shadow backend registry.

Target tuning searches legal degrees of freedom after semantic equivalence and
layout correctness are established. Avoid a universal lowest-common-denominator
schedule, but also avoid cloning the algorithm per backend. When no legal schedule
can express a useful target mechanism, first determine whether the deficiency is
in the public primitive contract, one backend's implementation, or the proposed
schedule.

## Backend obligations

Each TileLang backend must either lower a used portable primitive with its stated
semantics or report an actionable unsupported-target error. It must not silently
ignore validity, assume a fragment ownership that inference did not establish, or
select an instruction whose geometry makes the authored physical tile illegal.

An unavailable fast instruction does not by itself make a semantic operation
invalid. Where the portable contract permits a correct scalar, vector or generic
matrix fallback, TileLang should select it. Whether that fallback is acceptable
for production is then a performance decision, not a correctness ambiguity.

Backend-specific optimization belongs in TileLang when it changes instruction
selection, register encoding, subgroup realization, bank mapping, native memory
operations or source generation without changing the portable schedule contract.
It belongs in an Ops schedule when it changes portable work decomposition or data
movement using public behavioral capabilities.

## Qualification across targets

Qualification separates construction, correctness, lowering and performance.
Every retained kernel family must cover Metal, CUDA, HIP and LLVM in one of two
ways: a tested legal schedule for the declared domain, or a precise documented
unsupported capability when the operation fundamentally requires an absent
public primitive. A backend compilation accident is not an accepted exclusion.

For every applicable target and schedule regime:

1. construct and lower the real kernel through that backend's normal pipeline;
2. compare outputs and state effects with the independent semantic reference;
3. test empty work, unit extents, exact tile boundaries, every tail axis, and
   runtime-valid extents including zero and the full physical extent;
4. exercise producer/consumer layouts that can expose ownership incompatibility;
5. inspect inferred layouts, synchronization, guards and generated target code
   for the mechanism the schedule claims;
6. measure the complete formula after warm compilation, including exchanges,
   materializations and dispatch; and
7. qualify the enclosing prefill, decode or other production path on each target
   whose performance the change claims to preserve.

Correctness is required for every legal schedule. Maximum efficiency is evaluated
per target against the best correct applicable schedule, not against identical
parameters. A change that improves one backend is incomplete if it silently
degrades another; either recover that target through schedule selection or record
and explicitly accept the measured trade-off.

## Acceptance rule

A kernel family is IR-portable only when:

- its semantics and numerical boundaries are independent of backend identity;
- logical, physical and runtime-valid extents are distinct and complete;
- every fragment access has a valid inferred owner and every ownership change has
  explicit communication;
- all schedules use public TileLang constructs and behavioral target facts;
- backend-specific instructions and realization remain inside TileLang;
- every shape and tail selects a legal schedule or receives a precise rejection;
- correctness is demonstrated independently on every applicable backend; and
- performance evidence shows that each target can select an efficient schedule
  without weakening the shared semantic contract.

If any of these properties is unknown, portability is unqualified rather than
assumed from success on the current machine.
