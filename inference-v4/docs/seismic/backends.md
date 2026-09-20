# Seismic backends

**A backend maps checked logical loops, ownership effects, and operations to physical execution on
one target, with hard limits, an estimate, and deterministic realization.** A backend never
searches or ranks.

This document names physical region, tile, staging, and publication nodes where they still exist in
the execution IR. They are compiler-owned migration internals, not Seismic source syntax.

## The `Backend` contract

Joint selection is generic over a backend that implements these hooks. Every hook
is a deterministic function of the program and the family, and for `realize`, of the
witness. Equal inputs give equal outputs.

| Hook | Returns | Obligations |
| --- | --- | --- |
| `target` | The target name family construction uses to include matching lowerings and backend-specific helpers alongside portable bodies | — |
| `estimate_model` | The identity of the estimate model behind the cost factors | Changes whenever the factors' meaning changes |
| `bind_structure` | The finite value domain of every site | Binds the prescribed structural mapping for every candidate body. A reachable structure with no mapping fails as *unsupported structural mapping*; it is never dropped. Domains are defined by a stated rule that does not consult the estimate. |
| `constraints` | Hard legality relations over site values, each guarded by the candidates that activate it, with a reason | Only real limits: capacity, alignment, participant counts. A guessed resource preference is not a constraint. A limit that no site can repair on a path selection cannot avoid is reported as *incompatible composition*. |
| `intervals` | Every legal contiguous interval of every sequence, with the child candidates it depends on and the site pairs it ties equal | Lists all legal intervals, including each supported singleton. No profitability filter. One prescribed realization per interval. |
| `factors` | Local additive cost terms, each with its activating candidates and intervals and the exact sites it reads | Derived from the same prescribed mapping `realize` applies. A term that cannot be derived fails; it is never zero. No term hides a child choice or an inner optimization. |
| `seed` | One constructive complete witness | Built from authored candidates, domains, and hard limits only. It must pass the joint audit. It is never a fallback implementation. |
| `realize` | One realized execution of the instantiated execution IR, or an error | Applies only fixed rules. Never returns alternatives. Rechecks hard limits on the realized execution. |

### No hidden search

Nothing named bind, derive, estimate, seed, or realize may choose among alternatives
by predicted profit. Where a target has a real alternative, it is either an
authored lowering (a candidate), a legal interval, or a site. Everything else is
one rule, written down below for Metal, applied identically in the estimate and in
realization.

### Failure classification

| Condition | Outcome |
| --- | --- |
| Structure with no mapping; primitives of another target; no domain value satisfies a site's requirements | Unsupported structural mapping |
| A hard capacity cannot hold on an unavoidable path; realized execution exceeds a limit | Incompatible composition |
| A quantity or estimate has no static derivation | Analysis unavailable |
| The family names something the backend cannot locate | Reconstruction defect |

## Metal mapping

### Structure

| Temporary physical IR structure | Metal realization |
| --- | --- |
| Root `parallel` region of the entry | One launch dispatched over its pieces, one SIMD group of 32 threads per piece |
| Root `ordered` or `pipeline` region | One single-piece launch with serial windows |
| Nested region | Serial loops inside its owner |
| `pipeline` | Synchronous, same participant: prepare then consume each window; ring depth one |
| `merge` | Canonical adjacent-pair recurrence inside the owner |
| Region result | Storage inside one owner; never across launches |
| Root stages | Successive launches |
| Run of invocation-scope serial statements | One single-thread launch |
| Matrix intrinsics | Operands in threadgroup memory, addressed at the matrix element type |

### Former decisions, now rules

| Subject | Rule |
| --- | --- |
| Dispatch | One SIMD group per piece. Pieces per threadgroup is `ceil(pieces / 65535)`, at least one. No widening, no compiler split, no pointwise partition beyond the instantiated pieces. |
| Load | Borrow where the borrow proof holds, else materialize: the greatest fixpoint of the proof starting from all-borrow. |
| Tile storage | Threadgroup memory for a matrix-intrinsic operand, for a tile of at least 32 elements whose elements an owned loop reads at coordinates other than its own (its element loops then write it cooperatively, one share per lane, and a barrier publishes it), or when cooperation admits no private placement. Otherwise thread-private: lane-distributed from 32 elements upward (one element per lane) when admitted, else replicated. Cooperative covers partition a tile by its capacity (element `e` belongs to lane `e mod 32`), so a runtime extent is shared among the lanes like a fixed one. |
| Fold ownership | Serial. SIMD-group collectives appear as authored intrinsics and in the collective reduction below. |
| Reduction | Lane-local when structurally available and the output exceeds one SIMD group; else collective (each lane folds its share, one SIMD-group reduction per output) when available; else ordered; else the first available algorithm. A collective requires `unordered=true` over a lane-distributed 32-bit input. This is a numerical effect: the containing implementation still needs evidence accepted by the entry precision policy. Every ordered reduction keeps authored ascending order and is bit-exact against the interpreter. |
| Allocation | Always a new slot. No reuse, hence no optional barrier. |
| Transfer | The widest exact packed vector the transfer admits. |
| Traversal | Unroll width one. |
| Bounds | A coordinate or device address that the emitter proves in range from ranges that hold by construction (piece coordinates, loop indices, owned coordinates, clamped or validated slice starts and lengths, the runtime extent a slice names) is emitted unchecked. Every other check stays (including every read of a packed plane), subject to the interval simplifier. A slice with the same symbolic start, end and parent extent as one evaluated in an enclosing scope is that slice; an `i32` local assigned by exactly one statement is its own symbolic value. |
| Submission | An unprofiled batch is committed in command buffers of 64 dispatches, in source order on one queue, so the device executes while the host encodes. Completion, errors and the status word are checked for the whole batch before return. |
| Floating point | Contraction off in emitted source; FMA appears only where authored. |

### Local half-precision storage

A `bf16` or `f16` value held in a thread-private or threadgroup array is stored as
`f32`. A write rounds to the logical type and then widens; a read narrows. Widening
is exact, so assignment and publication rounding are unchanged, and device buffers
keep their declared element type. The one exception is a matrix-intrinsic operand
tile, which keeps its native type because the intrinsics address it at the matrix
element type.

Reason: Apple's Metal compiler miscompiles thread-address-space `bfloat` arrays
(wrong values in contiguous runs at real model dimensions, with logically correct
source). `half` shares the code path. The rule is a single function of the dtype,
and declarations, byte quantities, the private-bytes limit, and every emitted
access follow it.

### Site domains

- **Width:** divisors `d` of the static extent with `d = 1`, `d = extent`, `d` pinned
  by an equality requirement, or `d = unit · 2^k` where `unit` is one or the unit of
  any multiple-of requirement on the site; then intersected with the requirements
  of the site's owning candidate.
- **Parts:** one and the powers of two up to `min(extent, 1024)`, intersected
  likewise.
- A domain above 24 values is thinned to 24 evenly spaced ordinals keeping both
  ends.
- A width over a domain with runtime bounds has no static piece count; its domain is
  `{1}`, the only width instantiation realizes there.

Divisor-only widths mean no tail piece exists. The domain rule defines the family
the solver searches; a model proof is a proof over these domains only.

### Native limits as solver constraints

| Limit | Constraint |
| --- | --- |
| Launch grid | Pieces of a root `parallel` launch ≤ 65535: one threadgroup per piece. Realization groups pieces per threadgroup by the largest launch of the whole execution, so a launch beyond one grid would multiply the threadgroup memory and thread count of its sibling launches, which no per-launch constraint can see. Every width domain holds wider values, so the bound removes no entry. |
| Threadgroup memory | Bytes of threadgroup-placed tiles × pieces sharing one threadgroup ≤ the device's threadgroup memory length. Applies also to a caller's tile that a callee places in threadgroup memory, by using it as a matrix operand or by reading it at foreign coordinates inside an owned loop. A matrix operand counts at its native element type, any other placed tile at the widened local type. The ledger recognizes foreign-coordinate reads as indexed reads inside `owned` loops whose indices are not exactly the loop's coordinates; a snapshot of external storage is taken as borrowed (no array); a tile whose element count depends on selected geometry counts as placed. `realize` rechecks the emitted kernel. |
| Private stack | Bytes of thread-private arrays one kernel declares, at their widened local type, summed over the candidate and its ancestors in the same launch ≤ 128 KiB |

The first two are queried from the device. Metal neither documents nor exposes the
thread stack; pipeline creation fails beyond it. The 128 KiB limit is half of the
measured linkable maximum on Apple M4 Max, leaving the rest to compiler temporaries
and spills. Realization rechecks the launch grid and the declared private bytes of
every launch. A device offering fewer than 32 threads per threadgroup is rejected
at construction.

### Intervals

Every unit is a singleton interval. Two fused realizations exist:

1. A contiguous run of two or more elementwise units of one block that crosses no
   completion: one element loop; only the last output and outputs referenced after
   the run survive as storage.
2. A contiguous run of root-level `parallel` region units with equal binder counts
   and equal domain extents, no `merge`, no region result rebinding, and no
   completion between them: one launch. Corresponding width sites are tied equal.

A `Call` unit is a singleton only. The family reports when a callee candidate's root
block consists solely of statement-position parallel regions
(`Family::root_regions`), but no interval spans a call yet: instantiation has no
realization that inlines the root regions of several callees into one launch, so
every exported composition entry still pays one launch per callee region.

### Estimate model

Identity: `metal-estimate-probe-calibrated-m4max-20260919-v3`. Additive, in
nanoseconds. Probe-calibrated on one Apple M4 Max; every other device inherits the
coefficients unqualified. No result is an execution upper bound.

**Factors.**

- Per candidate: one factor per root launch and one for the remaining body. Child
  calls carry their own factors under their own guards.
- Per selected interval: one launch overhead for a run of root regions; for
  elementwise runs, a write plus a read of every surviving output.
- Runtime extents are charged at their static upper bound. Branches are charged at
  the maximum of their arms.

**Span of one scope.**
`launches × launch + norm(max(compute, tile traffic) × pressure, bus traffic)`.

| Term | Rule |
| --- | --- |
| Compute | Lane operations at a per-lane rate plus ordered-visit bookkeeping, divided by the pieces that run concurrently (scalar work and matrix atoms saturate at different piece counts), plus matrix multiplies and matrix transfers. |
| Lane operations | Element operations on the critical path of one piece. An `owned` loop over every axis of a tile of at least 32 elements is charged its lane share, with a runtime extent at its static bound (v3; v2 charged such loops whole). Integer arithmetic over loop binders, constants, shape parameters, selected geometry and the participant index is coordinate computation and is not charged; integer operations on loaded data are. |
| Reduction | Mirrors the reduction rule exactly: replicated input, every lane folds all of it; lane-local, the lane share; collective, the lane share plus one collective per output; ordered over a distributed input, one shuffle per element. |
| Bus traffic | Distinct bytes of external storage only: a snapshot is charged once per distinct value of the binders its view mentions. Further visits of the same view are cache-served and cost the lane operations that consume them. Packed representations are charged at their exact fractional rate. |
| Borrowed snapshot | A `load` of external storage passed straight to a call is borrowed (load rule): no copy is charged, and its bus traffic is charged in the selected callee's body scope, where it overlaps the callee's compute. |
| Tile traffic | Bits moved through threadgroup or thread-private tile storage; it runs inside the lanes that compute, so it combines with compute by maximum. |
| Norm | Bus traffic overlaps compute imperfectly: the Euclidean norm is the larger term when one dominates and 41% above it when they are equal. |
| Pressure | `1 + private bytes per thread / 8 KiB` for the thread-private arrays the scope's kernel declares (threadgroup-placed tiles excluded), as far as the candidate's own chain determines them (a tile handed to a call is held whole by every lane; a snapshot of external storage is taken as borrowed; sibling occurrences inlined into the launch are not seen). |

**Coefficients and provenance** (Apple M4 Max, macOS 15, 2026-09-19; steady GPU
clocks: 200 invocations per command buffer, best of several).

| Coefficient | Value | Provenance |
| --- | --- | --- |
| Launch | 3 µs | Empty dispatch 2-3 µs of GPU time. Host encoding measured 0.3 µs per dispatch (0.2 ms for 661 dispatches) and overlaps device execution under the submission rule; it is not charged. v2 charged 8 µs. |
| Lane rate | 6.5e8 operations/s | Emitted packed `linear` scalar body: about 6 ns per counted element operation. A serial FMA chain runs 12.8 ns per dependent step; four independent chains 3.2 ns per step. |
| Visit | 20 ns | Joint fit with the lane rate. |
| Concurrent lanes | 14336 (448 SIMD groups) | Compute-bound probes (FMA chain, independent chains, `simd_sum` loop): 245 groups busy at 256 dispatched, 330-380 at 512, 460-495 at 1024, 500-530 at 4096. The model clamps hard at the midpoint of that soft saturation. |
| Collective | 16 lane operations | `simd_sum` loop: 24.6 ns per collective. |
| Shuffle | 24 lane operations | Ordered fold over a distributed 2560-element tile: 39 ns per element. |
| Matrix multiply, transfer | 12 ns, 70 ns | Joint fit on the staged 8x8 lowerings of packed `linear` (v1). |
| Matrix SIMD groups | 512 | Same kernels (v1). |
| Bus bandwidth | 495 GB/s | Emitted packet vector kernel reads 357 MB of distinct q4g64 weights in 0.72 ms. |
| Tile bandwidth | 900 GB/s | Threads streaming their threadgroup slice, 600-900 GB/s aggregate; a lower bound. |
| Private pressure | 8 KiB | Emitted 2x2-fragment packed `linear`, equal work at 1, 2 and 8 KiB of replicated accumulator per thread: 1.77, about 2.2 and 3.3-3.8 ms. |

**Measured, not yet modeled** (same device and date):

- Threadgroup memory limits concurrency like a pool of about 2 MiB: 64 SIMD groups run
  at once with 32 KiB each, 128 with 16 KiB, 256 with 8 KiB, about 500 with 4 KiB.
- `simdgroup_multiply_accumulate` takes 24 ns when each depends on the last and about
  10 ns each when eight independent accumulators interleave; a `simdgroup_load` of a
  contiguous 8x8 block is cheap, a strided one (row stride 256-512) costs about 30 ns; a
  `simdgroup_barrier` 58 ns. Beyond about 14 live 8x8 matrices a kernel spills (9x
  slower per atom).
- A lane streaming windows of several tensor rows alternately (window-major order over
  more than one output row per lane) reads device memory at 100-260 GB/s instead of
  about 500. The ordered packed `linear` body at 248320 outputs runs 1.4 ms with 32
  columns per piece and 3.8 ms with 128 or more; the estimate is flat across them.
- The 130 ns per element once attributed to column-major reads of a transposed device view
  (the value product of attention over history) was the unproven bounds check in that
  loop: a checked coordinate or read (branch plus a status store on the cold path) costs
  about 100 ns per iteration of a dependent FMA chain, a proven one nothing. The attention
  launch over 4 heads of 256 at history 250: 1954 µs as emitted before, 1048 µs with the
  score tile written cooperatively and the value product history-major, 197 µs with the
  checks proven (40 positions: 262, 224, 50 µs). Branch-free checks that fold failure into
  one flag written at kernel exit measured 720 µs, so proving is the rule and the helpers
  are unchanged.
- The interleaved lane cover (element `e` on lane `e mod 32`) makes every lane stride 32
  elements through device and tile storage, which Apple's compiler does not vectorize. One
  row of `rms_norm` at 2560 elements runs 12.4 µs as emitted and 3.0 µs with a blocked
  cover (lane `e / slots`, contiguous slots); interleaved runs of 8 or 16 elements 8.0 and
  7.4 µs. Not used: a lane-local reduction is admitted only under the interleaved cover
  (all elements of one output share a lane when the inner extent is a whole number of
  subgroups), every tile of one ownership group must share one cover (slots are addressed
  without translation), and a blocked cover by capacity leaves most lanes idle on a short
  runtime extent (attention scores at 40 of 256 positions: 8 busy lanes of 32).
- Several SIMD groups per threadgroup sharing staged operands (round-three probe, a
  hand-written kernel, not an emitted one: bf16 x q4g64, 9216 outputs of 2560 columns;
  one threadgroup owns `8·G` rows by 32 outputs; per 64-column window its `32·G` threads
  stage the activation block and decode the weight block once into threadgroup memory as
  f32, one barrier, then each SIMD group accumulates its four 8x8 fragments with eight
  activation loads and 32 weight loads). Throughput by `G` at 32 / 128 rows: 1 SIMD group
  0.58 / 0.62 TMAC/s, 2: 1.62 / 1.70, 4: 2.72 / 3.05, 8: 3.04 / 3.46. The emitted
  one-group-per-threadgroup lowerings run 1.1-1.95 TMAC/s on the same shapes. The 32 KiB
  threadgroup limit ends this design at 8 groups with f32 staging (16 groups need 57 KiB).
  The gain is the weight window decoded once per `8·G` rows instead of once per piece.
- Host encoding is 0.3 µs per dispatch. A decode step spends about 1.3 ms of wall time
  outside the submission: engine-side binding of 66 entry invocations and logits readback.

### Seed policy

- **Choices:** occurrences in pre-order. Each takes its first candidate, in family
  declaration order across applicable portable bodies, Metal lowerings, and Metal-specific
  helpers, under which every required site keeps a non-empty admissible domain and
  the remaining occurrences can be completed the same way; otherwise the next
  candidate, with backtracking.
- **Sites:** binders of a root `parallel` launch take the smallest admissible widths
  whose piece count does not exceed the SIMD groups the device runs concurrently,
  raising the last binder first. Every other width takes its largest admissible
  value. Parts take one. A violated memory limit lowers the highest site of the
  offending tiles; a violated grid limit raises the highest site of the launch.
- **Covers:** all singletons.

This is feasibility construction from device capacities. It is not a ranking and it
excludes nothing from search.

### Emission and native compilation

Emission consumes the realized execution and prints MSL. It decides nothing. Native
compilation happens once, for the selected witness. Register allocation, spilling,
and instruction scheduling belong to Apple's compiler and the hardware; they are
unmodeled effects ([Accounting](accounting.md)).

## CPU mapping

Target `cpu`. **Coverage is scalar**: every candidate body is realized through the shared
scalar realization (`seismic-compiler` `scalar_*`) and compiled by Cranelift. There are no
vector microkernel lowerings yet. Only portable bodies and `lower … for cpu` bodies are
candidates; a body that requires another target's primitives is never applicable.

### Structure

| Temporary physical IR structure | CPU realization |
| --- | --- |
| Root `parallel` region of the entry | One phase; its pieces are claimed one at a time by the worker threads of the device |
| Root `ordered` or `pipeline` region | One single-piece phase with serial windows |
| Nested region | Serial loops inside its owner |
| `pipeline` | Synchronous, same thread: prepare then consume each window |
| `merge` | Canonical adjacent-pair recurrence inside the owner |
| Region result | Scratch storage inside one owner; never across phases |
| Root stages | Successive phases, each run to completion |
| Run of invocation-scope serial statements | One single-item phase on the calling thread |
| Target intrinsics, atomic updates | Unsupported structural mapping |

Values that cross a phase boundary use invocation-owned retained storage, hidden from the
binding ABI.

### Former decisions, now rules

| Subject | Rule |
| --- | --- |
| Dispatch | One work item per piece. No widening, no compiler split. One worker per unit of host parallelism. |
| Load | Borrow where the borrow proof holds, else materialize: the greatest fixpoint of the proof starting from all-borrow (`seismic_compiler::scalar_load_rule`). No stride or ownership alternatives. |
| Storage | Every tile, region result and materialized snapshot is a new bump slot of the executing worker's scratch at its native element type. No reuse. |
| Reduction | Ascending order on the owning thread. This satisfies both the ordered and the `unordered=true` contract, so every reduction is bit-exact against the interpreter. |
| Traversal | One element per iteration. No unrolling, no vector transfer. |
| Floating point | No contraction; FMA only where authored. `rsqrt`, `exp`, `log`, `sin`, `cos` evaluate in binary64 and round once, as the interpreter does. `bf16`/`f16` results round to the logical type at every assignment and publication. |
| Divisors | A symbolic quotient or remainder divisor is a compile-time constant: instantiation runs after selection with every width concrete. A violation is an error naming the expression. |

### Site domains, intervals

The same family rule as Metal: structured divisor widths, power-of-two parts up to 1024,
thinning to 24 values, width `{1}` over a runtime-bounded domain. Intervals are every
singleton plus the two fused forms instantiation realizes (adjacent elementwise units;
adjacent identical-geometry root `parallel` regions with their widths tied equal).

### Limit

| Limit | Constraint |
| --- | --- |
| Worker scratch | Bytes of the tiles a candidate and its ancestors declare in one phase ≤ 64 MiB per worker. A snapshot of external storage counts as materialized, because the load rule is applied only by `realize`. Realization rechecks the scratch of every compiled phase, which also sees sibling calls. |

Each worker holds one scratch allocation that the phases of all kernels reuse, grown to the
largest phase it has run; 64 MiB is the capacity the device offers a phase.

### Estimate model

Identity: `cpu-estimate-unqualified-v0`. Additive, in nanoseconds. No coefficient is
measured; no result is an execution upper bound.

`span = phases × phase + pieces × piece / usable + operations / (rate × usable) + bytes / bandwidth`,
with `usable = min(pieces of the enclosing root parallel phase, workers)`.

| Coefficient | Value |
| --- | --- |
| Phase (wake and join the workers) | 5 µs |
| Piece (claim and enter) | 200 ns |
| Rate | 2.5e8 scalar element operations per second per core |
| Workers | host parallelism |
| Bandwidth | 20 GB/s, shared |

Operations are counted over all pieces. Integer arithmetic over loop binders, constants and
selected geometry is address computation and is part of the rate. A packed element costs
eight operations at its consumption. External bytes are charged once per distinct view;
scratch traffic at its full multiplicity. Runtime extents are charged at their static upper
bound; branches at the maximum of their arms. Factors: one per root phase and one for the
remaining body of each candidate; per selected interval one phase overhead for a run of
root regions, and a write plus a read of every surviving elementwise output.

### Seed policy

- **Choices:** as on Metal, with `lower … for cpu` bodies first, requirement backtracking.
- **Sites:** binders of a root `parallel` phase take the smallest admissible widths whose
  piece count does not exceed `workers × 4`, raising the last binder first. Every other
  width takes its largest admissible value. Parts take one. A violated scratch limit lowers
  the highest site of the offending tiles.
- **Covers:** all singletons.

## CUDA mapping

Target `cuda`. **Coverage is scalar**: every candidate body is the shared scalar realization
printed as PTX and compiled by the driver (no nvcc, no NVRTC; the driver library is loaded at
run time). The intrinsic table is lane index, shuffle and sum; there are no matrix intrinsics
and no matrix coverage.

| Subject | Rule |
| --- | --- |
| Root `parallel` region | One launch over a one-dimensional grid; its pieces are the work items. |
| Participation | One thread per piece. One 32-lane warp per piece exactly when the instantiated body names a participant intrinsic. |
| Other root regions, serial root runs, stages | One single-thread launch each, in source order; values crossing a launch use invocation-owned buffers. |
| Nested regions, `pipeline`, `merge` | Serial loops, synchronous windows, recurrence in the owner thread. |
| Load | The shared borrow fixpoint (`seismic_compiler::scalar_load_rule`). |
| Tile storage | Every tile and materialized snapshot is an eight-byte-aligned bump allocation of the owner thread's scratch in device global memory; never reused. Tile accesses are memory traffic. |
| Reduction | Authored ascending order in the owner thread; a reassociation permission is not used. |
| Block size | Items per block = `min(work items, 256 / lanes per item)`, raised to `ceil(work items / grid blocks)` when one grid row cannot hold the launch. |
| Math | `exp`, `log`, `sin`, `cos` link the bundled PTX libm port (`exp` within 2 ULP of the interpreter). |
| Floating point | FMA only where authored. |

**Limits** come from the driver query (`Limits::from_device`; `Limits::gb10()` documents the
queried GB10 values for hosts without a device): pieces of a root `parallel` launch fit one
grid row of full blocks; scratch plus one status word per participating thread of a launch
fits one quarter of global memory (this mapping's rule, not a driver fact). A family in which
any candidate names a participant intrinsic is limited at the warp width throughout;
`realize` rechecks exactly. Participant intrinsics need a 32-lane warp.

**Estimate** `cuda-estimate-unqualified-v0`: every coefficient is unmeasured.
`launches × launch + (operations / rate + visits × visit) / min(pieces, concurrent threads) + memory bits / bandwidth`.

**Seed:** shared policy; root `parallel` pieces target the estimate's concurrent threads
within the grid limit. A violated scratch limit first raises the launch's widths, then lowers
the tiles' sites; a violated grid limit raises the launch's widths.

Site domains and intervals are the shared rules.

## Shared mapping code

The target-neutral half of every backend lives in `seismic-compiler`:
`selection::structure` (the candidate body walk, name and bound resolution including dynamic
parameters, `@dyn` range lengths and index ranges, multiplicity, distinct-view analysis, the
launch partition of root blocks), `selection::quantity` (derived quantities over site values;
a backend states a rule of its own as `Quantity::Rule`), `selection::mapping` (the divisor
domain rule, the two fused interval forms, limits as guarded constraints, the additive factor
skeleton, the constructive seed with requirement backtracking) and `load_rule` (the borrow
fixpoint). A backend supplies an `Accounting` (work classes, intra-piece share, tile ledger,
packed and reduction rules, intrinsic table), its limits with their seed repairs, its cost
formula (`Costs`), its seed piece target, and `realize`.

## Acceptance

- For every supported kernel, the selected Metal execution agrees with the
  reference interpreter, and the selected CPU execution under exact precision is
  bit-identical to it.
- Calling any hook twice with equal inputs gives equal results.
- The seed of every supported entry passes the joint audit.
- No emitted kernel exceeds a limit that a constraint was responsible for.
