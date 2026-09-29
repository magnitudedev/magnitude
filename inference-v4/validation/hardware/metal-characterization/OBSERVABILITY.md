# Metal performance observability closure audit

This audit follows the actual production path as of 2026-09-21:

- `ir/src/kernel/ops.rs::ClosedOpView`;
- `ir/src/metal.rs::{MetalIntrinsic, ScalarEmissionFamily}`;
- `backends/metal/src/render.rs` and `softfloat.metal`;
- `estimator/metal/src/lib.rs::execution_demand`;
- `backends/metal/src/{services,profile}.rs`;
- `ir/src/schedule.rs::ScheduleStep` and the workflow lifecycle.

“Closed” below means the mechanism is represented before selection with enough
structure to choose the applicable physical regime. “Blocked” means no valid
probe protocol can repair the missing representation.

| Mechanism | Current pre-tuning representation | Native correspondence | Required fact source | Current observation/accounting | Closure finding and required owner |
|---|---|---|---|---|---|
| Constants, geometry, arguments, extents | Dedicated `ClosedOpView` variants | literals, builtin geometry, parameter loads | queried topology + primitive integer/control facts | generic `CONTROL` latency | Partially closed. Cost must distinguish literal/no-instruction cases from parameter/global loads. Target-closed cost program owns expansion. |
| Native i32/u32 add/sub/mul/bit/shift | `Binary`, `Bit`, emission family `NativeIntegerBit` | native integer expressions | directly measured 32-bit dependency/capacity facts | one `INTEGER` service; probe is dependent multiply | Not closed: one multiply cannot represent add, shift, divide, select, or their pipelines. Physical primitive vocabulary owns distinct demanded mechanisms. |
| Native signed division/remainder | one `Binary` op | renderer adds remainder, branch, sign correction and division | primitive operations + explicit control paths | one `INTEGER` unit | Blocked by hidden expansion. Renderer and estimator must consume the same target-closed cost program. |
| 64-bit integer work | absent from scalar IR vocabulary | SoftFloat and matrix paths use `ulong`, 64-bit multiply/shift/jam | directly measured 64-bit primitive facts | folded into opaque F32/matrix services | Blocked. Add 64-bit cost primitives even if values remain internal to target lowering. |
| Boolean logic/select | `Cmp`, `Logic`, `Not`, `Select` | native Boolean/control expressions | directly measured select/control primitives | `CONTROL` or `INTEGER` unit | Partially closed. Branchless Boolean and divergent branch costs must remain distinct. |
| F32 sign/abs | `Unary(F32)` | bitcast + integer mask/xor + bitcast | derived primitive sequence + measured integer fact | `INTEGER` unit | Mechanism is statically derivable, but exact sequence is not a shared object. Make it a cost program used by rendering and estimation. |
| Strict F32 add/sub | `ScalarEmissionFamily::F32AddSub` | `f32_add/sub` SoftFloat state machine | primitive 32/64-bit facts + explicit paths | opaque F32 service; adjacent-count timing | Blocked: helper adds normalization, rounding, exceptional branches, loops. Expand before tuning. |
| Strict F32 mul/div/rem/min/max/FMA | separate semantic families | distinct SoftFloat helpers | primitive facts + explicit finite paths | opaque service per family; most use four-command matched subtraction | Blocked. Four-command residual is below noise for small effects and does not reveal path structure. Remove opaque services. |
| Strict F16 arithmetic | dtype on arithmetic op | F16 SoftFloat helpers with their own state machines | primitive facts + paths | one `F16_STRICT` service | Blocked for the same reason; grouping every F16 operation loses mechanism and path differences. |
| Strict BF16 arithmetic | BF16 op | widen each operand, F32 helper, narrow | exact composed cost program | one `BF16_STRICT` service | Blocked. Composition is visible in helper source but not authoritative pre-tuning structure. |
| BF16-to-F32 | cast emission family | `as_type<ushort>`, left shift by 16, `as_type<float>` | derived integer shift sequence | remapped to broad `INTEGER` service after failed standalone probe | Required resolution: never probe as conversion. Charge exact primitive sequence. |
| F32-to-BF16 | cast emission family | rounding/NaN-aware narrow helper | primitive facts + path structure | opaque conversion with matched baseline | Blocked by hidden control/rounding work. |
| F32/F16 conversions | cast emission families | SoftFloat conversion helpers | primitive facts + explicit paths | opaque directional services using forward/reverse scaffolding and subtraction | Required resolution: no directional coefficient from a dependency cycle. Expose and compose the forward primitive path. |
| F16/BF16 cross-conversion | `Cast` only; estimator manually composes classes | renderer composes widen/narrow helpers | exact composed cost program | two broad service units | Blocked until the composed program is shared. |
| F32-to-integer | cast family | clamp + truncation + native conversion | primitive math/control/conversion facts | matched-baseline opaque service | Confounded by feedback conversion. Expose clamp/trunc/convert sequence. |
| Integer-to-F32 | cast family | native conversion | direct primitive conversion fact, if independently observable; otherwise compiler instruction fact with bounded calibration | matched-baseline opaque service | Current directional residual is invalid. Production protocol must identify a direct single-command experiment or represent an inseparable native sequence. |
| F32 comparison | `ScalarEmissionFamily::F32Comparison` | SoftFloat equality/less-than, NaN checks, Boolean composition | primitive facts + comparison path program | direct opaque F32 comparison probe, no baseline | Required resolution: Boolean feedback scaffolding makes the semantic service unobservable. Expose helper structure; do not assign an opaque coefficient. |
| Approximate transcendentals | `ApproximateMath` with operation and dtype | `fast::{exp,log,sin,cos,sqrt,rsqrt}`, with half/BF16 widen/narrow | directly measured per operation/dtype plus composed conversion facts | one `APPROXIMATE_MATH` service | Not closed: one service erases substantial operation differences. Manifest must enumerate supported operation/dtype combinations constructionally. |
| Vector operations | explicit `ClosedOpView` variants | renderer panics because Metal advertises empty vector support | queried unsupported capability | estimator assigns hypothetical demand | Closed only because target capability excludes them. A future support change must add renderer and physical vocabulary in the same change. |
| Scalar global read/write | typed places, indices, representation geometry | address arithmetic + device memory access | queried limits; measured resident/spill/coalescing curves; derived address/working set | one `GLOBAL_MEMORY` capacity unit | Blocked: no cache or coalescing regimes, address arithmetic, reuse distance, or read/write distinction. Storage/access algebra owns derived regime inputs. |
| Workgroup read/write | typed local place | threadgroup access + address arithmetic | queried capacity; measured bank/conflict/concurrency curves | one `WORKGROUP_MEMORY` unit | Blocked: bank/access pattern and concurrency are absent from current service. |
| Packed/plane reads | representation geometry and recipe identity | plane loads + bit extraction + decode recipe | derived recipe cost program + primitive/memory facts | memory plus one `REPRESENTATION` unit | Blocked: recipe-specific work is hidden. Registry recipe lowering must produce the shared cost program. |
| Representation packet conversion | recipe pointer and packet/group geometry | looped byte loads, repack expression, strict publication, plane stores | exact recipe program + memory facts | group-count memory + representation units | Blocked: broad unit counts do not correspond to emitted mechanisms. |
| Atomics | `Atomic` op and dtype | native atomic or F32 CAS loop | measured operation/dtype/contention curves + explicit retry model | one `ATOMIC` capacity service | Blocked: contention and CAS retries are regimes, not a constant unit. No favorable contention distribution may be assumed. |
| Workgroup/subgroup barriers | scoped `Barrier` | Metal barrier with memory flags | directly measured by scope, participants, outstanding traffic | one `BARRIER` service | Partially represented; current service merges scopes and traffic state. |
| Lane index/shuffle/subgroup reduce | exact `MetalIntrinsic` variants | native simdgroup operations | queried support/width + measured op/dtype/participant curves | one `SUBGROUP` service | Not closed: lane index, shuffle, sum/min/max and dtypes are not interchangeable. |
| Matrix intrinsic | exact logical maps/dtypes/scratch in `MetalIntrinsic::Matrix` | renderer emits tile loops, staging, barriers, matrix ops, boundary predicates | queried support; measured native matrix facts; derived exact staging/control/memory costs | one matrix service plus hand-counted generic memory/barrier units | Partially closed. Structure exists, but emitted control/address/conversion work and occupancy are not shared as one cost program. |
| Kernel branch | condition and both blocks | native `if/else` | derived branch programs + divergence/path condition | one `CONTROL` unit plus both blocks risk depending traversal | Blocked until estimator semantics state which path executes and how SIMD divergence composes. No assumed branch probability. |
| Kernel repeat | symbolic bounds and body | native counted loop | derived trip count + exact body/control cost | one `CONTROL` unit plus body traversal | Mostly representable. Loop overhead and dependence must be explicit; symbolic bounds are evaluated from invocation facts. |
| Kernel launch | schedule launch + grid/workgroup expressions | command encoding and dispatch | queried limits, measured submission/dispatch, occupancy model | core submission + per-op services | Blocked by missing pre-native register pressure and inadequate occupancy/resource composition. |
| Occupancy | workgroup size and local allocation are present | pipeline has reflected thread/register constraints | queried limits + measured response surface + derived resource demand | no complete occupancy composition | Critical blocker: compiler register demand is unavailable before selection. Target-closed cost program must carry a conservative, mechanically derived live-register bound. |
| Schedule copy/fill | `ScheduleStep::Copy/Fill` with byte ranges | blit/compute transfer path | measured size/alignment/storage-state curves | broad core services | Needs exact endpoint/path ownership and resident/spill states; size is structurally available. |
| Scalar move/read/check | dedicated schedule variants | host/device scalar transfer and synchronization | measured transfer/sync + derived check work | core scalar/check services | Partially closed; scalar read may force completion and must not double-count synchronization. |
| Schedule if/repeat/choose | symbolic structured schedule | host control selects executable steps | derived symbolic control; child costs | generic control accounting | Structurally present. Choice must be resolved before the final evaluated domain; path predicates remain explicit. |
| Buffer allocation/materialization | storage/allocation plan and runtime workflow | Metal allocation, zero/fill/upload as applicable | measured size/storage-mode/lifecycle curves | incomplete lifecycle service set | Needs typed lifecycle facts and exact state transitions. Allocation reuse changes the applicable transition rather than applying a discount. |
| Command submission/completion | executable schedule/workflow boundary | queue submit, command buffer completion wait | directly measured batch lifecycle | submission exists; completion ownership unclear | Must have one owner each. Aggregate probe uses the same endpoint and queue lifecycle as production. |
| Cache state across launches | schedule/storage lifetimes expose order and allocations | unified-memory/cache hierarchy | queried topology where available + measured transition curves + derived reuse/working set | absent | Critical blocker. Every resident and spill branch must be represented; no “unassessed spill” leaf exists. |

## Why the existing acquisition is rejected

The current implementation does compile one library, which is useful, but then
measures services sequentially and returns on the first error. Most F32 and
conversion services use `(operation-large - operation-small) -
(baseline-large - baseline-small)`, combining four separately timed commands.
Remote runs took about 352–356 seconds and still failed at comparison after an
earlier BF16 failure. Only the strict-F32 dependent pair was used as held-out
composition evidence.

The replacement is one aggregate bundle with every endpoint represented, a
hard total deadline, and offline certification. The measurement vocabulary is
physical and reusable; semantic helpers are exact compositions in the shared
target-closed representation.

## Completeness ownership

| Change | Required same-change updates |
|---|---|
| New `ClosedOpView` or `MetalIntrinsic` mechanism | target-closed cost-program lowering, renderer consumption, physical fact demand, observability row |
| New renderer/helper primitive | closed physical primitive vocabulary and either queried/derived/measured fact source |
| New measured physical fact | fixed manifest field, raw bundle field, certified typed profile field, direct-observation argument |
| New schedule/lifecycle transition | schedule cost semantics, exact completion owner, lifecycle manifest/profile field |
| New capability | explicit supported/unavailable evidence; candidate domain excludes unavailable emissions |

This coupling must be implemented through shared exhaustive Rust variants and
records. Tests can validate examples, but they are not the completeness
mechanism.

## Required-fact disposition

This is the decision table used by the proof crate. “Currently not observable”
means the analytical profile is not constructible. It does not mean that a
partial profile or an unassessed evaluator leaf is allowed.

| Requirement | Allowed source in the replacement model | Current disposition |
|---|---|---|
| Strict F32/F16 helper cost | Derivable from named 32/64-bit primitive facts and an exact shared helper cost program with exhaustive data paths | **Currently not observable.** The program is hidden in `softfloat.metal`; opaque helper timing does not expose the path composition. |
| BF16-to-F32 widening | Derivable from exact representation casts plus one named 32-bit left-shift fact | **Currently not observable in the fact algebra.** Mapping it to broad `metal.integer` loses the exact primitive; a standalone BF16 conversion probe is invalid. |
| Other type-changing conversions | Derivable from a shared exact primitive sequence; a native conversion may be isolated-measurable only if one command can amplify it without a reverse conversion | **Currently not observable.** Existing dependency cycles contain a return transition and four-command subtraction. |
| F32 comparison | Derivable from exact bit/control helper paths and primitive facts | **Currently not observable.** The Boolean result requires feedback scaffolding and the helper expansion is hidden. |
| Occupancy response to threads/workgroup memory/registers | Isolated-measurable response surface with all three declared axes | The device response is measurable; **candidate applicability is currently not derivable** because pre-native register demand is absent. Therefore the combined requirement is currently not observable. |
| Global cache resident/spill service | Isolated-measurable working-set transition and bandwidth/latency regimes, keyed by explicit storage state | **Currently not observable** as `metal.global-memory`; the single class has no physical regimes. |
| Reuse distance and working set | Derivable from storage identity, schedule order, access maps, and symbolic invocation sizes | **Currently not derivable** by the estimator transfer. Required access/lifetime facts must become explicit. |
| Coalescing | Derivable from participant-to-address mapping, element width, and transaction geometry queried/measured for the device | **Currently not derivable.** Current demand records only a unit count. |
| Workgroup bank/access behavior | Isolated-measurable by access width, stride/bank pattern, and participant count; applicable regime derived from exact address map | **Currently not observable** in the broad workgroup-memory class. |
| Atomic service under contention | Isolated-measurable by op, dtype, participant count, alias group, and CAS retry regime | **Currently not observable** in the broad atomic class. Runtime alias/contention applicability is also not derived. |
| Submission | Isolated-measurable at one exact queue/command completion endpoint | Observable, provided completion is not charged elsewhere. |
| Copy/fill | Isolated-measurable by exact byte count, storage mode/state, and endpoint | Observable after the lifecycle state vocabulary is frozen. |
| Scalar read/completion | Submission/completion directly measurable; transfer derivable or separately measurable after operands are visible | **Currently not observable** as one scalar-read fact because visibility, transfer, and synchronization ownership are merged. |
| Allocation/materialization/reuse | Each exact state transition isolated-measurable by bytes/alignment/storage mode; applicable transition derived from the prepared allocation plan | **Currently incomplete.** Allocation/materialization are absent from the current closed service set and reuse is not an additive discount. |
| Subgroup operations | Support/width directly queryable; each op/dtype/participant response isolated-measurable | **Currently not observable** in one subgroup class that conflates lane index, shuffle, and reductions. |
| Matrix operation | Supported dtype combination directly queryable; exact native tile operation isolated-measurable; staging/control derived from the shared cost program | Native tile is measurable, but total intrinsic composition is incomplete until renderer and estimator share the staging/control program. |
| Concurrent service composition | Capacity response isolated-measurable for declared concurrent mechanisms; simultaneous demand and ordering derived from schedule/resource topology | **Currently not observable.** The additive service algebra does not state overlap, shared bottlenecks, or exhaustive concurrency regimes. |

No hardware campaign should begin while any row required by an admitted Metal
candidate remains “currently not observable.”
