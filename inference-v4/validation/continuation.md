# V4 continuation

Goal created 2026-09-17 09:57:51 UTC. Requested minimum work interval ends
2026-09-17 19:57:51 UTC; that time is not a completion criterion. Full completion
requires the master specification's gates. Work autonomously across sessions.
Local commits are authorized infrequently at verified milestones. Never push.

## Current governing direction — compiler closure

The active replacement goal is Appendix A of the master specification: qualify the
complete program-to-tuned-executable compiler boundary before resuming model-kernel
performance tuning. Historical engine work below is evidence, not the current work
order. Manual kernel/candidate performance experiments are frozen.

The user prohibits further commits. Keep all further work local and uncommitted;
no staging, commits or pushes. Earlier permissions below are historical and revoked.
The 24 implementation commits after `wip kernel tooling` were squashed at the user's
request into a single commit named `wip compiler engine`; no content was discarded.

Initial boundary audit confirms that the shared realization currently describes
Cranelift scalar programs, while Metal emits from a separate representation. Intrinsic
contracts contain signatures and limited write effects but lack execution-resource,
participation and synchronization contracts; CPU/CUDA intrinsic tables are empty.
The prediction API accepts separately supplied service mappings and free-text evidence,
not an emission-derived execution model or independently verified proof. These are
foundational integration gaps, not requests to tune model kernels.

## Starting point

- Repository reference: `d3c11098` (`wip kernel tooling`); initial working tree clean.
- Foundation tests before migration: 26 passed, no failures. No accounting or Metal
  unit tests existed. This is not numerical device or engine qualification.
- Compiler has parsing, symbolic checks, interpretation, Metal emission, static
  composition plans and manually configured split/tiling options.
- The old accounting walker caps accumulated touches at parameter allocation size,
  prices special functions as eight FLOPs, and converts some unknown extents to zero.
  It cannot serve the required roofline or selection contract without replacement.
- Supply is only a device-name-keyed streaming bandwidth cache. No resource topology,
  provenance-qualified calibration or complete realization predictor exists.
- Lowering selection uses first applicable/type-specialized preference, not modeled
  costs. CLI contains synthetic hardware values. These are unqualified prototype paths.
- Only Metal has native execution. CPU lowering currently supplies portable semantics;
  no production CPU compiler/runtime exists. CUDA and Vulkan are unimplemented.
- Qwen harness has manual bindings/tuning and model-specific assumptions. Preserve it
  as validation material; it does not satisfy the production engine requirements.

## Completed initial changes

- Migrated into one V4 workspace and renamed accounting; preserved dependency versions.
- Kept Qwen POC under validation, with no claim of engine parity.
- Replaced incorrect prototype demand/supply modules with exact byte-region unions,
  checked counts/uncertainty, scoped service units and evidence-aware rate constraints.
- Added checked-IR portable work derivation through calls/loops/whole-axis streams;
  runtime conditions and unsupported analysis remain explicit. Added `seismic account`.
- Removed misleading prototype roofline percentages from device-run reports.
- Added durable architecture contract and local workflow instructions.

Verification: 41 workspace tests pass (26 inherited, 15 new accounting tests).
Accounting passes targeted Clippy with warnings denied. Local M4 Max packed projection
smoke has zero observed reference error. Library/Qwen check: 21 files, 19 functions,
5 lowerings. Source hashes and test log are under ignored validation/results.

Remote discovery: all three hosts reachable; both M4 machines expose 48 GiB RAM;
Sparky reports GB10 driver 580.159.03. M4 Pro 01 has the V3 checkout and both MLX model
caches plus GGUF caches; Rust was not found in standard locations. Model artifact
identity and numerical/throughput qualification remain pending.

## Native CPU and access accounting checkpoint

- Added checked-IR memory derivation with backing identity/offset bindings, exact
  touched-region unions, packed word/scale/bias planes and explicit analysis budgets.
  Incomplete/runtime-dependent traversal reports unavailable coverage, not zero.
- Added implicit load/promotion/publication conversions to the portable work account.
- Added native scalar CPU compilation through pinned Cranelift 0.125.4, with executable
  and scratch ownership, validated bindings, snapshot loads and native view guards.
  Unsupported constructs are explicit errors. This is not yet the optimized CPU backend.
- CPU run/lower are available in the CLI; hardware-independent workspace builds on
  Linux. Metal runtime is macOS-gated, and the Qwen POC requires `metal-poc` explicitly.
- Installed an isolated official Rust 1.91.1 toolchain under Sparky's owned validation
  root. Its archive checksum was verified. Shared V3 checkouts remain untouched.
- Local and Sparky portable workspace passed 56 tests before the final native bounds
  regression; CPU now passes seven tests on both hosts (57 total workspace tests).
  Targeted CPU/accounting strict Clippy passed after the final guard change.
  Local explicit Qwen POC feature check passed with inherited warnings.
- Both M4 Pro hosts passed initial packed projection and RMS norm Metal smoke checks.
  Sparky passed native packed projection, RMS norm and gated projection checks, plus
  Linux CLI gated projection. These are numerical smoke checks, not performance parity.
- Remote artifacts live under `~/seismic-v4-validation-01a0ae7f/` on each host.
  Logs under ignored validation/results include commands and errors; CPU/Metal complete
  numerical, performance, distribution and generalization matrices remain incomplete.

## Shared compiler and CUDA checkpoint

- CPU and CUDA now consume the same typed scalar instruction realization. CUDA emits
  PTX directly and JITs with the installed driver, with owned context/code/buffers,
  checked capacities, native view guards and explicit launch geometry. Driver queries
  report actual registers, local bytes, capability and device limits. This is a scalar
  correctness baseline, not an optimized or automatically tuned CUDA implementation.
- Portable workspace passes 62 tests on both local macOS ARM64 and Sparky Linux ARM64.
  Eight explicitly executed CUDA device tests pass on Sparky, covering projection,
  gated projection, norm, runtime guards, zero domains, snapshots and scalar semantics.
  CUDA exp tests 300,117 finite/special/threshold/raw-bit inputs against f64 reference
  rounded to f32; maximum observed difference is one ULP (declared tolerance two).
  The implementation preserves its pinned libm/SunPro provenance and license.
- Shared semantic cases cover tile copies, loop-carried tiles, branch merges, BF16
  publication, mutation across producer substitution, mixed-width scalar ABI, and
  65-element direct/streamed snapshots across backing writes. Local Metal and both
  M4 Pro hosts passed these, plus renewed packed projection and RMS norm smokes.
  The explicit Metal Qwen POC feature still compiles.
- Corrected value-aliasing and snapshot bugs in Metal, mutation/publication bugs in
  producer substitution, BF16 NaN encoding and F16 subnormal reference decoding.
  All finite half encodings now round-trip in the reference test. ABI layouts derive
  from parameter types. Host buffer reads and invocation capacities are checked.
- Removed the fabricated Metal core count; unavailable is represented as None.
  Initial Metal placement/split policies remain explicitly unqualified prototype work.
- Shared compiler, CPU, CUDA and accounting pass targeted strict Clippy. Remote logs
  and emitted artifacts remain under ignored validation/results and owned host roots.

## Realization accounting and candidate work (in progress)

- Added independent seismic-realization contracts so accounting and compiler can
  consume the same artifact without a dependency cycle. Compiler emission attaches
  block execution domains, validity conditions, and memory-root identities.
- Accounting composes constant-loop counts without traversing their iteration space;
  branches retain intervals, dependent ranges remain unavailable. Reports separate
  concrete SSA requests from native instructions/cache/DRAM traffic. Five new tests
  include a million-item domain and exact candidate storage/access differences.
- Added explicit materialize versus borrow-proven-read-only load candidates, with
  tensor and tile mutation effects including declared intrinsic writes. Both CPU
  and CUDA pass the common semantic cases under both candidates. No automatic
  performance preference is implemented or implied.
- CUDA now reports actual occupancy capacity, L2/global/storage limits and native
  register/local/shared storage. Event timing has a separate boundary from host
  submission, synchronization and status reads. Nine CUDA runtime/device tests passed
  before the FP16 extension, including timing after the Device owner is dropped.
- Added shared SSA FP16 conversion/publication; CPU, local Metal and CUDA pass all
  65,536 encodings and 253,958 finite rounding-boundary cases. M4 Pro 02 also passed
  the complete half/semantic suite, including NaN and scalar publication cases.
  Local portable workspace has 68 tests; targeted strict Clippy passes.
- Pinned V3 reference sources are being built under m4-pro-01's owned validation root.
  Its TileLang and TVM revisions match the local V3 reference, exported from source
  archives without modifying any shared checkout/Git state. Seven native template
  tests pass. First frozen session-bench 4B run has started; no baseline result yet.
- m4-pro-02 exhausted disk during isolated dependency setup. Removed only this task's
  temporary v3-reference/.venv and uv-cache. Existing source/models/jobs were untouched;
  compiler device tests continue using small deployed artifacts. Further large builds
  use m4-pro-01 or Sparky, with disk capacity checked first.

## Artifact and model interpretation checkpoint

- Added Rust seismic-engine library with checked immutable open-file owners, bounded
  GGUF and Safetensors directory readers, MLX affine planes, V3 content identities,
  and Qwen dense/routed weight roles. No numerical model execution is implied.
- Qwen bindings preserve container-specific recurrent mapping and A_log import,
  tied embeddings, main layer ordering, speculative trailing blocks, experts and
  rejection of unbound GGUF roles. Source ownership survives artifact release.
- 85 portable workspace tests pass locally; 17 engine tests pass locally and on
  Sparky. Engine strict Clippy passes with --no-deps. Inherited language warnings
  remain. SHA-256 uses pinned RustCrypto 0.11 with runtime-detected native support;
  independent known digest and V3 composition tests pass.
- Real header and Qwen role comparisons match V3 for 4B GGUF (441 tensors, 32 layers)
  and 35B-A3B GGUF (733 tensors, 40 layers). 4B MLX matches all 1221 tensor directory
  entries, roles and full content identity on m4-pro-02, including a release build.
  Logs are artifact-directories-m4-pro-01, qwen-gguf-roles-m4-pro-01 and
  qwen-mlx-roles-m4-pro-02-release under results.
- V3 4B MLX single-session baseline completed on m4-pro-01: two balanced measured
  passes, both exact retrieval, 119 prompt/37 completion tokens, mean native-service
  decode 81.350 tokens/s and TTFT 200.274 ms. This is one reference workload cell,
  not broad performance qualification or V4 parity. Compact source/artifact/evidence
  manifest is v3-baseline-4b-m4-pro-01.json; full run is retained under results.
- Initial V3 900-second startup timed out during compilation. Retry explicitly
  recorded 3600-second allowance, with workload/timing policy unchanged. 35B-A3B
  GGUF baseline now running at m4-pro-01 owned root, log v3-sessionbench-35b.log,
  run 20260917T122815.582786Z-d81013ca. Avoid concurrent GPU work on that host.
- m4-pro-02 real artifact comparison uses current isolated V3 Python source with
  the existing Python environment for device-free interpretation only. It is not
  a claim of current TileLang execution qualification there.

## Runtime scalar contracts and reductions checkpoint

- Declared index bounds now persist through HIR and specialization into a shared
  ScalarParameter contract. CPU/CUDA encoding, Metal typed/raw ABI and interpreter
  enforce domains; native scalar SSA also retains guards. Runtime index atoms read
  current scalar bindings. Packed embedding is qualified on CPU and CUDA.
- Scalar CPU/CUDA now support floating max/min/argmax, ordered narrow sums and
  integer extrema. Native PTX supports the signed index widening this requires.
  The standard two-phase argmax matches the interpreter on CPU and CUDA's explicit
  sequential candidate; parallel multi-phase CUDA remains unimplemented.
- Added the spec's `ordered=true|false` reduction permission. Accounting retains it.
  Metal implements ordered cross-lane reduction with narrow publication. Default
  narrow reduction uses this legal realization until narrow collectives are admitted.
- Fixed Metal argmax returning INT_MAX for all-negative-infinity/all-NaN rows and
  typed extrema initialization. New tests exposed default Metal fast-math erasing
  half publication and implicit multiply/add contraction. Compilation now disables
  fast math via the macOS-13-compatible API and emits clang contract(off); explicit
  fma stays fused. This is a correctness requirement, not a performance claim.
- 92 portable workspace tests pass locally and targeted strict Clippy passes. The
  preceding 90-test portable checkpoint passed on Sparky; two subsequent domain/
  two-phase tests pass locally and the new CUDA case passed on Sparky. Eleven CUDA
  native tests plus exhaustive half passed across the two device runs. Native Metal
  semantic/raw-ABI plus exhaustive half passed locally and on m4-pro-02. Qwen POC
  feature check passed after scalar-contract changes.
- Evidence: reduction-index-workspace/clippy/sparky/cuda/m4-pro-02, argmax-standard-
  cpu/cuda and reduction-index-metal-and-half logs under results. Initial failing
  Metal regression evidence is retained. V3 35B baseline still compiling on m4-pro-01;
  do not compete for that GPU while it runs.

## Next work

Extend exact IR-derived access analysis with symbolic regions for large domains, then
physical realization/profile constraints and model-driven candidate selection. The
current work report explicitly declares memory paths, predictions and roofline unavailable.
Audit dependent loops, implicit conversions and branch conditions before extending exact
coverage. Keep precise provenance for conditions and costs.

Extend CUDA and CPU realization coverage and replace prototype Metal selection policies. Deploy owned test
artifacts to named hosts without modifying their shared checkouts or other jobs.
Continue V3 preservation inventory and baseline measurements alongside compiler work.
Full engine, tuning, backend and release gates remain incomplete. Never stop at this
foundation milestone or describe it as full completion.

## Runtime streams and attention checkpoint

- Shared CPU/CUDA scalar realization separates bounded allocation capacities from
  runtime logical extents. Runtime slices and points validate bounds; chunk loops
  preserve physical strides, actual tail extents, snapshots and loop-carried values.
- Removed the hidden 64-element streaming default. Lowering honors explicit positive
  capacities or derives a whole-axis bound from the view structure. This baseline is
  not a performance selector. Unsupported capacity proofs remain explicit errors.
- Empty streams now execute no pieces consistently in reference/native execution and
  portable accounting. Runtime-dependent realization traffic remains unknown; portable
  whole-piece work retains its nonempty-domain condition.
- Standard attention passes an independent f64 softmax reference for multiple visible
  ranges and piece choices on CPU, Sparky CUDA, local Metal and M4 Pro 02. CPU/CUDA
  test both materialized and effect-proven borrowed loads, tail and empty ranges, and
  rejection of negative/reversed/oversized ranges. This is correctness, not throughput.
- Local and Sparky portable workspaces pass 98 tests. Sparky passes 13 native CUDA
  tests plus exhaustive half; local/M4 Pro 02 pass three native Metal tests. Strict
  compiler/accounting/CPU/CUDA Clippy passes. Logs use dynamic-stream/attention prefixes.
- Metal's pre-existing runtime dynamic-slice path clamps invalid ranges and does not
  yet provide the scalar backend's bounds-error contract. Valid-input attention tests
  do not qualify that safety gap. Full backend safety/tooling remains required.
- V3 35B-A3B baseline remains compiling in the owned m4-pro-01 reference, run
  20260917T122815.582786Z-d81013ca. The 4B reference result remains one workload cell.

## Metal runtime bounds checkpoint

- Repaired the dynamic-slice gap identified above: invalid ranges record device
  failure and become safe empty domains. Logical point/packet indices and physical
  device reads/writes have checked guards. Store dimensions must match their views.
  Lanes do not exit through error paths inside collective work.
- Both standalone dispatch and composition plans bind compiler-owned status through
  completion, reject bounds failures, and reset status for each new invocation.
  Plan ranges and repeat counts are checked before submission. Failed invocations may
  have partial physical writes and must not be logically accepted.
- Fixed scalar reductions nested in expressions and materialization of borrowed views
  for reduction realizations that require owned lane storage. Natural stream test
  kernels remain unchanged across backends.
- Six native Metal tests and exhaustive half pass locally and on m4-pro-02, including
  invalid stream ranges, altered domains with backing canaries, plan failure followed
  by a valid invocation, attention and prior precision/value regressions. Renewed packed
  projection and norm smoke checks have zero observed reference error. Qwen POC feature
  check and 98 portable workspace tests pass. Logs use metal-bounds prefixes.
- Removed a remaining CLI 'achieved GB/s' calculation based on allocated buffer sizes;
  allocation size is now labeled as storage, not measured traffic.
- V3 35B baseline first pass completed with valid exact retrieval; second pass running.

## Resident runtime checkpoint

- Added owned CPU/CUDA/Metal buffers and checked byte subviews. Native code retains
  execution resources through synchronous completion; code caches do not retain model
  tensors between runtime invocations. CUDA compilation no longer allocates every
  tensor parameter. CPU aliases borrow each backing owner once and use raw native
  pointers without constructing overlapping Rust mutable slices.
- Shared realization buffer specifications carry natural alignment. Dense/packed plane
  offset geometry has one owner, reused by native Metal and backend-neutral plans.
  Metal validates device ownership and composes base view offsets with plan offsets.
- New seismic-runtime facade compiles explicit candidates and executes resident inputs.
  Compiled compositions deduplicate equal kernel/shape instances, preflight named
  inputs/ranges/scalar ABI, and execute synchronous steps. This is not a claim of
  optimized batched submission, complete invocation alias legality, or auto-tuning.
- Composed resident execution passes CPU/local Metal/M4 Pro 02/Sparky CUDA, including
  intermediate reuse, nested offsets, canaries, retained owners, repeated bindings and
  rejection of missing later-step scalars before earlier writes. Workspace and targeted
  strict Clippy pass. Evidence logs use resident prefixes.
- V3 35B Metal reference completed two exact-retrieval passes; summary stored alongside
  the 4B baseline. The pinned isolated V3 CUDA reference now builds on Sparky and its
  first 4B session benchmark is running. These remain single workload reference cells,
  not V4 parity evidence.

## Generic specialization and resident weight import

- Entry lowering now binds concrete floating/packed element parameters, propagates
  them through nested calls and expressions, and rejects missing/unsupported bindings.
  Composition plans preserve inferred element arguments; code reuse keys distinguish
  equal shapes with different element types. CPU/Metal/CUDA tests cover F32/BF16/F16
  entry and nested calls and a mixed-dtype composition requiring two code instances.
- Rust weight import owns retained native planes and compiles dtype conversions and
  NegativeExp through standard Seismic kernels. Canonical Q4 group-64 planes transfer
  unchanged. Dense reshape follows V3's equal-element-count rule. Malformed stored
  byte counts and unsupported affine geometry reject before numerical execution.
- CPU/local Metal/Sparky CUDA import tests pass. Real 4B MLX norm, recurrent decay and
  packed projection imports pass on M4 Pro 02: exact norm conversion, maximum decay
  absolute error 2.98e-8, exact packed planes. This is not complete model execution.
- Workspace and targeted strict Clippy pass; evidence uses generic/weight-import
  prefixes. No resident policy/cache, GGUF numerical codec import, full model runtime,
  physical predictor or automatic selection is claimed by this checkpoint.
- Sparky V3 baseline initial launch failed because uv was absent from SSH PATH; retained
  failure evidence and relaunched with the owned launcher plus explicit tool PATH.
  Run 20260917T134945.553681Z-18b670bb remains compiling its first CUDA pass.

## Service prediction and observation groundwork

- Added explicit calibrated-service prediction with latency/concurrency coverage,
  shared-pool serialization, independent-pool overlap, dependency limits, sequential
  stages and one submission boundary. Provenance and workload/profile/boundary
  identities survive into the report. Ranges represent input variation, not statistical
  confidence. Missing material demand, calibration or concurrency prevents ranking.
- Finite-set comparison chooses only a non-overlapping model winner (stable order for
  exact ties). This arithmetic layer is NOT yet the IR-to-physical mapper, calibrated
  production predictor, candidate generator, compiler selector or qualification gate.
- Runtime observations separate host binding/submission/completion/status time from
  GPU event/command time. CPU has no fabricated device timer. Local CPU/Metal and
  Sparky CUDA resident observation checks pass.
- Added an explicit copy calibration experiment recording source/executable identity,
  device facts, all samples, verified output and IR-derived logical access regions.
  It does not label allocation bytes as traffic or claim observed copy rates are DRAM
  bandwidth/capacity. Initial local CPU/Metal records use copy-probe prefixes.
- Predictor analytic tests, portable workspace and targeted strict Clippy pass.
  A generic-test Clippy style issue caught after the prior checkpoint was corrected.

## Symbolic access-union analysis

- Copy-probe evidence exposed iteration-sized access analysis (about 0.329 seconds
  for 65,536 elements in the debug tool). Added a symbolic affine-loop union path:
  prove affine indices, invariant contiguous view geometry, endpoint bounds and no
  inter-iteration holes before forming the union. This is a proof, not extrapolation.
- Nonlinear/conditional accesses, changing view geometry, tensor alias assignments
  and unsupported scopes retain the bounded evaluator. Attempted symbolic analysis
  consumes the same explicit analysis budget. No allocation-size substitute is used.
- A billion-element copy now takes the same analysis-step count as a 17-element copy.
  Independent bitmap tests cover 60 overlapping/reversed/strided combinations, and
  nonlinear/modulo/conditional regressions preserve exact coverage. Workspace and
  strict accounting/runtime Clippy pass. Renewed 65K probe analysis was ~0.000462 sec;
  this is an analysis-cost observation, not a kernel-throughput improvement claim.

## CUDA scalar math and integer division

- Added log/sin/cos backend primitives from immutable musl 1.2.5 sources, compiled
  offline to embedded PTX 7.0/SM80. Maintainer generator records source/adapter/output
  hashes and Clang 18.1.3 flags; rejects external calls, FTZ and contraction. Customer
  execution still needs only the driver, not Clang/nvcc/NVRTC. Full notices retained.
- Each primitive passes 564,093 independent-reference samples on Sparky (specials,
  random bits, normal exponent transitions, reduction thresholds and pi/2 neighbors),
  maximum observed error one binary32 ULP. This is empirical, not exhaustive proof.
- Natural rotary/recurrent preparation exposed missing scalar integer division.
  Shared scalar compilation now implements checked signed Euclidean / and %, unsigned
  / and %, rejecting zero divisors and i32 MIN/-1. Metal/interpreter follow the same
  error/value contract. No kernel reshaping was used to bypass this compiler gap.
- Native integer tests pass CPU/local Metal/M4 Pro 02/Sparky CUDA; interpreter tests
  confirm errors rather than panics. Standard rotary and recurrent preparation pass
  both load strategies on CUDA with sequential dispatch. Parallel multi-phase scalar
  dispatch is still to be implemented; this is not optimized model execution.
- 17 native CUDA regressions plus exhaustive half and portable PTX tests pass. Local
  workspace and strict compiler/CUDA/runtime Clippy pass. Math/preparation/integer
  logs retain evidence, including native register and local-stack feedback. Mapping
  primitive internals into physical accounting remains incomplete.

## Ordered scalar parallel phases

- Shared realization/compiler now represent source-ordered parallel phases with
  source-statement identity. CUDA compiles separate domains and validates every
  binding before executing the first phase, retaining owners until completion and
  stopping later launches after failure. Phase-local values cannot silently escape;
  unsupported cross-phase local storage/control remains a compilation error.
- Runtime facade now accepts natural multiple-parallel-loop CUDA kernels. Sequential
  candidate semantics remain available. GPU observation scope explicitly distinguishes
  CUDA kernel-event sums (host gaps excluded) from Metal command-buffer intervals.
- Accounting retains each phase's domain, completion edges and logical retained
  scratch. It does not independently sum shared interface obligations or claim the
  logical scratch count is physical traffic/register demand.
- CPU/local Metal/Sparky CUDA cross-phase reverse-read test passes. Real rotary and
  recurrent preparation now pass both sequential and parallel CUDA candidates with
  both load strategies. Portable workspace and strict targeted Clippy pass.
- V3 semantic audit found a major POC limitation: POC uses BF16 residuals and algebraic
  norm/projection folding; current V3 uses F32 residuals and explicit compact norm
  publication. Do NOT promote the POC composition to the engine. Preserve V3's exact
  publication boundaries in production model programs and qualify any later fusion.

## Generic publication and V3 dense suffix

- Generic floating output types specialize at stores; concrete packed output stores
  are rejected. Nested generic publication is tested at exact BF16/F16 rounding ties.
  The importer now uses one authored conversion kernel for all floating targets.
- Composition planning and portable work/access accounting accept the same checked
  element bindings as compilation; no unbound storage type is guessed. Updated
  normalization tests explicitly bind their former BF16 fixture types.
- `seismic-std` is a Rust package embedding the actual standard sources. Added generic
  batched linear, SiLU, multiply/add and generalized RMS normalization. Model dense
  suffix composition lives in the engine and preserves seven V3 stages: norm, two
  projections, SiLU, product, down projection, F32 residual add.
- DenseSuffix owns imported weights, intermediate publications and reusable compiled
  code. Shape/type/epsilon admission precedes allocation; tests cover repeated runs,
  in-place residual publication, all dense intermediate bytes, and q4g64 weight planes.
- This exposed a Metal generic-read bug: narrow native storage reads were emitted
  without their checked F32 conversion, causing ambiguous mixed-precision fma and
  potentially premature narrow arithmetic. Backend reads now preserve checked scalar
  type before native overload resolution. CPU/local Metal/M4 Pro 02 tests pass.
- Access/work accounts derive from this same composition and are checked for compact
  intermediate versus F32 residual bytes. These are logical accounts, not physical
  calibrated predictions. No tuning or full-engine parity claim is made.
- CUDA native qualification is pending the isolated V3 baseline's second compilation
  pass; the first CUDA warmup and measured retrieval completed valid/exact. Avoid
  introducing a simultaneous GPU probe during the reference's measured requests.

## Recurrent composition, reshape and dynamic point access

- Added dense contiguous `reshape(view, (dims, ...))` to the checked language,
  interpreter, composition planner, native CPU/CUDA/Metal and logical accounting.
  Shared checked geometry preserves size/order/offset, ignores irrelevant singleton
  strides, rejects holes/transposes, negative dimensions and overflow. Current native
  support requires static extents; packed reshape remains explicitly unsupported.
- Runtime I32 point indices now retain executable bounds obligations rather than
  requiring source workarounds. Parallel stores still need independent-region proof.
  Interpreter point reads/writes fail safely rather than panic on invalid indices.
  CPU/local Metal tests cover point/row reads, invalid indices, recovery and unsafe
  parallel-write rejection. Reshape tests cover subview canaries and snapshot survival.
- Recurrent preparation accepts activation/parameter type bindings. Delta step now
  exposes grouped/tiled mapping explicitly. Updated the development POC caller; POC
  remains unqualified and has its previously recorded publication/topology limitations.
- Engine authors a twelve-stage recurrent decode mixer + F32 residual composition.
  Dense reshape connects per-head norm to flattened gating without extra copy kernels.
- Independent fixture is generated with the pinned V3 primitive reference functions,
  explicit BF16 publication, source/generator hashes and NumPy version. Three rows
  preserve evolving convolution/delta state for each of two head mappings. CPU/local
  Metal/M4 Pro 02 pass, with compact stages/residual exact in this fixture and FP32
  state differences on the order of 1e-8. This is a small BF16 numerical cell, not full
  model, F16, batching, transactional-state or performance qualification.
- Prior dense/generic/import tests now pass Sparky CUDA as well. New reshape/recurrent
  CUDA tests are compiled but remain pending the V3 35B baseline measurement window.
- V3 4B CUDA baseline completed two valid/exact retrievals. Native-service mean TTFT
  140.968537 ms, decode 81.161410 tok/s, prefill 988.942523 tok/s. Source/artifact/result
  identities are retained in v3-baseline-4b-sparky.json; single-workload scope only.
- Active V3 35B CUDA run: 20260917T145632.550154Z-7b0c440b, SSH session 35575,
  remote v3-cuda-35b-baseline.log. Startup timeout3600; first pass still compiling.

## Attention composition and tile output checkpoint

- Added reusable V3 attention preparation with head-interleaved query/gate projections,
  RMS normalization, four-coordinate rotary selection and a nonrotary tail. Shared
  tile equations use ordinary functions; the checker now proves conservative tile
  output effects instead of requiring fake initialization at call sites.
- Attention decode streams visible old history then consumes the fresh dense row
  separately, before dense-history persistence. The eleven-stage engine composition
  retains compact publications, sigmoid gating, output projection and F32 residual.
  Dense history is a correctness baseline, not quantized KV/TurboQuant qualification.
- Fixed branch definite-assignment joins and symbolic proof search that could exhaust
  its budget following cyclic path bounds before a direct loop bound. Tests reject
  partial output helpers, input/output alias reads and one-branch initialization.
- CPU/CUDA boolean lowering and Metal emission now preserve the interpreter's eager
  boolean operands. CPU/Metal truth-table and invalid-access/recovery tests pass.
- Metal transposed element reads now retain materialized tile ownership rather than
  requiring every transposed operand to be a borrowed tensor view.
- Independent NumPy/V3 generators preserve source hashes and explicit BF16 publications.
  Three attention cases include empty history, nonzero visibility start, a destination
  independent of rotary position and long coordinates (131071). CPU outputs match the
  fixture exactly. Local Metal and M4 Pro 02 pass; the long case differs by up to
  0.00390625 in compact query, 0.0009765625 in attended output, and 0.00048828125 in
  final residual. Tests bound compact-stage deviations and separately require exact
  F32 addition of the actually published projection. This is not bitwise cross-backend
  equivalence or long-context model qualification. All untouched history is checked.
- Sparky CPU attention/rotary/boolean tests pass. Native CUDA runs are queued behind
  the 35B V3 session baseline (20260917T145632.550154Z-7b0c440b), still compiling.
- Workspace tests pass. Full-model engine, transaction policy, optimized realization,
  physical accounting, calibration and automatic model-driven selection remain open.

## Recurrent successors and V3 state ownership

- Recurrent preparation and delta update now take separate accepted and successor
  state buffers. The engine composition publishes successors without changing its
  accepted inputs; the validation-only POC explicitly aliases them for its old working
  state behavior. No tensor copy is introduced by this change.
- Added the Rust shared-history state store with V3 reservation, checkpoint/fork,
  trimming, exclusive-adjacent-range merging and capacity semantics. Mutable advances
  exclude overlapping logical operations, own fresh successor components, and become
  committable only after one successful synchronous execution. Abort/drop preserves
  accepted state even after partial physical writes. Async submission is not claimed.
- Runtime buffers retain allocation identity across clones/views. Reclamation counts
  distinct allocation bytes only when all handles are selected. State reclamation
  respects checkpoint/descendant/external-view pins; idle history ownership can release
  independently of external physical pins.
- Nine state tests adapt V3 shared-state cases, including fragmented reservations,
  failed work, parent-first destruction, checkpoint-local trimming, component version
  isolation and reclamation. Recurrent fixture execution now uses this transaction
  API for three consecutive grouped/tiled steps and checks accepted inputs unchanged.
- CPU/local Metal recurrent checks, local resident allocation checks, workspace tests,
  targeted strict Clippy and Sparky CPU state/recurrent/resident checks pass. Native
  CUDA remains queued while the 35B V3 reference compiles. Full engine scheduling,
  completion batching, memory admission/eviction and model parity remain unfinished.

## Full dense decoder integration and measured performance gap

- Added a checked resident-composition owner, scoped native compilation reuse and
  fixed per-kernel candidate choices. Every tensor has explicit ownership; runtime
  bindings cannot override fixed model parameters. Shared compiled kernels remain
  serial while they own invocation scratch. Optional observations retain native
  host/device timing boundaries and kernel identities.
- Added compact embedding -> F32 residual and final normalized readout compositions.
  The single-row dense decoder executes every described recurrent/attention/FFN
  block, returning pending state plus logits for caller acceptance. Dense KV and a
  single visible history range are explicit limitations; routed models are rejected.
- A four-block NumPy/V3-reference fixture passes CPU, local Metal and Sparky CUDA,
  with maximum observed logit deviation 2.9802322e-8 across three tokens. Its checks
  exercise abort/retry, acceptance and checkpoint/fork continuation. Composition
  ownership/reuse tests, workspace tests and targeted strict Clippy pass.
- The real 4B MLX artifact executes all 32 layers on local Metal. It compiles 23 unique
  kernels, preserving tied weight sharing in the qualification importer. Three token
  IDs produce finite logits and the same top-eight ordering as pinned V3 on M4 Pro 01
  with explicitly dense BF16 KV. These are different devices and only three synthetic
  tokens/top-eight logits: no performance/quality/engine parity claim. Evidence and
  limits are recorded in validation/decoder-4b-integration.json.
- Actual width exposed missing transpose-coordinate handling in Metal tile ownership
  analysis. Fixed that compiler analysis; a cross-lane transposed snapshot regression
  overwrites its source and verifies exact old values on CPU, Metal and CUDA.
- The explicit piece-64 realization took ~1.43 seconds/token. Device observations
  showed projections dominated; their packed dot had only eight words per 32-lane
  subgroup and repeated reductions across pieces. Whole-axis projection pieces,
  keeping attention at 64 to fit threadgroup memory, reduced the same local baseline
  to ~0.155 seconds/token. Device time is now ~0.04 seconds/token, with substantial
  per-kernel submission overhead. This is a diagnosed explicit candidate experiment,
  not an automatically selected policy or qualified tuner. All-whole-axis failed the
  real attention threadgroup-memory limit (41152 > 32768 bytes), as it should.
- Metal no longer silently halves widening/splitting after emission failures or
  ignores nondivisible/unsupported choices. Invalid/unsupported candidates reject.
- Queued Sparky native CUDA attention/rotary/recurrent, reshape, dynamic points,
  boolean, resident, plan and transpose checks all pass. The V3 35B CUDA reference
  timed out after 3600s startup (run 20260917T145632.550154Z-7b0c440b); its failed
  evidence is retained. Retry 20260917T160207.864424Z-d486699b is active with a
  10800s allowance, logging v3-cuda-35b-baseline-long.log in the owned remote root.
  Keep other GPU work away from its measurement requests.

## Ordered native submission

- Prepared submissions pin compiled code and resident views without executing work.
  Composition/decoder preparation appends source-ordered invocations, then Metal
  validates all device/buffer/scalar bindings before one command-buffer submission.
  One persistent status buffer retains any dynamic failure through later dispatches.
  All work physically completes before success/failure returns; state acceptance
  remains explicit and only successful completion makes an advance committable.
- CPU/CUDA retain sequential native invocations in this API; CUDA reports event sums,
  not a continuous batch GPU interval. Mixed backend batches reject before execution.
- Four-block decoder sequential/batched logits are exact on CPU and Metal; ownership,
  source order, canaries, resource pins after source owner destruction, status retention,
  recovery and foreign-device preflight checks pass. Workspace tests and targeted
  strict Clippy pass. M4 Pro 02 Metal decoder passes separately (an initial unfiltered
  ignored-test invocation also tried unavailable CUDA and failed, retained in logs).
- Same local real-4B explicit projection candidate with batched Metal submission gives
  ~33ms for tokens two/three, versus ~155ms with individual waits. GPU interval is
  ~30ms; top logits are unchanged. This short diagnostic is not parity qualification.
  Remaining GPU work, compiler parallelization and accountable candidate selection
  require further work. Evidence is appended to decoder-4b-integration.json.

## Proven pointwise partitioning

- Added a compiler proof/rewrite for independent rank-one dense tile computations
  over matching row coordinates. Explicit tile extents must divide the full width;
  reductions, cross-element/index-dependent expressions and unsupported control
  effects reject. Partitioning preserves statement order and every typed publication.
- Metal candidates carry this explicit choice. Bound parameters must be disjoint or
  exact aliases with equal coordinate/type mapping. Shifted overlap fails before
  writes; exact in-place execution remains legal. Single kernels, resident batches
  and the older Metal plan API now share one validation/submission implementation.
- Five partition sizes pass native arithmetic/launch-geometry/canary/alias tests.
  The four-block decoder with partitioned pointwise kernels passes CPU/Metal reference
  and exact serial/batch checks. Workspace tests and targeted runtime/engine Clippy pass.
  Broader strict compiler Clippy remains blocked by pre-existing lints in checker/HIR/
  interpreter/lowering; its diagnostics are retained, with no partition.rs findings.
- Local real4B pointwise partitions at 32 yield ~19.4ms/token; partition64 plus a single
  barrier-separated compute encoder yields ~18.5ms/token, ~15.6ms GPU. Top logits match
  prior runs. Candidate identity and diagnostic logs are retained in integration JSON.
  These experiments do not constitute automatic selection or full V3 qualification.

## Emitted dispatch/storage and native feedback

- Metal launch records now retain typed work-item/group/lane geometry, padding,
  materialized tile array placements/capacities and declared threadgroup bytes.
  Threadgroup capacity checks use typed byte counts, never parsed source strings;
  split merge launches receive the same capacity check as primary launches.
- Native compilation records execution width, maximum threads and static threadgroup
  storage, and rejects requests incompatible with the compiled pipeline/device.
  Runtime exposes this feedback without labeling declared private arrays as registers,
  live storage or occupancy. Unreported compiler temporaries/optimization remain unknown.
- Unit/padding/overflow checks, native partition/phase/status tests, decoder fixtures
  and targeted strict Clippy pass. Physical service mapping/calibration/ranking remains
  incomplete; these facts are evidence for that system, not a duration predictor.

## GGUF numerical residency and real 4B execution

- Added open GGUF artifact ownership, content identity and retained stored-value
  descriptions. Endianness/geometry/storage validation precedes execution; pathname
  replacement does not redirect retained reads. Dense import remains typed Seismic.
- Five Seismic block codecs import Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS into explicit
  packed resident planes. Raw word and half views alias one byte upload; CPU performs
  no numerical decode. Shared representation semantics now include signed, offset and
  codebook codes; CPU/CUDA scalar IR, interpreter and Metal agree on fused F32 affine
  decode. Q5/Q6 use byte code lanes; K-quant coefficients expand to F32. This preserves
  values but increases storage versus V3, so it is not memory/performance qualification.
- Reference fixture executes the exact V3 GGUFCodec methods via NumPy, retaining source
  and generator hashes. All codes/coefficients compare byte-for-byte and all decoded
  values compare numerically on CPU/local Metal/M4 Pro 01; Sparky CPU passes.
  M4 Pro 02 transfer failed for lack of disk space; no GGUF result is claimed there. Packed
  matrix checks cover all five representations and both single/multiple input rows.
- Fixed compiler ownership analysis for read-only packed accessor views and skipped
  statically empty Metal loops before allocating their nonexistent tile storage.
  Added reusable word/coefficient Metal contraction lowerings. These honor checked
  M==1/K-alignment applicability; general shapes retain their admitted lowerings.
- GGUF unaligned blocks exposed missing correlated quotient bounds. Added the sound
  common-divisor monotonicity/translation proof, with positive and out-of-bounds
  regressions. Fixed mixed-integer shift lowering and uniform checked shift semantics
  across interpreter/CPU/CUDA/Metal; invalid counts fail, including after prior valid
  invocations. Native signed/unsigned tests pass locally. Shared scalar bit-not works.
- Real 4B GGUF executes every layer on M4 Pro 01. Initial ~81ms/token improved to ~38ms
  with packed-word reuse, versus pinned V3 ~33ms in this short dense-KV diagnostic.
  Greedy choices agree for tokens 1/2/3; logits/top rankings differ slightly. No quality
  or parity gate is closed. Evidence and limits: decoder-4b-gguf-integration.json.
- Workspace tests, targeted strict Clippy and local GGUF/shift/matrix checks pass.
  Native CUDA GGUF/matrix/shift checks are queued in session 64058 until the existing
  V3 35B benchmark wrapper PID2170606 exits, preserving its measurement conditions.
  Its retry still compiles in pass1. queued-gguf-cuda.log retains progress/results.

## Routed experts and compiler publication/reduction repairs

- Added the V3 routed feedforward suffix: stable ascending top-k scores/IDs with
  larger-ID cutoff ties, optional selected normalization, dynamically selected packed
  expert banks, compact gate/up/product/down publications, ordered F32 merge and the
  separately gated dense shared expert. Routed SiLU intentionally has no separate
  compact publication before its product. Shared-router parameters remain F32.
- Dense and routed single-row decoders now share the same transactional state/mixer
  execution. Tiny alternating four-block routed CPU/Metal decoder fixtures match all
  three V3 reference steps exactly, including serial/batched execution checks. Dense
  tests retain their existing results. This is not scheduling or full engine parity.
- Five packed expert representations pass CPU/Metal projections and invalid expert-ID
  rejection. Routing tests include exact-score cutoff ties, K=1/K=E and unnormalized
  scores. Reference generators execute V3 primitive methods with source identities.
- Removed the arbitrary two-letter element-parameter limit. Local tiles may use an
  enclosing dense element parameter; specialization rejects packed allocations.
  Interpreter and native backends preserve its concrete publication precision.
  A regression exposed and fixed optimizer elimination dropping generic compact-tile
  rounding; accounting now also counts that publication and subsequent widening.
- Metal reduction assignment computes into a fresh temporary and copies to the
  existing destination, preserving assignments through conditionals rather than
  shadowing scalar/tile values. Scalar/tile regressions pass CPU and Metal.
- Workspace tests and strict runtime/engine Clippy pass. Real 35B GGUF reached its
  first attention block but rejected explicit piece64 for 49,408 bytes of shared
  storage versus 32,768 available. Kernel/shapes now accompany compiler errors.
  Attention piece32 also exceeded storage (41,216 bytes). The declared storage
  scales with work items per group, so an explicit two-item group candidate is now
  running on M4 Pro 01. This is a fit experiment, not automatic tuning. No real-35B parity gate is closed.

- The real 35B two-item attention-group candidate completed all layers and tokens
  1/2/3 on M4 Pro 01. Compile/import ~10s excluding artifact hashing; warm steps
  ~61ms host/~58ms GPU. Evidence: decoder-35b-gguf-integration.json. Pinned V3
  dense-BF16-history comparison is running separately (session37771); do not claim
  numerical/performance parity before its results are analyzed.

## Mechanically checked lower-bound specification

- Added [the lower-bound specification](../../specs/26-09-17/seismic-sound-lower-bounds.md).
  Necessary demand and capacity claims must enter proofs through checked rules and
  explicit hardware contracts; explanation strings cannot establish authority.
- The sequence is checker/trust boundary, required-transfer derivation, alternatives
  and composition, real-hardware applicability, then certified pruning/selection.
  The spec covers aliases, residency, packed representations, exact arithmetic,
  conditional claims, proof replay and independent adversarial validation.
- This checkpoint is documentation only. It implements no bound derivation and
  closes no numerical, hardware-model or optimizer qualification gate. Local links,
  Markdown fences/whitespace and design applicability were checked; no new runtime
  test results are claimed.
