# Independent Metal mapping evidence

These tools qualify hardware and native mappings outside compiler selection.
Their outputs are observations, not production hardware profiles. Write generated
JSON, captures and measurements under `../results/hardware/` (ignored). Commit
the tools and commands only; never add generated reports to Git.

`metal_services.swift` records timed arithmetic, copy, and matrix fixtures.
`qualify_services.py` computes slopes and retains their unqualified status:
loop control, native transformations, and compound conversions prevent treating
source operation counts as native instruction counts.
Passing `OUTPUT.json --archive-directory DIRECTORY` to the compiled probe also
captures each measured pipeline's source and native archive. Every observation
references its pipeline identity; archive/source hashes and public pipeline facts
are stored in the same report.

`metal_archive.swift` captures a runtime-compiled native archive even when the
command-line Metal compiler is unavailable. It uses the production runtime's
`fastMathEnabled = false` setting and records source/archive hashes, OS/device
identity, and public pipeline facts. It does not run or rank Seismic candidates.

```sh
swiftc -O metal_archive.swift -o metal-archive
./metal-archive arithmetic_mapping.metal max1 output/max1
./metal-archive arithmetic_mapping.metal max8 output/max8
```

The same fixture supplies `fma1` and `fma8`. Each capture retains its source,
native archive, and JSON report. Pipeline thread limits do not determine
resident-group capacity, register allocation, or spills.

## M4 Pro observation, 2026-09-18

`../results/hardware/m4-pro-native-mapping-2026-09-18.json` identifies all artifacts and the pinned
third-party extractor. The extracted `_agc.main` bytes for `max1` and `max8`
are identical (120 bytes each); the FMA variants differ (122 and 182 bytes).
A fixed positive cost per source max occurrence cannot describe both fixtures.
This is native byte-comparison evidence, not a timing or numerical qualification.

The pinned applegpu G13/M1 decoder cannot reliably decode these M4 binaries.
All four have failed decode markers, and `fma8` raises an assertion. Its
mnemonics and inferred resource counts must not be used. Native inspection
still needs a decoder appropriate to the target or another independently
validated interpretation. No complete qualified hardware objective exists yet.

## M3/M4 decoder and linked timing captures

The upstream [M4 issue](https://github.com/dougallj/applegpu/issues/62) points to
the [M3 branch](https://github.com/TellowKrinkle/applegpu/tree/b317f20bbf308959fa7d36246352070a8543f15b).
That pinned revision decodes the four arithmetic fixture bodies without reported
decode failures. It identifies one and eight FMA instructions in the FMA loop
bodies and one max instruction in both max loop bodies. This supersedes the
decoder-availability limitation above, not the failed G13 results or qualification
requirements.

`inspect_archives.py` checks the captured source/archive hashes before extraction
and joins native inspection with the actual timing observations. It accepts
explicit paths to an external extractor and decoder, records their identities,
and withholds mnemonic counts on decode errors. It distinguishes partially named
instructions and static main-shader counts from dynamic execution counts.

```sh
./metal-services measurements.json --archive-directory archives
python3 inspect_archives.py measurements.json archives /path/to/extractor \
  /path/to/M3/disassemble.py inspection.json
```

`../results/hardware/m4-pro-services-native-2026-09-18.json` retains 130 observations from 34 captured
pipelines. The pinned decoder handles 28 bodies without reported errors; all six
matrix bodies fail operand decoding (`Bad register size 3`). Scalar values were
checked for finiteness, while the uniform matrix fixture was checked against its
exact expected value. Scalar numerical qualification is still missing. Division
and transcendental bodies show compound instruction sequences, and some decoded
fields remain partially understood. None of these observations establishes a
complete primitive timing contract, resource occupancy, or held-out prediction.

## Scalar numerical observations

`metal_services.swift` now compares every scalar output against a stepwise host
Float reference, with explicit BF16/FP16 rounding and Darwin math functions.
It records absolute, relative and ULP error, nonfinite and unequal-bit counts,
and raw input/expected/actual bits for all 31 distinct inputs. These comparisons
do not choose a tolerance or qualify arbitrary inputs.

`../results/hardware/m4-pro-services-numerical-2026-09-18.json` records a fresh 34-pipeline,
130-observation run and the location/hashes of its complete report and archives.
All main shader bodies are byte-identical to the earlier mapping capture.
Most scalar chains match the host references bit-for-bit; logarithm and sine
chains reach maximum relative errors of approximately `8.37e-5` and `7.53e-5`,
respectively. Cosine differs by at most two ULPs. No scalar output is nonfinite.
The input set is positive and small, and the conversion chains quickly become
stationary. Matrix fixtures still use uniform exact data, and all six matrix
native decodes still fail. These results improve numerical evidence for the
observed fixtures, not full-model qualification or timing-model validity.

## Held-out chain prediction

`--held-out` measures body counts 2/4 instead of calibration counts 1/8, with
different loop lengths (scalar 512/2048, matrix 128/512). Numerical checks and
native archive capture remain enabled. `predict_chains.py CALIBRATION HELD_OUT
OUTPUT` fits only calibration data to `dispatch + iterations * (loop + count *
operation)`, then reports every held-out error without admitting any coefficient
as a hardware contract. It rejects overlapping calibration/test cases and
different device/OS identities. The fit is a deliberately limited hypothesis.

The M4 Pro result is `../results/hardware/m4-pro-heldout-chains-2026-09-18.json`: 120 predictions,
38.8% median relative error, 225.3% maximum, and only eight predictions inside
the observed three-sample ranges. Control kernels remain within roughly 1% of
their earlier medians, but copy kernels take 2.07–2.92 times as long. Device
operating-state variation is therefore a confounder; these errors cannot all be
assigned to native compiler transformations. Neither this additive hypothesis
nor the previous raw slopes justify a production timing contract.

The full held-out measurements (`metal-services-heldout.json`) and archives
(`heldout-service-archives/`) stay in the measuring machine's working directory
and are not checked in; the driver is `metal_services_heldout.swift`.
Calibration uses the preceding `metal-services-numerical.json` capture.

## Native kernel program probes (P2, P3/R6, P4)

Each probe prints one JSON document on stdout (host, device, parameters, every
configuration's samples and medians); progress goes to stderr. Run them only on
an otherwise idle GPU. Store the output under `../results/` (ignored), named
by host. Build and run them on the machine being measured.

**P3/R6 streaming-read bandwidth** (`stream_read.swift`, `stream_read.cu`). Each lane
issues `unroll` independent 16 B loads (Metal `uint4`, CUDA `ld.global.nc.v4.u32`) per
grid-stride step and adds them into a register accumulator. Each simdgroup or warp
writes one sum. For every configuration, the probe checks that the sum of all outputs
equals the closed-form wrapping sum of the GPU-filled buffer. This proves that every
byte was read exactly once.

- The sweep covers buffer sizes (default 1, 2, 4 GB), `unroll` 1/2/4/8, and 64 to 1024
  threads per group.
- The grid is either a multiple of the core or SM count (0.125x to 128x) or `full`,
  meaning one step per lane.
- Bandwidth is `bytes / median GPU time / 1e9`, taken over 10 repetitions after 2
  warmups, with one dispatch per command buffer or event pair.
- The whole sweep is reported, along with the best configuration per size and overall.
- Bytes in flight is `threads * unroll * 16`. CUDA caps the thread count at resident
  threads from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Metal has no occupancy
  query, so it reports issued grid threads.
- Metal gets its core count from the IORegistry (`AGXAccelerator` `gpu-core-count`) and
  uses a private-storage buffer. CUDA uses `cudaMalloc`.

```sh
# Apple silicon (e.g. an Apple M4 Pro)
swiftc -O stream_read.swift -o stream-read
./stream-read > ../results/bandwidth-<host>.json               # flags: --sizes-gb 1,2,4 --repetitions N
# NVIDIA (e.g. a GB10, sm_121, CUDA 13)
nvcc -O3 -arch=sm_121 stream_read.cu -o stream-read
./stream-read > ../results/bandwidth-<host>.json
```

**P4 `simdgroup_matrix` throughput** (`simdgroup_throughput.swift`, Metal source inline,
MSL 3.1, safe math). It measures three operand types (half, bfloat and float), each
accumulated in float, plus half operands with a half accumulator.

- Each threadgroup stages 16 distinct A and B fragments from device memory into
  threadgroup memory.
- In each iteration, a simdgroup loads `na` A and `nb` B fragments. Each load is indexed
  by the iteration and a runtime mask, so the fragments differ every iteration.
- The simdgroup then issues `na*nb` data-dependent chains `c = a*b + c` (1, 2, 4 or 8
  accumulators). The accumulators start from values in device memory and are all stored
  back to it.
- A one-threadgroup run is compared with a host sequential-FMA reference. Half
  accumulation rounds after every FMA step. A relative error of 1e-2 or more aborts the
  run; unequal bits are reported.
- Times are measured at 512 to 4096 iterations, interleaved per repetition. TFLOP/s is
  `simdgroups * mmas * 1024 / slope`, and `linearityMaxRelativeResidual` shows that time
  is linear in the iteration count.

```sh
# Apple silicon
swiftc -O simdgroup_throughput.swift -o simdgroup-throughput
./simdgroup-throughput > ../results/simdgroup-throughput-<host>.json   # flags: --iterations, --threadgroups-per-core
```

**P4b register-resident `simdgroup_matrix` peak** (`simdgroup_peak.swift`, same compile
options). Each simdgroup loads its `na` A and `nb` B fragments from device memory once and
then issues `na*nb` independent chains per iteration with no memory instruction in the
loop, so the slope is the matrix units' issue rate alone (P4 includes the threadgroup
fragment loads). Variants: half and bfloat operands with float or same-type accumulation,
and float; shapes 1x1 to 4x4; 1k to 8k iterations.

```sh
swiftc -O simdgroup_peak.swift -o simdgroup-peak
./simdgroup-peak > ../results/simdgroup-peak-<host>.json   # flags: --threads, --threadgroups-per-core, --iterations
```

**P2 GB10 `mma.sync` determinism and swap-AB** (`mma_determinism.cu`, inline PTX
`mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}`). One kernel computes `D = P*Q^T`
with one warp per 16x8 tile, chaining one mma per k16 step.

- **Forms:** the standard form uses activations as A (`out[m][n]`). The swapped form
  uses weights as A with the M rows in N=8 (`out[n][m]`, the K1 GEMV form).
- **Problems:** M = 1, 5, 8 and 16 with N = K = 1024, over uniform and wide-exponent
  data, for both types.
- **Determinism:** every form is rerun 10 times under four launch variants. The variants
  differ in warps per block, reversed tile order, and two concurrent half-grid launches.
  Each rerun is compared bitwise with the first run.
- **Swap-AB:** the swapped result is compared bitwise with the standard one.
- **Host models:** both forms are compared with three models, reporting unequal-bit
  count, maximum ULP and maximum absolute error. The models are sequential f32 FMA,
  and the exact k16 block sum added to the accumulator with round-to-nearest or with
  truncation. Exact sums use binary128 `long double`, so the probe builds only on an
  aarch64 Linux host.
- **Crafted cases:** single dot products that separate those models. Their results
  show the accumulation order and rounding inside the mma.

```sh
# NVIDIA GB10 (aarch64 Linux, sm_121)
nvcc -O3 -arch=sm_121 mma_determinism.cu -o mma-determinism
./mma-determinism > ../results/mma-determinism-<host>.json     # flags: --n, --k, --repetitions
```

## Vulkan device facts and driver probes (`vulkan-facts/`)

`vulkan-facts` is a standalone crate, outside the workspace. It uses `ash` and the same `glslang` crate as
formation. It answers the Vulkan backend spec's §4 floor and §16 questions. For every Vulkan device it prints
one JSON document with:

- identity, driver and UUIDs;
- subgroup, integer-dot, float-control and limit facts;
- the §4 floor features;
- cooperative-matrix shapes, atomics and ReBAR;
- memory heaps and budgets;
- queue families;
- whether the watched extensions are present.

It also runs two probes:

- **`fma`:** witness inputs (a = b = 1 + 2^-12, c = −(1 + 2^-11); fused 2^-24, two roundings 0) through
  four forms:
  - `fma()`;
  - `a*b+c`;
  - `precise a*b+c`;
  - `fma()` next to a `precise a*b+c` over the same operands.

  Each form runs as compiled by glslang (`GLSL.std.450 Fma`) and, when `VK_KHR_shader_fma` is present,
  rewritten to `OpFmaKHR`. The probe also prints the driver's ISA lines through
  `VK_KHR_pipeline_executable_properties`.
- **`shared_spec`:** a shared array sized by a specialization constant (64 B to 64 KiB). It reports whether
  the result is correct and what shared size the driver reports.

CPU devices are skipped unless you pass `--all`.

```sh
cargo build --release    # in vulkan-facts/
VULKAN_FACTS_DUMP=out ./target/release/vulkan-facts > ../../results/hardware/vulkan-facts-<host>.json
spirv-val --target-env vulkan1.3 out/fma-khr.spv                   # the rewritten module
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ./target/release/vulkan-facts --all   # lavapipe
```
