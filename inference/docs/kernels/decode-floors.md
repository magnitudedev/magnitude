# Decode floors

How far each kernel role runs from what the hardware allows, per backend and per class of weight
format, at one activation row (decode). The table orders the work on decode kernels: a cell's gap is
the time per decode step its role loses against its floor. Fractions only rank cells; no target
fraction is set here.

## Method

- **Time.** Each role's kernel entries are timed at a model's own shapes with every packed weight
  stored as one GGUF type, at the tuner's chosen configuration. Qwen3.6-35B-A3B gives attention,
  recurrent (gated-delta), routed-expert and readout entries; Qwen3.5-4B gives the dense
  feed-forward. A role's time per step is its entries' times times their calls in one decode step
  (Qwen3.6-35B-A3B: 30 recurrent and 10 attention layers, 40 routed; Qwen3.5-4B: 32 dense layers).
- **Floor.** At one row, weight bytes / the device's stream rate + one launch. Bytes are the
  entry's weight values at the type's nominal bits per weight, block scales included; the routed
  down projection reads the shared expert's down weight too. One launch is a lower
  bound on an entry's launches, so a fraction (floor / time) is an upper bound. At 512 rows,
  max(bytes / stream, 2 x values x 512 / the device's highest product rate) + one launch.
- **Machine lines** (the floors' denominators, measured by the assessment's line probes,
  `executor/src/assessment/line.rs`):
  - Apple M4 Pro, Metal: stream 267.3 GB/s (in rows of 2 KiB), launch 1.49 us, f32 matrix 8.04 TFLOPS (16.07
    TFLOPS of products for the token-packing form, which carries two products a multiply-add).
  - NVIDIA RTX 3090: stream 877.5 GB/s (CUDA) and 870.4 (Vulkan), in rows of 8 KiB, launch 1.50 us (CUDA) and 6.21
    (Vulkan), int8 251 TOPS.
  - AMD Threadripper 2990WX: stream 90.0 GB/s (interval 78.5-109.8), launch 0.18 us, f32 1.035
    TFLOPS, int8 0.975 TOPS.
- **Classes** follow from each representation's `DecodeAxes::profile()` (`seismic/lang/src/repr.rs`),
  in order: **E grid** where the value map is a grid (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S,
  IQ1_M); **D float** where the codes carry a power of two of their own (the E2M1 codebook, E4M3:
  MXFP4); **C table** where the map is a codebook (IQ4_NL, IQ4_XS); **B k-quant** where the local
  scale is an integer code under an upper factor (Q2_K to Q6_K); otherwise linear codes under a
  stored scale, **F odd** where they span at most 2 bits (Q1_0, Q2_0, TQ2_0) and **A direct** beyond
  (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1). A class cell is the mean over the class's measured types.

## What the times include

- **Warm weights off CUDA.** The tuner's timed passes rotate through copies sized from the
  device's last-level cache, and only CUDA reports one (`seismic/runtime/src/native/point.rs`,
  `cold_bytes`), while the assessment bounds an unreported cache at 256 MiB
  (`executor/src/assessment/executor_costs.rs`). Metal, Vulkan and CPU times therefore re-read
  their weights warm: an entry of a few MB can run faster than bytes / stream (Metal's 4.7 MB
  attention output reads 293 GB/s, above the M4 Pro's 273 GB/s DRAM rate). This is why Metal
  cells exceed 100%; the readout head (286-540 MB) is unaffected.
- **Vulkan device state.** On the RTX 3090 a Vulkan search's short sessions run in a slower device
  state than its rounds of 8 repetitions or more (3 to 5 times; a 286 MB `Q4_0` head read in
  `2091.6 us = 137 GB/s` against `357.1 us` in 8-repetition rounds), even with first-load tuning's
  same-session timing, so Vulkan times are read from samples of 8 repetitions or more. A type with
  none on an entry is a bound (`4r`, owned by tuner timing and the Vulkan runtime) and stays out of
  its class cell; the 14 cells left with no type are listed apart and are not ranked, and a few `8r`
  `attention_output` samples (`Q4_K`, `IQ4_NL`, `MXFP4` `W = 128`, `IQ2_XXS`) also ran in that
  slower state.
- **Causes.** Read from the kernel, representation and dispatch code each cell runs;
  microarchitectural links not directly visible in source or compiler output are marked
  `(inferred)`. Against `llama.cpp` b12242 on the 2990WX (`14336x4096`, 30 threads, 1 row), the CPU
  kernels run at `0.31-0.63x` `llama.cpp`'s time across classes A-D and F (`Q1_0` `0.31x`,
  `IQ4_NL` `0.51x`, `MXFP4` `0.53x`, `Q4_K` `0.56x`, `Q8_0` `0.63x`) and `2.65x` on `IQ2_XXS`
  (class E, where `Grid` forms neither `PANEL` nor `Codes::x86`).

## Shared mechanisms by backend

- **CPU (`x86v3` AVX2, 2990WX, `@rows8`).** (1) `RowGeometry::row_in` sets `panel = false` whenever
  `grids = true` (`seismic/native-cpu/src/weights.rs:349,398`), leaving `Grid::PANEL = None` and
  `Codes::x86 = None` (`native-cpu/src/weights.rs:1199,1257-1260,2154`); where joint `(m1, m512)` tuning also picks
  `INT8 = 1`, `M = 1` runs `tile::convert` + `Hier::tile_side` (`512` scalar `Carried::read` bit
  reads + 64-vector AVX2 `8x8` transpose into a single-use stack `Panel<8, 256>`,
  `native-cpu/src/weights.rs:3438-3465,4111-4141`, `seismic/native-cpu/src/weights/tile.rs:446-465,625-669`),
  collapsing class E to `4-5%`. (2) `Sweeps::written` zeroes per-reader slices inside each worker
  thread (`90.0 GB/s`, `executor/src/assessment/streaming_costs.rs:1864-1889`), whereas weights sit
  in one main-thread `alloc_zeroed` buffer (`seismic/backends/cpu/src/buffer.rs:114-123`) read by 30
  unpinned `fetch_add` workers (`seismic/backends/cpu/src/workers.rs:552-563,714-716`) across
  2990WX's 2 memory dies and 2 memoryless WX dies, capping cold streaming at `~48-54 GB/s`
  `(inferred)`. (3) Without VNNI (`x86v3`), signed-span `Q8_0` and `IQ4_NL` (`span > 128`) fall back
  from `Q4_K`'s `Steps::Plain` (`together = 8`) to `Steps::Sign` (`together = 1`, widening `i16 ->
  i32` every 4 weights, `seismic/native-cpu/src/weights/tile/x86.rs:68-75,139-142,415-431,475-490`), `MXFP4`
  calls scalar `E8m0::read8_each` (`native-cpu/src/weights.rs:2218-2228,2327-2338`), single-thread Launch 0
  `quant::quantize` executes 1 scalar `f64` `vdivsd` per 32-column group (`4,608` in
  `routed_output_stage`, `seismic/native-cpu/src/quant.rs:241-266`), and `Q1_0` (`1.125 bpw`)
  demands `1.280 TOPS` at `90 GB/s` against a `0.975 TOPS` `int8` ceiling.
- **CUDA (`sm_86`, RTX 3090, `@mma16`).** Every `M = 1` entry runs 1 launch (`INT8 = 1` stages only
  above `M = 16`) on swap-AB `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`7/8` `N`-columns
  empty at `M = 1`, half-rate `62.80 TFLOPS` pipe,
  `kernels/kernels/cuda/lib/projection/projection.cuh:402-404`). For all 18 non-`Records` types,
  `GemvResidency` sets `RESIDENCY = 1` whenever `GRID == 0` (`projection.cuh:132-145,1079`),
  reserving barrier ID 15 (all 16 SM barriers) and locking occupancy to **1 block/SM**: `8/48` warps
  (`16.7%` = 2 warps/scheduler) on 256-thread GEMVs and **`4/48` warps (`8.3%` = 1 warp/scheduler)**
  on 128-thread `routed` GEMVs (`kernels/kernels/cuda/lib/routed/routed.cuh:14-19`); only
  `dense_output` passes non-zero `GRID` (`kernels/kernels/cuda/dense_output.cu:54,73`). `readout`
  (`3,880` blocks = 47.3 waves) hides this at `91-95%` on `Q4_0`/`Q4_1`/`MXFP4`, whereas short
  `128-260`-block projections suffer wave-tail idle SMs, per-block `SharedNorm`,
  `gated_delta_project`'s 4-segment `switch` (`168-179r`,
  `kernels/kernels/cuda/gated_delta_project.cu:44-73,125-134`), and `routed_output`'s `36 KiB` global
  `SCRATCH_PUBLISHED` `__threadfence() + atomicAdd` combine
  (`kernels/kernels/cuda/lib/routed/output.cuh:85-129`).
- **Metal (Apple M4 Pro, `@rows32`).** Class A, `Q4_K`-`Q6_K`, Class C and `MXFP4` stream near the
  `267.3 GB/s` 2 KiB-row line (`98-103%` on Class A, with small outputs above `100%` in warm 36 MB
  SLC). `@rows32` interleaves 32 rows (`seismic/lang/src/registry.rs:69-70,1239-1241`), yet only
  `dense` exposes `TILED in [0, 1, 2]` (`kernels/kernels/dense.seismic:144-149`); the other 6
  entries hardcode `TILED = 0` (`gemv_body`, `kernels/kernels/metal/lib/projection/gemv.h:170-194`),
  and `gated_delta_project_gemv` holds `MAXM = 8` accumulators at `M = 1` without `LANES = 8`
  (`kernels/kernels/metal/gated_delta_project.metal:80-92`, `kernels/kernels/recurrent.seismic:128-134`).
  Classes E and F hit M4 Pro's scalar F32 instruction ceiling (`8.04 TFLOPS = 4.02 G-inst/s`,
  `126.5 us` per op/weight on `readout`), where `Q1_0` runs `64.8 us` slower than `Q2_0` because
  `OddCodes<1>` + unfolded `Packet::values` issue +1 FMA/weight over `Offset<1>`
  (`seismic/lang/native-library/metal/packets.h:501,508-515,1175-1177`).
- **Vulkan (RTX 3090 `NVIDIA_PROPRIETARY`, `@rows16`).** `DirectSequence::record_launch` emits a
  global `VkMemoryBarrier2(COMPUTE -> COMPUTE)` per dispatch (`6.21 us` launch floor = `35-82%` of
  small floors, `seismic/backends/vulkan/src/direct.rs:156-191`) and standalone kernel points submit
  ungraphed primary `DirectBatch` buffers (`~15 us` empirical floor, `vulkan/src/direct.rs:849-895`).
  `routed_output` unconditionally runs **2 dispatches at `M = 1`** (`routed_output_project` +
  `36.9 KB` scratch + `routed_output_combine`, `kernels/kernels/routed.seismic:496-508`), paying an
  uncredited second `6.21 us` launch (`0.248 ms/step`); `forms.rs` defines zero Vulkan consumer
  forms (`seismic/lang/src/forms.rs:91-99,145`, forcing F16 shared staging `sm = 20992 B` + scalar
  F32 `projection_dot8` FMAs, `kernels/kernels/vulkan/lib/projection/projection.glsl:607`); and
  `facts::matrix_path` disables `coopmat` on `NVIDIA_PROPRIETARY`
  (`seismic/backends/vulkan/src/facts.rs:304-305`, `1-4%` at 512 rows).

## Cells: one per role, backend and class

Floor and time are ms per decode step (the role's entries x their calls a step, mean over the class's measured types); fraction = floor / time over the class.

| role | class | backend | floor ms | time ms | fraction | gap ms/step | worst type | best type | types | fraction at 512 rows | cause |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | A direct | CUDA | 2.193 | 3.210 | 68% | 1.016 | Q4_0 65% | Q8_0 71% | 2 | 23% | `dense_expand` (`GRID = 0`, `kernels/kernels/cuda/dense_expand.cu:45-52`) locks `GemvResidency` to 1 block = 8/48 warps/SM (`kernels/kernels/cuda/lib/projection/projection.cuh:132-145,1079`), stalling `Q4_0` (`63%`) and `Q8_0` (`65%`, `TwoSteps<8>` byte-pair unpack) across a 288-block (3.5-wave) paired gate+up launch with per-block `SharedNorm`; `dense_output` passes explicit `GRID` (`kernels/kernels/cuda/dense_output.cu:54,73`), admitting `RESIDENCY = 2` (16 warps/SM) to reach `68%` (`Q4_0`) and `83%` (`Q8_0`). |
| dense | A direct | Metal | 6.344 | 6.442 | 98% | 0.097 | Q5_0 97% | Q5_1 99% | 5 | 76% | At the 267.3 GB/s stream line (`98%`: `dense_expand` `96-99%`, `dense_output` `99-102%`); `dense` exposes `TILED in [0, 1, 2]` over `@rows32` (`kernels/kernels/dense.seismic:144-149`) and decodes integer codes without int-to-float conversions via `byte_floats` (`seismic/lang/native-library/metal/packets.h:127-132,246-321`), losing only 1-4% on `Q5_0`/`Q5_1` to `Planes<4, 1>` 5th-bit `high` loads (`metal/packets.h:274-295`). |
| dense | B k-quant | CUDA | 1.245 | 2.718 | 46% | 1.473 | Q2_K 36% | Q4_K 55% | 2 | 18% | `dense_expand` (`RESIDENCY = 1`, 8/48 warps/SM) and `dense_output` (`RESIDENCY = 2`) stall on `Hier`/`CarriedHier` scale extraction (`seismic/lang/native-library/cuda/packets.cuh:1167-1260`) plus `gemv_sums<2, 16 or 32>` activation reductions (`projection.cuh:405-467`): `Q2_K` (`36%`: `56.1 us` gate+up, `26.2 us` down) runs 16-value sub-blocks (twice `Q4_K`'s `55%` scale/bias frequency) with `LowPlane<2>` shifts (`cuda/packets.cuh:286-304`). |
| dense | B k-quant | Metal | 4.888 | 5.446 | 90% | 0.558 | Q2_K 73% | Q5_K 97% | 5 | 64% | `Q4_K`/`Q5_K`/`Q6_K` run near the line (`95-100%`) using `TILED = 1`/`2` (`gemv_tile_row`/`gemv_classes`, `kernels/kernels/metal/lib/projection/gemv.h:301-380`), while `Q2_K` (`73%`) and `Q3_K` (`75%`) pull the class to `90%`: both apply `GROUP = 16` dual scale/bias per packet (`metal/packets.h:1161-1178`) over 1.7-2.2x fewer weight bytes per packet, with `Planes<2, 0>`/`Planes<2, 1>` bit-shifts (`metal/packets.h:206-244`) and `Q3_K`'s 3-byte `plane_bits<12>` scale loads (`metal/packets.h:356-371`). |
| dense | C table | CUDA | 1.548 | 2.717 | 57% | 1.169 | IQ4_NL 57% | IQ4_NL 57% | 1 | 22% | `IQ4_NL` (`57%`: `58.4 us` gate+up = `54%`, `26.5 us` down = `63%`) decodes each `k16` step through `Table::codes` (`6` `prmt.b32` nibble lookups + `4` `OpBF16::bytes` int-to-float/pack sequences = ~26 ALU ops/step vs `Q4_0`'s 4 `fma.rn.bf16x2`, `cuda/packets.cuh:107-125,597-612,875-886`), exposed under `dense_expand`'s 1-block/SM `GroupBarrier` lock (`projection.cuh:1079`). |
| dense | C table | Metal | 4.729 | 5.245 | 90% | 0.516 | IQ4_XS 89% | IQ4_NL 92% | 2 | 72% | Every 32-weight packet executes 32 scalar `constant`-cache `table_codes` lookups (`metal/packets.h:522-540`), trimming `IQ4_NL` to `92%` (`90%` paired gate+up, `94%` down); `IQ4_XS` (`89%`) adds `Hier` `plane_bits<6>` cross-byte local-scale extraction (`metal/packets.h:356-371,1080`) on top of the codebook lookups. |
| dense | D float | CUDA | 1.467 | 2.269 | 65% | 0.802 | MXFP4 65% | MXFP4 65% | 1 | 23% | `MXFP4` (`65%`: `49.7 us` gate+up = `60%`, `21.2 us` down = `74%`) decodes E2M1 without a LUT via `CodebookOf<ID, true>` bit-shifts (`cuda/packets.cuh:628-645`) and `E8M0Scale::decode` (`cuda/packets.cuh:679-686`), avoiding `IQ4_NL`'s `OpBF16::bytes` conversions; the remaining gap is `dense_expand`'s `GRID = 0` (`RESIDENCY = 1`, 8/48 warps/SM) lock and per-block `SharedNorm` (`dense_expand.cu:22-55`, `projection.cuh:1079`). |
| dense | D float | Metal | 4.597 | 5.075 | 91% | 0.478 | MXFP4 91% | MXFP4 91% | 1 | 45% | `MXFP4` (`91%`: `89%` gate+up, `95%` down) routes E2M1 nibbles through the same 32 scalar `constant`-cache `table_codes` lookups per packet as `IQ4_NL` (`seismic/lang/src/repr.rs:202-205,1494-1504`, `metal/packets.h:522-536`) plus ternary `E8M0Scale::value` branches (`metal/packets.h:706-714`); at 512 rows (`45%`), E8M0/float codes are excluded from `forms::metal_packing` (`seismic/lang/src/forms.rs:274-328`). |
| dense | E grid | CUDA | 0.681 | 2.963 | 23% | 2.282 | IQ1_S 15% | IQ2_XXS 42% | 2 | 18% | `IQ2_XXS` (`42%`: `38.9` / `17.2 us`) uses coalesced `gemv_records_group` with a 1 KiB shared-memory LUT and `RESIDENCY = 0` (`projection.cuh:840-934,1080`), whereas `IQ1_S` (`15%`: `89.4` / `39.7 us`) has `SignedBias<-3>` (`K::BIAS = true`, `lang/src/repr.rs:1760-1763`), which disqualifies it from `gemv_entries_group` in `cuda/packets.cuh:1429` into fragment-order `gemv_mma_group` (`16` random `ld.global.nc` reads/k-block/lane into the 16 KiB `__device__ const` `IQ1S` grid, `229r` on `dense_expand`, `seismic/lang/build.rs:170-180`, `cuda/packets.cuh:1832-1840`). |
| dense | E grid | Metal | 2.630 | 4.765 | 55% | 2.135 | IQ1_M 42% | IQ3_S 72% | 7 | 59% | Compute-bound on M4 Pro's scalar F32 pipe (`8.04 TFLOPS = 4.02 G-inst/s`) across paired `dense_expand` (`42-72%`, holding double-buffered `gate`+`up` packets, `gemv.h:170-194,301-333`) and `dense_output` (`42-71%`): each packet runs `Points` `constant`-cache grid lookups + `byte_floats` + `sign_bytes[16]` masks (`IQ2_*`/`IQ3_*`) or 8 `DELTA` FMAs + `plane_bits<6>` (`IQ1_M`, worst at `42%`, `metal/packets.h:590-662`). |
| dense | F odd | CUDA | 0.459 | 1.613 | 28% | 1.154 | Q1_0 28% | Q1_0 28% | 1 | 25% | `Q1_0` (`28%`: `35.3 us` gate+up = `26%`, `15.1 us` down = `35%`) is compute- and occupancy-bound: `LowPlane<1>` bit-spreading + `Odd<1>` (`2c - 1`) + 4 `fma.rn.bf16x2` feed swap-AB `mma.m16n8k16.f32` (`7/8` `N`-columns unused at `M = 1`, half-rate `62.80 TFLOPS` pipe, `projection.cuh:402-404`, `cuda/packets.cuh:305-320,801-845`) under `dense_expand`'s 1-block/SM `RESIDENCY = 1` cap (`projection.cuh:1079`). |
| dense | F odd | Metal | 2.015 | 3.504 | 58% | 1.489 | Q1_0 35% | Q2_0 72% | 3 | 63% | Bound by M4 Pro's 4.02 G-inst/s F32 issue ceiling (`58%`): `Q2_0` (`72%`) and `TQ2_0` (`67%`) fold `-s` into one `Packet::values` FMA via `Offset<1>` (`metal/packets.h:501,1175-1177`), whereas `Q1_0` (`35%`: `77.1 us` vs `71.6 us` on `Q2_0` despite half the bytes) runs `OddCodes<1>` `fma(code, 2, -1)` plus an unfolded `fma(s, code, 0)` (`metal/packets.h:508-515,1176-1177`, +1 FMA/weight). |
| dense | F odd | Vulkan | 0.763 | 3.174 | 24% | 2.410 | Q1_0 24% | Q1_0 24% | 1 | 1% | `Q1_0` (`24%`: `69.7 / 13.8 us = 20% raw` gate+up, `29.5 / 10.0 us = 34% raw` down) pays Vulkan's `6.21 us` `VkMemoryBarrier2` launch floor (`45-62%` of the 10.0-13.8 us floor, `seismic/backends/vulkan/src/direct.rs:156-191`) plus ungraphed `DirectBatch` submit overhead (`vulkan/src/direct.rs:849-895`), F16 shared staging (`GEMV_SHARED = 20992 B`, `kernels/kernels/feed_forward.seismic:86-98,197-211`), `packets_split_low1` bit-shifts and scalar F32 `projection_dot8` FMAs (`seismic/lang/native-library/vulkan/packets.glsl:291-298,993`, `kernels/kernels/vulkan/lib/projection/projection.glsl:607`); `1%` at 512 rows because `facts::matrix_path` disables `coopmat` on `NVIDIA_PROPRIETARY` (`seismic/backends/vulkan/src/facts.rs:304-305`). |
| attention | A direct | CPU | 3.222 | 7.282 | 44% | 4.060 | Q8_0 44% | Q8_0 44% | 1 | 56% | `Q8_0` (`q8g32s@rows8`, `44%`: `542.1 us` qkv = `41%` chose vs `426.9 us` raw, `186.1 us` out = `53%`) has signed span `255 > 128`, so `x86v3` (no VNNI) falls back from `Steps::Plain` (`together = 8`) to `Steps::Sign` (`together = 1`: `vpaddb + vpabsb + vpsignb + vpmaddubsw + vpmaddwd + vpaddd` every 4 weights, `seismic/native-cpu/src/weights.rs:5031-5047`, `seismic/native-cpu/src/weights/tile/x86.rs:68-75,139-142,415-422,475-490`), compounded by unpinned 30-worker `fetch_add` reads across 2990WX's 2 memory + 2 WX compute dies (`seismic/backends/cpu/src/workers.rs:552-563,714-716`) `(inferred)` and single-thread Launch 0 `f64` `vdivsd` activation quantization (`seismic/native-cpu/src/quant.rs:241-266`). |
| attention | A direct | CUDA | 0.214 | 0.393 | 55% | 0.178 | Q4_1 52% | Q4_0 58% | 2 | 21% | Single-launch `attention_project` (`192` blocks = 2.34 waves on 82 SMs, `53-59%`) and `attention_output` (`128` blocks = 1.56 waves, `51-55%`) pass `GRID = 0` (`kernels/kernels/cuda/attention_project.cu:88-139`, `attention_output.cu:43-74`), locking `GroupBarrier` to 1 block = 8/48 warps/SM (`projection.cuh:132-145,1079`) with 35-44% last-wave SM idle tail `(inferred)` and per-block `SharedNorm`; `Q4_1` (`52%`) trails `Q4_0` (`58%`) on `gemv_sums<1, 32>` nonzero-min reductions (`projection.cuh:405-467`). |
| attention | A direct | Metal | 0.782 | 0.762 | 103% | -0.020 | Q5_1 98% | Q4_1 108% | 5 | 73% | Above 100% (`103%`: qkv `97-101%`, out `102-130%`) because `4.7-6.1 MB` `attention_output` weights stay warm in M4 Pro's 36 MB SLC (`seismic/runtime/src/native/point.rs:56-60` allocates zero cold copies off CUDA) `(inferred)`, while `byte_floats` decodes Class A codes at full memory rate (`metal/packets.h:246-321`). |
| attention | B k-quant | CPU | 1.708 | 2.905 | 59% | 1.197 | Q4_K 59% | Q4_K 59% | 1 | 91% | `Q4_K` (`59%`: `221.6 us` qkv = `53%`, `68.9 us` out = `76%`) hits `x86v3`'s fastest `Steps::Plain` loop (`together = 8`: 8 `vpmaddubsw` per `vpmaddwd`, `tile/x86.rs:105-126,424-431,475-490`); `10.62 MB` qkv is capped at ~48 GB/s by unpinned 4-die WX NUMA reads from main-thread `alloc_zeroed` pages vs `Sweeps`' per-worker first-touch (`seismic/backends/cpu/src/buffer.rs:114-123`, `executor/src/assessment/streaming_costs.rs:1864-1889`) `(inferred)` plus serial Launch 0 `f64` `quant::quantize` (`quant.rs:241-266`), while `4.72 MB` `attention_output` fits warm in 64 MB L3 `(inferred)`. |
| attention | B k-quant | CUDA | 0.167 | 0.427 | 39% | 0.261 | Q2_K 31% | Q4_K 50% | 3 | 18% | Locked to 1 block = 8/48 warps/SM (`GRID = 0`, `RESIDENCY = 1`, `projection.cuh:1079`) across 1.56-2.34 waves, `Hier`/`CarriedHier` scale bit-unpacking (`cuda/packets.cuh:1167-1260`) and `gemv_sums` bias reductions stall on 2 warps/scheduler: `Q2_K` (`31%`) decodes 16-value sub-blocks (2x scale/min frequency), `Q3_K` (`38%`) reconstructs carried 2+1-bit codes + 6-bit scales, and `Q4_K` reaches `50%`. |
| attention | B k-quant | Metal | 0.607 | 0.697 | 87% | 0.090 | Q2_K 67% | Q6_K 99% | 5 | 63% | `Q4_K`/`Q5_K`/`Q6_K` hit `93-99%` on qkv and `102-109%` (warm SLC) on out, while `Q2_K` (`67%`: `37.4` / `17.3 us`) and `Q3_K` (`68%`: `48.1` / `21.4 us`) pull the class to `87%`: `TILED = 0` hardcoded on both entries (`kernels/kernels/metal/attention_project.metal:136`, `attention_output.metal:79`) reloads `Hier` headers every strided `@rows32` packet on top of `GROUP = 16` dual scale/min and `Q3_K` `plane_bits<12>` byte loads (`metal/packets.h:356-371,1161-1178`). |
| attention | B k-quant | Vulkan | 0.262 | 0.616 | 43% | 0.354 | Q4_K 37% | Q3_K 53% | 3 | 3% | Single-dispatch `L1` GEMVs (`kernels/kernels/attention.seismic:233-251,550-567`) pay Vulkan's `6.21 us` `VkMemoryBarrier2` launch (`40-66%` of 9.4-18.4 us floors, `vulkan/src/direct.rs:156-191`), F16 shared staging (`sm = 20992 B`) and scalar F32 `projection_dot8` FMAs (`seismic/lang/src/forms.rs:91-99,145`, `projection.glsl:607`); `Q2_K` (`42%`) pays `PACKETS_TWO_BIASES` (`group = 16`, `vulkan/packets.glsl:187-235`), `Q3_K` reaches `53%`, and `Q4_K` (`37%`) is dragged down by a slow-state `8r` `attention_output` sample (`52.6 us = 22%` with `SPLIT = 2` vs `29.6 us = 62%` on qkv). |
| attention | C table | CPU | 1.708 | 2.907 | 59% | 1.199 | IQ4_NL 59% | IQ4_NL 59% | 1 | 55% | `IQ4_NL` (`59%`: `221.5 us` qkv = `53%`, `69.2 us` out = `76%`) decodes via `vpshufb` (`tile.rs:305-321`) + `Steps::Sign` (`span = 240 > 128`, `together = 1`, `tile/x86.rs:139-142,415-422,475-490`, costing `55%` vs `Q4_K`'s `91%` at 512 rows); at 1 row, ALU hides behind the same ~48 GB/s unpinned 4-die WX NUMA ceiling on qkv and warm L3 + serial `f64` Launch 0 quantize on out as `Q4_K` (`cpu/src/workers.rs:552-563`, `quant.rs:241-266`) `(inferred)`. |
| attention | C table | CUDA | 0.200 | 0.426 | 47% | 0.227 | IQ4_XS 46% | IQ4_NL 48% | 2 | 22% | Under `RESIDENCY = 1` (8/48 warps/SM, 1.56-2.34 waves, `projection.cuh:1079`), `IQ4_NL` (`48%`) spends 6 `prmt.b32` + 4 `OpBF16::bytes` (~26 ALU ops/step, `cuda/packets.cuh:107-125,597-612,875-886`), and `IQ4_XS` (`46%`) adds `Hier` 6-bit scale unpacking (`3` `u16` loads + 64-bit shifts, `cuda/packets.cuh:1248-1290`). |
| attention | C table | Metal | 0.588 | 0.647 | 91% | 0.060 | IQ4_XS 88% | IQ4_NL 93% | 2 | 68% | 32 scalar `constant`-cache `table_codes` lookups per packet (`metal/packets.h:522-540`) plus hardcoded `TILED = 0` strided `@rows32` accesses (`attention_project.metal:136`, `attention_output.metal:79`) put `IQ4_NL` at `93%` (`91%` qkv, `100%` warm-SLC out) and `IQ4_XS` (`plane_bits<6>` `Hier` local scales, `metal/packets.h:356-371,1080`) at `88%`. |
| attention | C table | Vulkan | 0.295 | 0.559 | 53% | 0.264 | IQ4_NL 46% | IQ4_XS 62% | 2 | 3% | `attention_project` hits `62%` (`IQ4_NL`) and `59%` (`IQ4_XS`) behind Vulkan's `6.21 us` barrier/launch floor + scalar F32 `projection_dot8` + 16-entry `tables.glsl` lookup (`vulkan/packets.glsl:381-383,710-725`); the class drops to `53%` because `IQ4_NL` `attention_output` recorded `35.2 us` (`33%`) in a slower `8r` device state vs `17.1 us` (`66%`) on `IQ4_XS`. |
| attention | D float | CPU | 1.613 | 3.041 | 53% | 1.428 | MXFP4 53% | MXFP4 53% | 1 | 70% | `MXFP4` (`53%`: `212.5 us` qkv = `53%`, `91.6 us` out = `54%`) decodes E2M1 nibbles via `vpshufb` + `Steps::Sign` (`together = 1`), but `E8m0::x86_groups` is unimplemented so every 8-group tile calls scalar `E8m0::read8_each` (`8` scalar IEEE-754 bit-hacks, `native-cpu/src/weights.rs:2218-2228,2327-2338`), losing `22.7 us` on L3-warm `attention_output` (`91.6` vs `68.9 us` on `Q4_K`) on top of unpinned 4-die NUMA streaming and serial Launch 0 `f64` quantize. |
| attention | D float | CUDA | 0.195 | 0.354 | 55% | 0.159 | MXFP4 55% | MXFP4 55% | 1 | 21% | `MXFP4` (`55%`: `23.2 us` qkv = `56%`, `12.2 us` out = `54%`) decodes fast via tableless `CodebookOf<ID, true>` E2M1 shifts + `E8M0Scale` (`cuda/packets.cuh:628-645,679-686`; `92%` on `readout`), so the 45% gap here is pure short-grid latency: `RESIDENCY = 1` (8/48 warps/SM, `projection.cuh:1079`), 1.56-2.34-wave tail on 82 SMs `(inferred)`, and per-block `SharedNorm` (`attention_project.cu:88-139`, `attention_output.cu:43-74`). |
| attention | D float | Metal | 0.572 | 0.629 | 91% | 0.057 | MXFP4 91% | MXFP4 91% | 1 | 45% | `MXFP4` (`91%`: `89%` qkv, `95%` out) runs 32 scalar `constant`-cache `table_codes` lookups per packet (`metal/packets.h:522-536`) plus ternary `E8M0Scale::value` decode (`metal/packets.h:706-714`) with `TILED = 0` hardcoded (`attention_project.metal:136`); `45%` at 512 rows because E8M0/float codes bypass `forms::metal_packing` (`lang/src/forms.rs:274-328`). |
| attention | D float | Vulkan | 0.291 | 0.458 | 63% | 0.167 | MXFP4 63% | MXFP4 63% | 1 | 4% | `MXFP4` (`63%`: `30.0 / 17.7 us = 59%` qkv, `15.8 / 11.3 us = 72%` out) is Vulkan's highest ranked cell: E2M1 uses the 16-float `tables.glsl` codebook + `packets_e8m0` `ldexp` (`vulkan/packets.glsl:381-383,557`), with the remaining gap set by the `6.21 us` `VkMemoryBarrier2` launch floor (`35-55%` of floor, `vulkan/src/direct.rs:156-191`), F16 shared staging (`sm = 20992 B`), and scalar F32 `projection_dot8` (`projection.glsl:607`). |
| attention | E grid | CPU | 0.785 | 19.413 | 4% | 18.628 | IQ2_XXS 4% | IQ2_XXS 4% | 1 | 21% | `IQ2_XXS` (`4%`, `18.628 ms/step` gap: `1386.4 us` qkv, `554.9 us` out) hits a double cliff: `RowGeometry::row_in` sets `panel = false` for `grids` (`native-cpu/src/weights.rs:349,398`), leaving `Grid::PANEL = None` and `Codes::x86 = None` (`native-cpu/src/weights.rs:1199,1257-1260,2154`), while joint `(m1, m512)` tuning picked `INT8 = 1`, routing `M = 1` through `tile::convert` + `Hier::tile_side` (`512` scalar `Carried::read` bit-reads + `IQ2XXS_GRID` u64 unpacks + 64-vector AVX2 `8x8` transpose into a one-shot stack `Panel<8, 256>`, `native-cpu/src/weights.rs:3438-3465,4111-4141`, `tile.rs:446-465,625-669`). |
| attention | E grid | CUDA | 0.123 | 0.505 | 24% | 0.382 | IQ1_S 15% | IQ2_XXS 36% | 7 | 16% | 3-tier grid gap (`24%` mean): `IQ2_XXS` (`36%`) uses `gemv_records_group` (1 KiB shared LUT, `RESIDENCY = 0`); `IQ2_XS/S`, `IQ3_*`, `IQ1_M` (`18-31%`) use unpipelined `gemv_entries_group` (`load(sb + 1)` after decode, `RESIDENCY = 1`, random `ld.global.nc` into 2-16 KiB `__device__ const` global grids, `projection.cuh:758-825`, `seismic/lang/build.rs:170-180`); `IQ1_S` (`15%`) is disqualified by `SignedBias` (`K::BIAS = true`, `cuda/packets.cuh:1429`) into fragment-order `gemv_mma_group` (`16` global 16 KiB LUT reads/k-block/lane). |
| attention | E grid | Metal | 0.335 | 0.643 | 52% | 0.309 | IQ1_M 40% | IQ3_S 67% | 7 | 58% | Compute-bound on M4 Pro's scalar F32 pipe (`52%`: `IQ1_M 40%` to `IQ3_S 67%`), worsened by hardcoded `TILED = 0` strided `@rows32` loads and `attention_output_gemv` choosing `LANES = 32` (`32`-lane `simd_sum` across a small 128-256-thread grid, stalling at `17.8-22.2 us`); `IQ1_M` adds 8 `DELTA` FMAs + `plane_bits<6>` (`metal/packets.h:356-371,590-662`). |
| attention | E grid | Vulkan | 0.218 | 0.557 | 39% | 0.339 | IQ2_XXS 26% | IQ2_S 47% | 7 | 3% | Tiny 8.1-15.5 us floors (`6.21 us` launch = `40-77%` of floor, `vulkan/src/direct.rs:156-191`) combine with heavy `SEISMIC_CODES_GRID` ALU (`const` `grids_*` lookups, `unpack8`, 8 sign shifts/XORs + scalar F32 FMAs, `vulkan/packets.glsl:446-483,799-900,947-965`); `IQ3_S` (`44.7 us` qkv = `35%`) loads 3 `u32` + 1 `u16` planes with dual `PACKETS_GRID_POINT4` lookups, while `IQ2_XXS` (`26%`) reflects a slow-state `8r` `attention_output` sample (`46.5 us = 19%`). |
| attention | F odd | CPU | 0.430 | 2.387 | 18% | 1.957 | Q1_0 18% | Q1_0 18% | 1 | 72% | `Q1_0` (`18%`: `183.3 us` qkv = `16%`, `55.4 us` out = `24% raw`) is compute-bound against the 2990WX `0.975 TOPS` `int8` ceiling (`1.125 bpw` at `90 GB/s` needs `1.280 TOPS`): `Odd<1>` (`vpshufb + vpand + vpcmpeqb`) + `Steps::Sign` (`vpsignb + vpmaddubsw + vpmaddwd + vpaddd`) execute ~7 AVX2 ops per 64 OPs (`native-cpu/src/weights.rs:1780-1789,5093-5103`, `tile/x86.rs:462-473`), plus single-thread Launch 0 RMSNorm + `6,144` scalar `f64` divisions (`quant.rs:241-266`). |
| attention | F odd | CUDA | 0.100 | 0.287 | 35% | 0.186 | Q1_0 26% | Q2_0 39% | 3 | 23% | All three types take `19.4-20.1 us` on qkv and `8.6-9.8 us` on out regardless of byte size (`Q1_0 26%`, `TQ2_0 38%`, `Q2_0 39%`): bound by 1-block/SM `RESIDENCY = 1` (8/48 warps/SM, `projection.cuh:1079`), 1.56-2.34-wave grid tail, per-block `SharedNorm`, and swap-AB `mma.m16n8k16.f32` (`7/8` empty `N`-cols) + `LowPlane<1/2>`/`Odd<1>` SIMT bit-unpack (`cuda/packets.cuh:286-320,801-845`). |
| attention | F odd | Metal | 0.261 | 0.484 | 54% | 0.223 | Q1_0 35% | Q2_0 65% | 3 | 62% | F32 instruction-rate bound (`54%`) plus `SharedNorm` prologue + `LANES = 32` reduction stall on `attention_output_gemv` (`15.6-16.5 us` flat): `Q1_0` (`35%`) runs slower (`33.1` / `16.5 us`) than `Q2_0` (`65%`, `32.9` / `15.6 us`) because `OddCodes<1>` + unfolded `Packet::values` issue +1 FMA/weight over `Offset<1>` (`metal/packets.h:501,508-515,1175-1177`). |
| attention | F odd | Vulkan | 0.195 | 0.445 | 44% | 0.250 | Q1_0 39% | Q2_0 47% | 3 | 4% | All three types cluster at `28.1-28.6 us` (qkv) and `15.0-16.4 us` (out) (`Q1_0 39%`, `TQ2_0 45%`, `Q2_0 47%`): ideal weight stream time is only `1.4-6.1 us`, so Vulkan's `6.21 us` `VkMemoryBarrier2` floor + ungraphed `DirectBatch` submit (`~15 us` min, `vulkan/src/direct.rs:156-191,849-895`) + `packets_split_low1`/`low2` bit-shifts and scalar F32 FMAs (`vulkan/packets.glsl:282-298,982-995`) dominate. |
| recurrent | A direct | CPU | 11.895 | 31.884 | 37% | 19.989 | Q8_0 37% | Q8_0 37% | 1 | 53% | `Q8_0` (`37%`, `19.989 ms/step` gap) is dominated by `gated_delta_project` (`895.1 / 297.3 us = 33%` chose vs `626.8 us = 47%` raw): joint `(m1, m512)` tuning picked `ROWS = 8` for `M = 512`, inflating `M = 1` row-chunk tail and cross-die traffic over `ROWS = 4` (`626.8 us`), while signed `Q8_0` forces non-VNNI `Steps::Sign` (`together = 1`, `tile/x86.rs:68-75,415-422,475-490`) across 4 unpinned 2990WX NUMA dies (`cpu/src/workers.rs:552-563`) `(inferred)` (`gated_delta_output` hits `59% raw` in L3). |
| recurrent | A direct | CUDA | 0.771 | 1.473 | 52% | 0.702 | Q4_1 50% | Q4_0 56% | 2 | 19% | `gated_delta_project` (`49-55%`, `260` blocks) compiles a 4-segment (`qkv/z/a/b`) `switch` inside one kernel (`kernels/kernels/cuda/gated_delta_project.cu:44-73,125-134`), bloating `L1` to `168-174r` (1 block = 8/48 warps/SM even without `GroupBarrier`) with `a`/`b` blocks computing 1-2 rows (`1/16` of a 16-row MMA tile); `attention_output` (`W = 128`, `52-57%`) is capped at 1 block/SM (`GRID = 0`, 128 blocks = 1.56 waves, `projection.cuh:1079`). |
| recurrent | A direct | Metal | 2.867 | 2.886 | 99% | 0.019 | Q5_0 97% | Q4_1 101% | 5 | 70% | Near the 267.3 GB/s line (`99%`: out `106-126%` warm SLC, input `92-98%`); `gated_delta_project_gemv` trails `attention_project` by 4-8 points across every class because `kernels/kernels/metal/gated_delta_project.metal:80-92` uses one `PROJECTION_FOR_ROWS` (`MAXM = 8` live accumulators at `M = 1` across 4 `gemv` instantiations), hardcodes `TILED = 0`, and omits `LANES = 8` (`kernels/kernels/recurrent.seismic:128-134`). |
| recurrent | B k-quant | CPU | 6.302 | 11.604 | 54% | 5.302 | Q4_K 54% | Q4_K 54% | 1 | 78% | `Q4_K` (`54%`, `5.302 ms/step` gap) pairs fast `Steps::Plain` (`together = 8`, `tile/x86.rs:105-126,424-431,475-490`) with L3-warm `gated_delta_output` (`4.72 MB`, `60.6 / 52.6 us = 87%`), while `14.16 MB` `gated_delta_project` (`326.2 / 157.5 us = 48%`, `43.4 GB/s`) is capped by unpinned 30-worker Infinity Fabric DRAM reads across 4 WX NUMA dies (`cpu/src/workers.rs:552-563`, `cpu/src/buffer.rs:114-123`) `(inferred)` plus 4-segment dispatch (`kernels/kernels/cpu/gated_delta_project.rs:59-88`) and serial Launch 0 `f64` quantize (`quant.rs:241-266`). |
| recurrent | B k-quant | CUDA | 0.595 | 1.690 | 35% | 1.095 | Q2_K 27% | Q4_K 45% | 3 | 18% | 1-block/SM occupancy (`168-179r` from `gated_delta_project`'s 4-segment `switch` + `RESIDENCY = 1` on output, `gated_delta_project.cu:44-73,125-134`, `projection.cuh:1079`) exposes `Hier`/`CarriedHier` scale bit-unpacking and `gemv_sums` latency on 2 warps/scheduler: `Q2_K` (`27%`: `43.6` / `13.8 us`) runs 16-value sub-blocks, `Q3_K` (`35%`) 2+1-bit planes, and `Q4_K` `45%`. |
| recurrent | B k-quant | Metal | 2.220 | 2.608 | 85% | 0.389 | Q2_K 65% | Q6_K 96% | 5 | 61% | `Q4_K`/`Q5_K`/`Q6_K` average `92-96%` (`90-95%` input, `102-108%` warm-SLC output), whereas `Q2_K` (`65%`: `51.2` / `17.2 us`) and `Q3_K` (`67%`: `62.9` / `21.0 us`) pull the class to `85%`, combining `GROUP = 16` dual scale/min + `Q3_K` `plane_bits<12>` loads (`metal/packets.h:356-371,1161-1178`) with `gated_delta_project_gemv`'s `TILED = 0`, `MAXM = 8` register bloat, and forced `LANES >= 16` (`gated_delta_project.metal:80-92`, `recurrent.seismic:128-134`). |
| recurrent | B k-quant | Vulkan | 0.752 | 1.679 | 45% | 0.927 | Q2_K 45% | Q2_K 45% | 1 | 4% | Only `Q2_K` is timed in device state on both entries (`45%`: `39.2 / 15.7 us = 40% raw` input, `16.8 / 9.4 us = 56% raw` out; `Q3_K`/`Q4_K` input stayed `4r` after 30s SPIR-V compiles): `q2k@rows16` evaluates `PACKETS_TWO_BIASES` (`group = 16`, `vulkan/packets.glsl:187-235`) with scalar F32 `projection_dot8` + F16 shared staging (`sm = 20992 B`, `recurrent.seismic:222-238`) atop the `6.21 us` `VkMemoryBarrier2` launch floor (`vulkan/src/direct.rs:156-191`). |
| recurrent | C table | CPU | 6.302 | 10.941 | 58% | 4.639 | IQ4_NL 58% | IQ4_NL 58% | 1 | 54% | `IQ4_NL` (`58%`, `4.639 ms/step` gap: `275.7 / 157.5 us = 57% raw` input vs `415.7 us` chose `ROWS = 8`; `89.0 / 52.6 us = 59%` out) hits the ~54 GB/s unpinned 4-die WX NUMA DRAM ceiling on `14.16 MB` input `(inferred)`, while L3-warm `gated_delta_output` loses `28.4 us` vs `Q4_K` (`89.0` vs `60.6 us`) to `vpshufb` + `Steps::Sign` (`together = 1` vs `Q4_K`'s `together = 8`, `tile/x86.rs:139-142,415-422,475-490`) and serial `4,096`-group `f64` Launch 0 quantize (`quant.rs:241-266`). |
| recurrent | C table | CUDA | 0.717 | 1.707 | 42% | 0.990 | IQ4_XS 41% | IQ4_NL 43% | 2 | 18% | `IQ4_NL` (`43%`) and `IQ4_XS` (`41%`) run 6 `prmt.b32` + 4 `OpBF16::bytes` (~26 ALU ops/step, `cuda/packets.cuh:107-125,597-612,875-886`, plus `Hier` 6-bit scale shifts on `IQ4_XS`) inside 1-block/SM `gated_delta_project` (`168-174r` 4-segment `switch`, `gated_delta_project.cu:44-73,125-134`) and 1.56-wave `RESIDENCY = 1` output (`projection.cuh:1079`). |
| recurrent | C table | Metal | 2.149 | 2.522 | 85% | 0.373 | IQ4_XS 82% | IQ4_NL 88% | 2 | 66% | 32 scalar `constant`-cache `table_codes` lookups per packet (`metal/packets.h:522-540`, plus `plane_bits<6>` `Hier` local scales on `IQ4_XS`, `metal/packets.h:356-371,1080`) combine with `gated_delta_project_gemv`'s `MAXM = 8` register footprint, hardcoded `TILED = 0`, and missing `LANES = 8` (`gated_delta_project.metal:80-92`, `recurrent.seismic:128-134`) to hold `IQ4_NL` at `88%` (`85%` input) and `IQ4_XS` at `82%` (`80%` input). |
| recurrent | C table | Vulkan | 1.005 | 2.281 | 44% | 1.276 | IQ4_XS 38% | IQ4_NL 53% | 2 | 2% | `IQ4_NL` (`53%`: `39.1 us = 57%` input, slow-state `8r` `25.2 us = 46% raw` out) beats `IQ4_XS` (`38%`: `59.3 us = 36% raw` input, `28.4 us = 40% raw` out) because `IQ4_XS` adds unaligned `SEISMIC_COEFF_HIER` 6-bit byte loads (`vulkan/packets.glsl:710-725`) and a 37.4s `vkCreateComputePipelines` compile exhausted `gated_delta_project`'s budget after timing only challenger `SIMDGROUPS = 4` (`t = 128`, 47% slower than `SIMDGROUPS = 16`). |
| recurrent | D float | CPU | 5.953 | 11.153 | 53% | 5.201 | MXFP4 53% | MXFP4 53% | 1 | 66% | `MXFP4` (`53%`, `5.201 ms/step` gap: `269.0 / 148.7 us = 55% raw` input, `102.8 / 49.7 us = 48%` out) hits the unpinned 4-die WX NUMA ceiling on `13.37 MB` input `(inferred)` while losing `42.2 us` on L3-warm `gated_delta_output` (`102.8` vs `60.6 us` on `Q4_K`) to unvectorized scalar `E8m0::read8_each` (`native-cpu/src/weights.rs:2218-2228,2327-2338`) + `vpshufb` + `Steps::Sign` (`together = 1`, `tile/x86.rs:139-142,415-422,475-490`). |
| recurrent | D float | CUDA | 0.699 | 1.335 | 52% | 0.636 | MXFP4 52% | MXFP4 52% | 1 | 19% | `MXFP4` (`52%`: `32.0 / 16.7 us = 52%` input, `12.5 / 6.6 us = 53%` out) matches `Q4_0` to within 0.4 us: E2M1 + E8M0 decode is cheap (`92%` on `readout`), so the gap is `gated_delta_project`'s 4-segment `switch` (`168r` -> 1 block = 8/48 warps/SM + sparse 1-2-row `a`/`b` tiles, `gated_delta_project.cu:44-73,125-134`) and `attention_output`'s `RESIDENCY = 1` 1.56-wave grid (`projection.cuh:1079`). |
| recurrent | D float | Metal | 2.090 | 2.472 | 85% | 0.382 | MXFP4 85% | MXFP4 85% | 1 | 45% | `MXFP4` (`85%`: `81%` input, `96%` warm-SLC out) pays 32 scalar `constant` `table_codes` lookups + ternary `E8M0Scale::value` per packet (`metal/packets.h:522-536,706-714`) on top of `gated_delta_project_gemv`'s `MAXM = 8` register pressure, `TILED = 0`, and missing `LANES = 8` (`gated_delta_project.metal:80-92`, `recurrent.seismic:128-134`); `45%` at 512 rows without `metal_packing` (`lang/src/forms.rs:274-328`). |
| recurrent | D float | Vulkan | 0.987 | 2.645 | 37% | 1.658 | MXFP4 37% | MXFP4 37% | 1 | 1% | `MXFP4` (`37%`: `41.8 / 21.6 us = 52% raw` input, `46.4 / 11.3 us = 24% raw` out) combines `gated_delta_project`'s 4-segment F32 GEMV (`recurrent.seismic:222-238`, `6.21 us` `VkMemoryBarrier2` floor + `tables.glsl`/`packets_e8m0` decode, `vulkan/packets.glsl:381-383,557`) with a slow-state 8-repetition `attention_output` (`W = 128`) sample (`46.4 us` against `17.4 us` in device state). |
| recurrent | E grid | CPU | 2.894 | 55.342 | 5% | 52.448 | IQ2_XXS 5% | IQ2_XXS 5% | 1 | 23% | `IQ2_XXS` (`5%`, `52.448 ms/step` gap — #2 overall: `1321.8 us raw` input, `522.9 us` out) has `Grid::PANEL = None` and `Codes::x86 = None` (`native-cpu/src/weights.rs:349,398,1199,1257-1260,2154`), while joint `(m1, m512)` tuning chose `INT8 = 1`, forcing `M = 1` through `tile::convert` + `Hier::tile_side` (`512` scalar `Carried::read` bit-gathers + `IQ2XXS_GRID` u64 unpack + 64-vector AVX2 `8x8` transpose into a single-use stack `Panel`, `native-cpu/src/weights.rs:3438-3465,4111-4141`, `tile.rs:446-465,625-669`). |
| recurrent | E grid | CUDA | 0.433 | 2.038 | 21% | 1.605 | IQ1_S 13% | IQ2_XXS 32% | 7 | 16% | `21%` mean across 7 types: even `IQ2_XXS` (`32%`) is forced down to 1 block/SM on `gated_delta_project` because the 4-segment `switch` bloats `gemv_records_group` to `168r` (`gated_delta_project.cu:44-73,125-134`), `IQ2_XS/S`, `IQ3_*`, `IQ1_M` (`15-26%`) run unpipelined `gemv_entries_group` over global `__device__ const` grids (`projection.cuh:758-825`, `seismic/lang/build.rs:170-180`), and `IQ1_S` (`13%`, `61.2 us` input) falls back to fragment-order `gemv_mma_group` via `K::BIAS = true` (`cuda/packets.cuh:1429`). |
| recurrent | E grid | Metal | 1.216 | 2.394 | 51% | 1.178 | IQ1_M 39% | IQ3_S 65% | 7 | 57% | Scalar F32 decode-ALU ceiling (`51%`: `IQ1_M 39%` to `IQ3_S 65%`) compounded by `gated_delta_project_gemv`'s `MAXM = 8` register footprint, hardcoded `TILED = 0` strided `@rows32` loads, and `LANES >= 16` (`gated_delta_project.metal:80-92`, `recurrent.seismic:128-134`) plus `Points` `constant` LUT + `sign_bytes[16]`/`DELTA` ALU (`metal/packets.h:590-662`). |
| recurrent | E grid | Vulkan | 0.734 | 1.932 | 38% | 1.198 | IQ3_XXS 35% | IQ2_S 41% | 4 | 4% | `38%` over 4 device-state-timed types (`IQ3_XXS 35%` to `IQ2_S 41%`; `IQ3_S`/`IQ1_S`/`IQ1_M` input stayed `4r` under 30s SPIR-V compiles): `6.21 us` `VkMemoryBarrier2` floor (`36-71%` of 8.7-17.3 us floors, `vulkan/src/direct.rs:156-191`) + 4-segment `gated_delta_project` dispatch (`recurrent.seismic:222-238`) + `SEISMIC_CODES_GRID` `const` LUT/sign-mask/F32 FMA decode (`vulkan/packets.glsl:446-483,947-965`, `r = 100` on `IQ3_XXS` `ROWS = 4`). |
| recurrent | F odd | CPU | 1.584 | 7.256 | 22% | 5.672 | Q1_0 22% | Q1_0 22% | 1 | 68% | `Q1_0` (`22%`, `5.672 ms/step` gap: `186.1 / 39.5 us = 21% raw` input, `55.8 / 13.3 us = 24% raw` out) is compute-bound against 2990WX's `0.975 TOPS` `int8` peak (`1.280 TOPS` needed at `90 GB/s`) + ~7 AVX2 ops per 64 OPs in `Odd<1>` + `Steps::Sign` (`native-cpu/src/weights.rs:1780-1789,5093-5103`, `tile/x86.rs:462-473`) + serial Launch 0 `f64` divisions (`2,048` in input, `4,096` in out, `quant.rs:241-266`). |
| recurrent | F odd | CUDA | 0.350 | 1.139 | 31% | 0.789 | Q1_0 22% | Q2_0 35% | 3 | 19% | `Q1_0`, `Q2_0`, `TQ2_0` all take `28.1-29.7 us` on input and `8.7-9.6 us` on output (`Q1_0 22%`, `TQ2_0 35%`, `Q2_0 35%`): invariant to weight bytes because 1-block/SM occupancy (`168r` 4-segment `switch` on input + `RESIDENCY = 1` on output, `gated_delta_project.cu:44-73,125-134`, `projection.cuh:1079`) + swap-AB `mma.m16n8k16.f32` (`7/8` empty `N`-cols) + `LowPlane`/`Odd` SIMT unpack set a fixed compute/latency floor. |
| recurrent | F odd | Metal | 0.943 | 1.801 | 52% | 0.858 | Q1_0 35% | Q2_0 63% | 3 | 60% | Bound by M4 Pro's 4.02 G-inst/s F32 issue ceiling plus `gated_delta_project_gemv` (`MAXM = 8`, `TILED = 0`, `LANES >= 16`, `gated_delta_project.metal:80-92`) and `SharedNorm` prologue (`52%`): `Q1_0` (`35%`, `42.8` / `16.5 us`) trails `Q2_0` (`63%`, `45.6` / `15.6 us`) on `OddCodes<1>` + unfolded `Packet::values` (+1 FMA/weight, `metal/packets.h:508-515,1175-1177`). |
| recurrent | F odd | Vulkan | 0.635 | 1.651 | 38% | 1.016 | Q1_0 33% | Q2_0 41% | 3 | 4% | `Q1_0` (`33%`), `TQ2_0` (`40%`), and `Q2_0` (`41%`) all finish in `34.6-36.6 us` (input) and `18.2-19.7 us` (out): ideal stream time is `1.4-8.1 us`, so the `6.21 us` `VkMemoryBarrier2` launch floor + ungraphed `DirectBatch` submit (`vulkan/src/direct.rs:156-191,849-895`) + 4-segment `gated_delta_project` + `packets_split_low1`/`low2` shifts and scalar F32 FMAs (`vulkan/packets.glsl:282-298,982-995`) dominate. |
| routed | A direct | CPU | 13.384 | 26.464 | 51% | 13.080 | Q8_0 51% | Q8_0 51% | 1 | - | `Q8_0` (`51%`, `13.080 ms/step` gap: `450.9 / 223.0 us = 49%` expand, `210.7 / 111.6 us = 53%` down) runs non-VNNI `Steps::Sign` (`together = 1`, `tile/x86.rs:68-75,415-422,475-490`) over 9 scattered expert + shared slices (`F = 512` short rows) across 4 unpinned 2990WX NUMA dies (`cpu/src/workers.rs:552-563`) `(inferred)`, plus single-thread Launch 0 `f64` `vdivsd` activation quantization (`2,048` divs in expand, `4,608` divs in `routed_output_stage`, `kernels/kernels/cpu/routed_expand.rs:16-38`, `routed_output.rs:19-48`, `quant.rs:241-266`). |
| routed | A direct | CUDA | 0.830 | 1.506 | 55% | 0.676 | Q4_1 53% | Q4_0 57% | 2 | - | `routed_gate_up` (`54-60%`) declares `DECODE_WARPS = 4` (128 threads, `kernels/kernels/cuda/lib/routed/routed.cuh:14-19`) with `GRID = 0` (`RESIDENCY = 1`, `projection.cuh:1079`), locking occupancy to **4/48 warps/SM (`8.3%` = 1 warp/scheduler)** over `128` blocks (1.56 waves); `routed_output` (`52-54%`, 4/48 warps/SM, `144` blocks) writes 9 slots to global `SCRATCH_PUBLISHED` (`36 KiB`), executes `__threadfence() + atomicAdd`, and re-reads all 9 slots to sum (`kernels/kernels/cuda/lib/routed/output.cuh:85-129`, `arrive.cuh:18-31`, `output_sum.cuh:15-33`). |
| routed | A direct | Metal | 3.012 | 3.026 | 100% | 0.014 | Q5_1 98% | Q4_0 102% | 5 | - | At the 267.3 GB/s stream line (`100%`: gate+up `95-98%`, down `101-111%`); `routed_output_resident` (`kernels/kernels/metal/lib/routed/channels.h:75-111`) runs barrier-free (`STAGE = 0` at `LANES = 16`, 1 packet/lane in registers) with warm SLC reuse on the shared expert `(inferred)`, while `routed_gate_up` loses 2-5% to hardcoded `TILED = 0` over 8 scattered experts. |
| routed | A direct | Vulkan | 1.174 | 2.060 | 57% | 0.886 | Q4_0 57% | Q4_0 57% | 1 | - | `Q4_0` (`57%`: `30.3 / 17.1 us = 56%` gate+up, `21.2 / 12.3 us = 58%` down): `routed_output` (`kernels/kernels/routed.seismic:496-508`) unconditionally runs **2 dispatches at `M = 1`** (`routed_output_project` -> `36.9 KB` scratch + `VkMemoryBarrier2` + `routed_output_combine`), paying an uncredited second `6.21 us` launch (`70%` of `routed_output`'s `8.9 us` gap, `vulkan/src/direct.rs:156-191`), while `routed_gate_up` (`routed.seismic:1436-1444`) hardcodes 32-lane subgroups over 8 scattered experts with F16 shared staging, scalar F32 `projection_dot8` (`projection.glsl:607`), and ungraphed `DirectBatch` submit (`vulkan/src/direct.rs:849-895`). |
| routed | B k-quant | CPU | 7.092 | 14.700 | 48% | 7.608 | Q4_K 48% | Q4_K 48% | 1 | - | `Q4_K` (`48%`, `7.608 ms/step` gap: `209.5 / 118.1 us = 56%` expand, `158.0 / 59.2 us = 37%` down) uses `Steps::Plain` (`together = 8`, `tile/x86.rs:105-126,424-431,475-490`) across unpinned 4-die WX NUMA memory `(inferred)`, but `routed_output` drops to `37%` because single-thread `routed_output_stage` executes `4,608` scalar `f64` divisions (`quant.rs:241-266`) before 256 work items loop over 9 scattered `F = 512` (2-superblock) expert rows (`routed_output.rs:19-48,108-154`). |
| routed | B k-quant | CUDA | 0.646 | 1.584 | 41% | 0.938 | Q2_K 33% | Q4_K 50% | 3 | - | With `DECODE_WARPS = 4` + `RESIDENCY = 1` locking **1 warp per warp scheduler (4/48 warps/SM = 8.3%)** (`routed.cuh:14-19`, `projection.cuh:1079`), `Hier`/`CarriedHier` scale bit-unpacking and `gemv_sums` stall directly on instruction latency (`Q2_K 33%`, `Q3_K 39%`, `Q4_K 50%`), compounded by `routed_output`'s `36 KiB` global `SCRATCH_PUBLISHED` fence + 9-way combine (`output.cuh:85-129`). |
| routed | B k-quant | Metal | 2.338 | 2.514 | 93% | 0.176 | Q3_K 80% | Q4_K 102% | 5 | - | `Q4_K`/`Q5_K`/`Q6_K` average `97-102%` (`92-97%` gate+up, `103-111%` warm-SLC down), while `Q2_K` (`81%`) and `Q3_K` (`80%`) pull the class to `93%` via `GROUP = 16` dual scale/min + `Planes<2, 0/1>` bit-shifts + `Q3_K` `plane_bits<12>` byte loads (`metal/packets.h:206-244,356-371,1161-1178`), softened on `routed_output` by `channels.h:75-111`'s barrier-free `STAGE = 0` `F = 512` loop. |
| routed | B k-quant | Vulkan | 1.027 | 2.096 | 49% | 1.069 | Q2_K 42% | Q4_K 56% | 3 | - | `routed_output` (`kernels/kernels/routed.seismic:496-508`) unconditionally runs **2 dispatches at `M = 1`** (`routed_output_project` -> `36.9 KB` scratch + `VkMemoryBarrier2` + `routed_output_combine`), paying an uncredited second `6.21 us` launch (`0.248 ms/step` over 40 layers, `vulkan/src/direct.rs:156-191`), while `routed_gate_up` (`routed.seismic:1436-1444`) hardcodes 32-lane subgroups (2 packets/lane) with scalar F32 `projection_dot8` (`Q2_K 42%`, `Q3_K 49%`, `Q4_K 56%`). |
| routed | C table | CPU | 7.092 | 14.572 | 49% | 7.480 | IQ4_NL 49% | IQ4_NL 49% | 1 | - | `IQ4_NL` (`49%`, `7.480 ms/step` gap: `197.0 / 118.1 us = 60%` expand, `167.3 / 59.2 us = 35%` down) runs `vpshufb` + `Steps::Sign` (`together = 1`, `tile/x86.rs:139-142,415-422,475-490`) across unpinned 4-die WX NUMA memory `(inferred)`, with `routed_output` (`35%`) bottlenecked by single-thread `routed_output_stage` (`4,608` scalar `f64` `vdivsd` quantize ops, `quant.rs:241-266`) and 9 scattered `F = 512` expert reads (`routed_output.rs:19-48,108-154`). |
| routed | C table | CUDA | 0.773 | 1.668 | 46% | 0.895 | IQ4_XS 44% | IQ4_NL 49% | 2 | - | Locked to 1 warp per scheduler (4/48 warps/SM, `DECODE_WARPS = 4`, `RESIDENCY = 1`, `routed.cuh:14-19`, `projection.cuh:1079`), `IQ4_NL` (`49%`) and `IQ4_XS` (`44%`, +`Hier` 6-bit scale shifts) stall on 6 `prmt.b32` + 4 `OpBF16::bytes` (~26 ALU ops/step, `cuda/packets.cuh:107-125,597-612,875-886`), plus `routed_output`'s `36 KiB` global `SCRATCH_PUBLISHED` fence + combine (`output.cuh:85-129`). |
| routed | C table | Metal | 2.265 | 2.386 | 95% | 0.121 | IQ4_XS 93% | IQ4_NL 97% | 2 | - | `IQ4_NL` (`97%`: `95%` gate+up, `101%` down) and `IQ4_XS` (`93%`: `91%` gate+up, `95%` down, `plane_bits<6>` local scales) stay near the line (`95%` mean): `routed_output_resident` runs barrier-free in registers (`STAGE = 0`, `channels.h:75-111`), leaving only 32 scalar `constant` `table_codes` lookups/packet (`metal/packets.h:522-540`) and `TILED = 0` `@rows32` striding. |
| routed | D float | CPU | 6.699 | 13.060 | 51% | 6.361 | MXFP4 51% | MXFP4 51% | 1 | - | `MXFP4` (`51%`, `6.361 ms/step` gap: `177.7 / 111.6 us = 63%` expand, `148.8 / 55.9 us = 38%` down) pays scalar `E8m0::read8_each` (`native-cpu/src/weights.rs:2218-2228,2327-2338`) + `vpshufb` + `Steps::Sign` (`together = 1`) over unpinned 4-die WX NUMA memory `(inferred)`, while `routed_output` (`38%`) is dragged down by single-thread `4,608`-group `f64` quantization (`quant.rs:241-266`) + 9-expert `F = 512` scatter (`routed_output.rs:19-48,108-154`). |
| routed | D float | CUDA | 0.755 | 1.344 | 56% | 0.589 | MXFP4 56% | MXFP4 56% | 1 | - | `MXFP4` (`56%`: `19.6 / 11.7 us = 59%` gate+up, `14.0 / 7.2 us = 52%` down) decodes E2M1 + E8M0 cheaply (`92%` on `readout`), so the gap is 4/48-warp (`8.3%`, 1 warp/scheduler) `RESIDENCY = 1` occupancy across 1.56-1.76 waves (`routed.cuh:14-19`, `projection.cuh:1079`) plus `routed_output`'s `36 KiB` `SCRATCH_PUBLISHED` `__threadfence() + atomicAdd` combine (`output.cuh:85-129`). |
| routed | D float | Metal | 2.203 | 2.380 | 93% | 0.177 | MXFP4 93% | MXFP4 93% | 1 | - | `MXFP4` (`93%`: `91%` gate+up, `95%` down) pays 32 scalar `constant` `table_codes` lookups + ternary `E8M0Scale::value` per packet (`metal/packets.h:522-536,706-714`) with `TILED = 0` `@rows32` striding on `routed_gate_up`, partially offset by barrier-free `STAGE = 0` `routed_output_resident` (`channels.h:75-111`). |
| routed | D float | Vulkan | 1.137 | 2.228 | 51% | 1.091 | MXFP4 51% | MXFP4 51% | 1 | - | `MXFP4` (`51%`: `29.8 / 16.4 us = 55%` gate+up, `25.9 / 12.0 us = 46%` down) pays `routed_output`'s uncredited second `6.21 us` launch + `36.9 KB` scratch round-trip (`routed.seismic:496-508`, `vulkan/src/direct.rs:156-191`), `routed_gate_up`'s fixed 32-lane subgroup (`routed.seismic:1436-1444`), and `tables.glsl` + `packets_e8m0` + scalar F32 `projection_dot8` (`vulkan/packets.glsl:381-383,557`). |
| routed | E grid | CPU | 3.258 | 66.644 | 5% | 63.386 | IQ2_XXS 5% | IQ2_XXS 5% | 1 | - | `IQ2_XXS` (`5%`, `63.386 ms/step` gap — **#1 overall**: `1026.0 / 54.2 us` expand, `640.1 / 27.2 us` down; chose `INT8 = 0`, `ROWS = 1`) has `Grid::PANEL = None` and `Codes::x86 = None` (`native-cpu/src/weights.rs:349,398,1199,1257-1260,2154`), forcing scalar F32 `Grid::values` (`4` scalar `IQ2XXS_GRID` u64 lookups + byte shifts + `SIGN_MASKS` XORs per 32 weights) and scalar `Carried::read` bit-gathering (`native-cpu/src/weights.rs:2159-2184,3438-3465`). |
| routed | E grid | CUDA | 0.477 | 1.920 | 25% | 1.443 | IQ1_S 15% | IQ2_XXS 35% | 7 | - | `25%` mean across 7 types: `IQ2_XXS` (`35%`) runs `gemv_records_group` (`RESIDENCY = 0`, 1 KiB shared LUT); `IQ2_XS/S`, `IQ3_*`, `IQ1_M` (`18-32%`) run unpipelined `gemv_entries_group` at **1 warp/scheduler (4/48 warps/SM, `RESIDENCY = 1`)** with `230-251r` on `routed_gate_up` and global `__device__ const` LUT reads (`projection.cuh:758-825`, `seismic/lang/build.rs:170-180`); `IQ1_S` (`15%`, `238-254r`) falls back to fragment-order `gemv_mma_group` (`K::BIAS = true`, `cuda/packets.cuh:1429`). |
| routed | E grid | Metal | 1.293 | 2.142 | 60% | 0.849 | IQ1_M 46% | IQ3_S 77% | 7 | - | Compute-bound on M4 Pro's scalar F32 pipe (`60%`: `IQ1_M 46%` to `IQ3_S 77%`), running 8-9 points faster than `attention`/`recurrent` because `routed_output_resident` (`46-78%`) needs no `SharedNorm` barrier and keeps `F = 512` in registers (`STAGE = 0` at `LANES = 16`, `channels.h:75-111`), leaving `Points` `constant` LUT + `sign_bytes[16]`/`DELTA` decode ALU (`metal/packets.h:590-662`). |
| routed | E grid | Vulkan | 0.952 | 2.406 | 40% | 1.454 | IQ2_S 37% | IQ3_XXS 41% | 3 | - | `40%` across 3 device-state-timed types (`IQ2_S 37%`, `IQ3_S 40%`, `IQ3_XXS 41%`; `IQ2_XXS`/`IQ2_XS`/`IQ1_S`/`IQ1_M` gate+up stayed `4r`): `routed_output` pays **2 × `6.21 us` launches** (`12.42 us` > the entire `9.7-10.9 us` 1-launch floor!, `routed.seismic:496-508`) + `36.9 KB` scratch, while both entries run heavy `SEISMIC_CODES_GRID` `const` LUT + sign-mask + scalar F32 FMAs (`vulkan/packets.glsl:446-483,947-965`). |
| routed | F odd | CPU | 1.784 | 11.880 | 15% | 10.096 | Q1_0 15% | Q1_0 15% | 1 | - | `Q1_0` (`15%`, `10.096 ms/step` gap — #7 overall: `132.6 / 29.7 us = 22%` expand, `164.4 / 14.9 us = 9%` down) combines 2990WX's `0.975 TOPS` `int8` ceiling + ~7-op `Odd<1>`/`Steps::Sign` AVX2 unpack (`native-cpu/src/weights.rs:1780-1789,5093-5103`, `tile/x86.rs:462-473`) with `routed_output`'s `9%` cliff: single-thread `routed_output_stage` runs `4,608` scalar `f64` divisions (`quant.rs:241-266`) before 9-expert `F = 512` scatter (`routed_output.rs:19-48,139-151`). |
| routed | F odd | CUDA | 0.391 | 1.023 | 38% | 0.632 | Q1_0 29% | Q2_0 44% | 3 | - | `Q1_0`, `Q2_0`, `TQ2_0` all take `14.8-15.5 us` (gate+up) and `9.4-11.0 us` (down) (`Q1_0 29%`, `TQ2_0 41%`, `Q2_0 44%`): invariant to weight bytes under 4/48-warp (`8.3%`, 1 warp/scheduler) `RESIDENCY = 1` occupancy (`routed.cuh:14-19`, `projection.cuh:1079`), swap-AB `mma.m16n8k16.f32` (`7/8` unused `N`-cols), `LowPlane`/`Odd` SIMT unpack, and `routed_output`'s `36 KiB` global `SCRATCH_PUBLISHED` fence + combine (`output.cuh:85-129`). |
| routed | F odd | Metal | 1.008 | 1.507 | 67% | 0.499 | Q1_0 42% | Q2_0 83% | 3 | - | Bound by M4 Pro's 4.02 G-inst/s F32 pipe (`67%`), with barrier-free `STAGE = 0` `routed_output_resident` (`channels.h:75-111`) lifting `Q2_0` to `83%` (`82%` gate+up, `86%` down) and `TQ2_0` to `78%`, while `Q1_0` drops to `42%` (`25.0` / `15.1 us` — slower than `Q2_0`'s `23.4` / `13.3 us`) on `OddCodes<1>` + unfolded `Packet::values` (+1 FMA/weight, `metal/packets.h:508-515,1175-1177`). |
| routed | F odd | Vulkan | 0.751 | 2.154 | 35% | 1.404 | Q1_0 29% | Q2_0 42% | 2 | - | `35%` (`Q1_0 29%`: `34.0` / `23.6 us`; `Q2_0 42%`: `26.0` / `24.1 us`): `routed_output`'s **2 launches** (`2 × 6.21 = 12.42 us`, `routed.seismic:496-508`) exceed `Q1_0`'s `7.7 us` and `Q2_0`'s `9.3 us` 1-launch floors before reading a byte, compounded by ungraphed `DirectBatch` submit (`vulkan/src/direct.rs:849-895`) and `packets_split_low1`/`low2` + scalar F32 FMAs (`vulkan/packets.glsl:282-298,982-995`). |
| readout | A direct | CPU | 6.004 | 11.376 | 53% | 5.372 | Q8_0 53% | Q8_0 53% | 1 | - | `Q8_0` (`53%`, `5.372 ms/step` gap: `11.376 / 6.004 ms`, `47.5 GB/s`) exceeds 64 MB L3 (`540.1 MB`) and hits both non-VNNI `Steps::Sign` (`together = 1`: `vpaddb + vpabsb + vpsignb + vpmaddubsw + vpmaddwd + vpaddd`, `tile/x86.rs:68-75,415-422,475-490`) and unpinned 30-worker `fetch_add` Infinity Fabric DRAM reads from main-thread `alloc_zeroed` pages across 2990WX's 2 memory + 2 WX compute dies vs `Sweeps`' NUMA-local slices (`cpu/src/workers.rs:552-563`, `cpu/src/buffer.rs:114-123`, `streaming_costs.rs:1864-1889`) `(inferred)`. |
| readout | A direct | CUDA | 0.346 | 0.373 | 93% | 0.027 | Q4_1 91% | Q4_0 95% | 2 | - | Near the 877.5 GB/s cold-L2 line (`93%`: `Q4_0 95%` at `344.1 / 327.5 us = 831 GB/s`, `Q4_1 91%` at `401.4 / 363.7 us`): `3,880` blocks (`47.3` waves on 82 SMs, `KSPLIT = 2`) amortize `readout_head_rows`'s `GRID = 0` 1-block/SM cap (`kernels/kernels/cuda/readout_head_rows.cu:35-43`, `projection.cuh:1079`); `Q4_1` loses 4 points to `gemv_sums<1, 32>` nonzero-min reductions (`projection.cuh:405-467`). |
| readout | A direct | Metal | 1.405 | 1.399 | 100% | -0.006 | Q4_1 99% | Q5_1 102% | 5 | - | At the 2 KiB-row 267.3 GB/s stream line (`100%`, `-0.006 ms`: `Q4_1 99%` to `Q5_1 102%`, `286-540 MB` cold of 36 MB SLC): `byte_floats` (`metal/packets.h:246-321`) and `LANES = 8, SIMDGROUPS = 8, ROWS = 2` stream 16 rows/threadgroup at full DRAM bandwidth. |
| readout | B k-quant | CPU | 3.179 | 5.859 | 54% | 2.680 | Q4_K 54% | Q4_K 54% | 1 | - | `Q4_K` (`54%`, `2.680 ms/step` gap: `5.859 / 3.179 ms`, `48.8 GB/s` over `285.9 MB`) runs `x86v3`'s optimal `Steps::Plain` (`together = 8`, `tile/x86.rs:105-126,424-431,475-490`) and matches `Q8_0`'s `47.5 GB/s`, proving the ~48-50 GB/s cold-DRAM ceiling of unpinned 30-worker `fetch_add` reads across 2990WX's 2 memory + 2 memoryless WX dies vs `Sweeps::written`'s per-worker NUMA first-touch (`90.0 GB/s`, `cpu/src/workers.rs:552-563`, `streaming_costs.rs:1864-1889`) `(inferred)`. |
| readout | B k-quant | CUDA | 0.257 | 0.435 | 59% | 0.179 | Q2_K 42% | Q4_K 80% | 3 | - | Under `RESIDENCY = 1` (8/48 warps/SM = 2 warps/scheduler, `readout_head_rows.cu:35-43`, `projection.cuh:1079`), `Q4_K` reaches `80%` (`411.5 / 327.5 us`), whereas `Q3_K` (`57%`, `440.3 us`) and `Q2_K` (`42%`, `453.6 us` vs `191.7 us` floor) hit a ~440-454 us SIMT/MMA ceiling from `Hier`/`CarriedHier` scale bit-unpacking (`cuda/packets.cuh:1167-1260`) + `gemv_sums<1, 16>` 16-value sub-block reductions (`projection.cuh:405-467`). |
| readout | B k-quant | Metal | 1.078 | 1.129 | 95% | 0.052 | Q3_K 85% | Q4_K 100% | 5 | - | `Q4_K`/`Q5_K`/`Q6_K` sit at `100%` of the 267.3 GB/s line (`1072-1568 us`), while `Q2_K` (`86%`, `731.4 / 625.8 us`) and `Q3_K` (`85%`, `958.6 / 819.0 us`) pull the mean to `95%`: `readout_head_rows` hardcodes `TILED = 0` (`kernels/kernels/metal/readout_head_rows.metal:39`), reloading `Hier` headers per strided `@rows32` packet atop `GROUP = 16` dual scale/min and `Q3_K` `plane_bits<12>` byte loads (`metal/packets.h:356-371,1161-1178`). |
| readout | C table | CPU | 3.179 | 6.379 | 50% | 3.200 | IQ4_NL 50% | IQ4_NL 50% | 1 | - | `IQ4_NL` (`50%`, `3.200 ms/step` gap: `6.379 / 3.179 ms`, `44.8 GB/s` over `285.9 MB`) combines the ~49 GB/s unpinned 4-die WX NUMA DRAM ceiling (`cpu/src/workers.rs:552-563`, `cpu/src/buffer.rs:114-123`) `(inferred)` with `520 us` of extra `vpshufb` + `Steps::Sign` (`together = 1`, widening `i16 -> i32` every 4 weights vs `Q4_K`'s every 32, `tile.rs:305-321`, `tile/x86.rs:139-142,415-422,475-490`). |
| readout | C table | CUDA | 0.318 | 0.408 | 78% | 0.089 | IQ4_XS 72% | IQ4_NL 84% | 2 | - | Across `3,880` blocks under `RESIDENCY = 1` (2 warps/scheduler, `projection.cuh:1079`), `IQ4_NL` (`84%`, `388.1 / 327.5 us`) pays `+44.0 us` over `Q4_0` (`344.1 us`) for `Table::codes` (6 `prmt.b32` + 4 `OpBF16::bytes` = ~26 ALU ops/step, `cuda/packets.cuh:107-125,597-612,875-886`), and `IQ4_XS` (`72%`, `427.0 / 309.4 us`) pays another `+38.9 us` for `Hier` 6-bit scale `u16` loads + shifts (`cuda/packets.cuh:1248-1290`). |
| readout | C table | Metal | 1.042 | 1.052 | 99% | 0.010 | IQ4_XS 98% | IQ4_NL 100% | 2 | - | At the 267.3 GB/s stream line (`99%`: `IQ4_NL 100%` at `1075.9 / 1071.7 us`, `IQ4_XS 98%` at `1029.0 / 1012.2 us`): across `31,040` threadgroups, M4 Pro's `constant` cache + F32 pipe (`8.04 TFLOPS`) hide the 32 `table_codes` lookups/packet (`metal/packets.h:522-540`) behind cold DRAM transfers. |
| readout | D float | CPU | 3.002 | 6.142 | 49% | 3.140 | MXFP4 49% | MXFP4 49% | 1 | - | `MXFP4` (`49%`, `3.140 ms/step` gap: `6.142 / 3.002 ms`, `43.9 GB/s` over `270.0 MB`) hits the unpinned 4-die WX NUMA DRAM ceiling (`cpu/src/workers.rs:552-563`) `(inferred)` plus scalar `E8m0::read8_each` (`8` scalar IEEE-754 bit-hacks per 8-group tile, `native-cpu/src/weights.rs:2218-2228,2327-2338`) and `vpshufb` + `Steps::Sign` (`together = 1`, `tile/x86.rs:139-142,415-422,475-490`). |
| readout | D float | CUDA | 0.309 | 0.335 | 92% | 0.025 | MXFP4 92% | MXFP4 92% | 1 | - | Near the 877.5 GB/s cold-L2 line (`92%`: `334.8 / 309.4 us = 812 GB/s`): tableless `CodebookOf<ID, true>` E2M1 bit-shifts + `E8M0Scale::decode` (`cuda/packets.cuh:628-645,679-686`) beat `Q4_0` by `9.3 us` (`334.8` vs `344.1 us`), leaving only the 8% overhead of 1-block/SM `RESIDENCY = 1` (`8/48` warps/SM, `readout_head_rows.cu:35-43`, `projection.cuh:1079`) and `KSPLIT = 2` reduction. |
| readout | D float | Metal | 1.012 | 1.027 | 99% | 0.015 | MXFP4 99% | MXFP4 99% | 1 | - | At the 267.3 GB/s stream line (`99%`: `1027.4 / 1012.2 us`): `31,040` threadgroups hide `MXFP4`'s 32 `constant` `table_codes` lookups and `E8M0Scale::value` ternaries (`metal/packets.h:522-536,706-714`) behind cold `270 MB` DRAM reads. |
| readout | E grid | CPU | 1.457 | 29.792 | 5% | 28.335 | IQ2_XXS 5% | IQ2_XXS 5% | 1 | - | `IQ2_XXS` (`5%`, `28.335 ms/step` gap — #3 overall: `29.792 / 1.457 ms`, `4.4 GB/s`) has `Grid::PANEL = None` and `Codes::x86 = None` (`native-cpu/src/weights.rs:349,398,1199,1257-1260,2154`), and `INT8 = 1` routes `248,320` rows at `M = 1` through `tile::convert` + `Hier::tile_side` (`512` scalar `Carried::read` bit-gathers + `IQ2XXS_GRID` unpacks + 64-vector AVX2 `8x8` transpose into a one-shot stack `Panel` per `8x256` block, `native-cpu/src/weights.rs:3438-3465,4111-4141`, `tile.rs:446-465,625-669`). |
| readout | E grid | CUDA | 0.175 | 0.576 | 30% | 0.401 | IQ1_S 16% | IQ2_XXS 52% | 7 | - | Clean 3-tier split over `508.56M` weights (`30%` mean): `IQ2_XXS` (`52%`, `290.8 us`) uses coalesced `gemv_records_group` (`1 KiB` shared LUT, double-buffered prefetch, `RESIDENCY = 0`, `projection.cuh:840-934,1080`); `IQ2_XS/S`, `IQ3_*`, `IQ1_M` (`20-39%`, `564-636 us`) use unpipelined `gemv_entries_group` (`RESIDENCY = 1`, random `ld.global.nc` into 2-16 KiB `__device__ const` global grids, `projection.cuh:758-825`, `seismic/lang/build.rs:170-180`); `IQ1_S` (`16%`, `738.2 us`) is disqualified by `K::BIAS = true` (`cuda/packets.cuh:1429`) into fragment-order `gemv_mma_group` (`16` global 16 KiB LUT reads/k-block/lane, `128r`). |
| readout | E grid | Metal | 0.571 | 0.904 | 63% | 0.333 | IQ1_M 47% | IQ3_S 82% | 7 | - | Pure scalar F32 instruction-issue ceiling (`8.04 TFLOPS = 4.02 G-inst/s` -> `126.5 us` per op/weight across `508.56M` weights; `63%` mean): times step strictly with per-weight decode ops (`metal/packets.h:590-662`) — `IQ1_S` `754.2 us` (~5.98 ops/wt, `49%`), `IQ1_M` `887.6 us` (+8 `DELTA` FMAs + `plane_bits<6>`, `47%`), `IQ2_XXS/XS/S` `910-917 us` (`sign_bytes[16]`, `54-67%`), `IQ3_XXS/S` `954-994 us` (dual `Points<..., 4>` `constant` LUT reads, `76-82%`). |
| readout | F odd | CPU | 0.795 | 2.582 | 31% | 1.788 | Q1_0 31% | Q1_0 31% | 1 | - | `Q1_0` (`31%`, `1.788 ms/step` gap: `2.582 / 0.795 ms` raw, `27.7 GB/s = 0.394 TOPS` over `508.56M` weights) is strictly compute-bound: `1.125 bpw` at `90.0 GB/s` requires `1.280 TOPS` > 2990WX's `0.975 TOPS` `int8` peak (`76%` roof at zero unpack cost), and `Odd<1>` (`vpshufb + vpand + vpcmpeqb`) + `Steps::Sign` (`vpsignb + vpmaddubsw + vpmaddwd + vpaddd`) spend ~7 AVX2 ops per 64 OPs (`native-cpu/src/weights.rs:1780-1789,5093-5103`, `tile/x86.rs:462-473`). |
| readout | F odd | CUDA | 0.133 | 0.248 | 54% | 0.115 | Q1_0 33% | Q2_0 66% | 3 | - | `Q1_0` (`83.0 us` floor), `TQ2_0` (`150.9 us`), and `Q2_0` (`164.5 us`) all finish in **`244.7-249.1 us`** (`33%`, `62%`, `66%`): `1,986,560` swap-AB `mma.m16n8k16.f32` instructions (`7/8` empty `N`-cols at `M = 1`, `129.6 us` pure `62.80 TFLOPS` Tensor Core floor, `projection.cuh:402-404`) + ~24 `LowPlane`/`Odd`/`bf16x2` SIMT ops/step (`cuda/packets.cuh:286-320,801-845`) stalling on 2 warps/scheduler (`GRID = 0` -> `RESIDENCY = 1`, `readout_head_rows.cu:35-43`, `projection.cuh:1079`) set a flat ~248 us compute ceiling `(inferred)`. |
| readout | F odd | Metal | 0.433 | 0.656 | 66% | 0.224 | Q1_0 38% | Q2_0 85% | 3 | - | Pure F32 instruction-rate ceiling (`126.5 us` per op/weight at `4.02 G-inst/s`; `66%` mean): `Q2_0` (`634.8 us`, `85%`) and `TQ2_0` (`635.0 us`, `77%`) tie at ~5.03 ops/weight via folded `Offset<1>` `fma(s, code, -s)` (`metal/packets.h:501,1175-1177`), whereas **`Q1_0` is 64.8 us slower (`699.6 us`, `38%`)** despite half the bytes because `OddCodes<1>` runs `fma(code, 2, -1)` plus unfolded `fma(s, code, 0)` in `Packet::values` (`metal/packets.h:508-515,1176-1177`, +1.0 FMA/weight = `+63-65 us`). |

## Ranked by gap (ms per decode step)

| rank | gap ms/step | fraction | backend | role | class | worst type |
|---|---|---|---|---|---|---|
| 1 | 63.386 | 5% | CPU | routed | E grid | IQ2_XXS 5% |
| 2 | 52.448 | 5% | CPU | recurrent | E grid | IQ2_XXS 5% |
| 3 | 28.335 | 5% | CPU | readout | E grid | IQ2_XXS 5% |
| 4 | 19.989 | 37% | CPU | recurrent | A direct | Q8_0 37% |
| 5 | 18.628 | 4% | CPU | attention | E grid | IQ2_XXS 4% |
| 6 | 13.080 | 51% | CPU | routed | A direct | Q8_0 51% |
| 7 | 10.096 | 15% | CPU | routed | F odd | Q1_0 15% |
| 8 | 7.608 | 48% | CPU | routed | B k-quant | Q4_K 48% |
| 9 | 7.480 | 49% | CPU | routed | C table | IQ4_NL 49% |
| 10 | 6.361 | 51% | CPU | routed | D float | MXFP4 51% |
| 11 | 5.672 | 22% | CPU | recurrent | F odd | Q1_0 22% |
| 12 | 5.372 | 53% | CPU | readout | A direct | Q8_0 53% |
| 13 | 5.302 | 54% | CPU | recurrent | B k-quant | Q4_K 54% |
| 14 | 5.201 | 53% | CPU | recurrent | D float | MXFP4 53% |
| 15 | 4.639 | 58% | CPU | recurrent | C table | IQ4_NL 58% |
| 16 | 4.060 | 44% | CPU | attention | A direct | Q8_0 44% |
| 17 | 3.200 | 50% | CPU | readout | C table | IQ4_NL 50% |
| 18 | 3.140 | 49% | CPU | readout | D float | MXFP4 49% |
| 19 | 2.680 | 54% | CPU | readout | B k-quant | Q4_K 54% |
| 20 | 2.410 | 24% | Vulkan | dense | F odd | Q1_0 24% |
| 21 | 2.282 | 23% | CUDA | dense | E grid | IQ1_S 15% |
| 22 | 2.135 | 55% | Metal | dense | E grid | IQ1_M 42% |
| 23 | 1.957 | 18% | CPU | attention | F odd | Q1_0 18% |
| 24 | 1.788 | 31% | CPU | readout | F odd | Q1_0 31% |
| 25 | 1.658 | 37% | Vulkan | recurrent | D float | MXFP4 37% |
| 26 | 1.605 | 21% | CUDA | recurrent | E grid | IQ1_S 13% |
| 27 | 1.489 | 58% | Metal | dense | F odd | Q1_0 35% |
| 28 | 1.473 | 46% | CUDA | dense | B k-quant | Q2_K 36% |
| 29 | 1.454 | 40% | Vulkan | routed | E grid | IQ2_S 37% |
| 30 | 1.443 | 25% | CUDA | routed | E grid | IQ1_S 15% |
| 31 | 1.428 | 53% | CPU | attention | D float | MXFP4 53% |
| 32 | 1.404 | 35% | Vulkan | routed | F odd | Q1_0 29% |
| 33 | 1.276 | 44% | Vulkan | recurrent | C table | IQ4_XS 38% |
| 34 | 1.199 | 59% | CPU | attention | C table | IQ4_NL 59% |
| 35 | 1.198 | 38% | Vulkan | recurrent | E grid | IQ3_XXS 35% |
| 36 | 1.197 | 59% | CPU | attention | B k-quant | Q4_K 59% |
| 37 | 1.178 | 51% | Metal | recurrent | E grid | IQ1_M 39% |
| 38 | 1.169 | 57% | CUDA | dense | C table | IQ4_NL 57% |
| 39 | 1.154 | 28% | CUDA | dense | F odd | Q1_0 28% |
| 40 | 1.095 | 35% | CUDA | recurrent | B k-quant | Q2_K 27% |
| 41 | 1.091 | 51% | Vulkan | routed | D float | MXFP4 51% |
| 42 | 1.069 | 49% | Vulkan | routed | B k-quant | Q2_K 42% |
| 43 | 1.016 | 68% | CUDA | dense | A direct | Q4_0 65% |
| 44 | 1.016 | 38% | Vulkan | recurrent | F odd | Q1_0 33% |
| 45 | 0.990 | 42% | CUDA | recurrent | C table | IQ4_XS 41% |
| 46 | 0.938 | 41% | CUDA | routed | B k-quant | Q2_K 33% |
| 47 | 0.927 | 45% | Vulkan | recurrent | B k-quant | Q2_K 45% |
| 48 | 0.895 | 46% | CUDA | routed | C table | IQ4_XS 44% |
| 49 | 0.886 | 57% | Vulkan | routed | A direct | Q4_0 57% |
| 50 | 0.858 | 52% | Metal | recurrent | F odd | Q1_0 35% |
| 51 | 0.849 | 60% | Metal | routed | E grid | IQ1_M 46% |
| 52 | 0.802 | 65% | CUDA | dense | D float | MXFP4 65% |
| 53 | 0.789 | 31% | CUDA | recurrent | F odd | Q1_0 22% |
| 54 | 0.702 | 52% | CUDA | recurrent | A direct | Q4_1 50% |
| 55 | 0.676 | 55% | CUDA | routed | A direct | Q4_1 53% |
| 56 | 0.636 | 52% | CUDA | recurrent | D float | MXFP4 52% |
| 57 | 0.632 | 38% | CUDA | routed | F odd | Q1_0 29% |
| 58 | 0.589 | 56% | CUDA | routed | D float | MXFP4 56% |
| 59 | 0.558 | 90% | Metal | dense | B k-quant | Q2_K 73% |
| 60 | 0.516 | 90% | Metal | dense | C table | IQ4_XS 89% |
| 61 | 0.499 | 67% | Metal | routed | F odd | Q1_0 42% |
| 62 | 0.478 | 91% | Metal | dense | D float | MXFP4 91% |
| 63 | 0.401 | 30% | CUDA | readout | E grid | IQ1_S 16% |
| 64 | 0.389 | 85% | Metal | recurrent | B k-quant | Q2_K 65% |
| 65 | 0.382 | 85% | Metal | recurrent | D float | MXFP4 85% |
| 66 | 0.382 | 24% | CUDA | attention | E grid | IQ1_S 15% |
| 67 | 0.373 | 85% | Metal | recurrent | C table | IQ4_XS 82% |
| 68 | 0.354 | 43% | Vulkan | attention | B k-quant | Q4_K 37% |
| 69 | 0.339 | 39% | Vulkan | attention | E grid | IQ2_XXS 26% |
| 70 | 0.333 | 63% | Metal | readout | E grid | IQ1_M 47% |
| 71 | 0.309 | 52% | Metal | attention | E grid | IQ1_M 40% |
| 72 | 0.264 | 53% | Vulkan | attention | C table | IQ4_NL 46% |
| 73 | 0.261 | 39% | CUDA | attention | B k-quant | Q2_K 31% |
| 74 | 0.250 | 44% | Vulkan | attention | F odd | Q1_0 39% |
| 75 | 0.227 | 47% | CUDA | attention | C table | IQ4_XS 46% |
| 76 | 0.224 | 66% | Metal | readout | F odd | Q1_0 38% |
| 77 | 0.223 | 54% | Metal | attention | F odd | Q1_0 35% |
| 78 | 0.186 | 35% | CUDA | attention | F odd | Q1_0 26% |
| 79 | 0.179 | 59% | CUDA | readout | B k-quant | Q2_K 42% |
| 80 | 0.178 | 55% | CUDA | attention | A direct | Q4_1 52% |
| 81 | 0.177 | 93% | Metal | routed | D float | MXFP4 93% |
| 82 | 0.176 | 93% | Metal | routed | B k-quant | Q3_K 80% |
| 83 | 0.167 | 63% | Vulkan | attention | D float | MXFP4 63% |
| 84 | 0.159 | 55% | CUDA | attention | D float | MXFP4 55% |
| 85 | 0.121 | 95% | Metal | routed | C table | IQ4_XS 93% |
| 86 | 0.115 | 54% | CUDA | readout | F odd | Q1_0 33% |
| 87 | 0.097 | 98% | Metal | dense | A direct | Q5_0 97% |
| 88 | 0.090 | 87% | Metal | attention | B k-quant | Q2_K 67% |
| 89 | 0.089 | 78% | CUDA | readout | C table | IQ4_XS 72% |
| 90 | 0.060 | 91% | Metal | attention | C table | IQ4_XS 88% |
| 91 | 0.057 | 91% | Metal | attention | D float | MXFP4 91% |
| 92 | 0.052 | 95% | Metal | readout | B k-quant | Q3_K 85% |
| 93 | 0.027 | 93% | CUDA | readout | A direct | Q4_1 91% |
| 94 | 0.025 | 92% | CUDA | readout | D float | MXFP4 92% |
| 95 | 0.019 | 99% | Metal | recurrent | A direct | Q5_0 97% |
| 96 | 0.015 | 99% | Metal | readout | D float | MXFP4 99% |
| 97 | 0.014 | 100% | Metal | routed | A direct | Q5_1 98% |
| 98 | 0.010 | 99% | Metal | readout | C table | IQ4_XS 98% |
| 99 | -0.006 | 100% | Metal | readout | A direct | Q4_1 99% |
| 100 | -0.020 | 103% | Metal | attention | A direct | Q5_1 98% |

## Cells with no type timed in device state on every entry of the role (bounds, not ranked)

| backend | role | class | types timed only in 4-repetition sessions on some entry |
|---|---|---|---|
| Vulkan | attention | A direct | Q4_0 |
| Vulkan | dense | A direct | Q8_0, Q4_0 |
| Vulkan | dense | B k-quant | Q2_K, Q4_K |
| Vulkan | dense | C table | IQ4_NL |
| Vulkan | dense | D float | MXFP4 |
| Vulkan | dense | E grid | IQ2_XXS, IQ1_S |
| Vulkan | readout | A direct | Q4_0, Q4_1 |
| Vulkan | readout | B k-quant | Q2_K, Q3_K, Q4_K |
| Vulkan | readout | C table | IQ4_NL, IQ4_XS |
| Vulkan | readout | D float | MXFP4 |
| Vulkan | readout | E grid | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M |
| Vulkan | readout | F odd | Q1_0, Q2_0, TQ2_0 |
| Vulkan | recurrent | A direct | Q4_0 |
| Vulkan | routed | C table | IQ4_NL, IQ4_XS |

## Per-type cells (µs at 1 row / floor = fraction; `raw`: read from the raw samples)

**Metal**

| class | type | recurrent input projection | recurrent output projection | attention q/k/v projection | attention output projection | routed experts gate+up | routed experts and shared down | readout full head | dense gate+up | dense down |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Q8_0 | 103.8 / 101.5 = 98% | 31.7 / 34.8 = 110% | 75.5 / 76.5 = 101% | 32.4 / 34.8 = 108% | 69.6 / 68.2 = 98% | 38.6 / 39.0 = 101% | 2014.3 / 2023.0 = 100% | 192.7 / 189.1 = 98% | 95.5 / 95.3 = 100% |
| A | Q4_0 | 59.1 / 54.4 = 92% | 16.2 / 19.1 = 118% | 41.1 / 41.2 = 100% | 16.1 / 19.1 = 119% | 37.7 / 36.8 = 98% | 19.3 / 21.3 = 111% | 1067.3 / 1071.7 = 100% | 103.4 / 100.8 = 97% | 50.4 / 51.1 = 101% |
| A | Q4_1 | 63.7 / 60.3 = 95% | 16.7 / 21.1 = 126% | 45.7 / 45.6 = 100% | 16.2 / 21.1 = 130% | 42.3 / 40.7 = 96% | 22.3 / 23.6 = 106% | 1199.7 / 1190.6 = 99% | 115.2 / 111.8 = 97% | 55.4 / 56.7 = 102% |
| A | Q5_0 | 70.5 / 66.2 = 94% | 21.7 / 23.1 = 106% | 51.7 / 50.0 = 97% | 21.2 / 23.1 = 109% | 46.0 / 44.6 = 97% | 24.6 / 25.8 = 105% | 1318.1 / 1309.5 = 99% | 128.4 / 122.9 = 96% | 62.5 / 62.2 = 99% |
| A | Q5_1 | 76.3 / 72.1 = 94% | 21.3 / 25.0 = 117% | 56.3 / 54.4 = 97% | 24.6 / 25.0 = 102% | 51.1 / 48.6 = 95% | 26.8 / 28.0 = 104% | 1396.0 / 1428.4 = 102% | 135.1 / 133.9 = 99% | 67.9 / 67.7 = 100% |
| B | Q2_K | 51.2 / 32.4 = 63% | 17.2 / 11.8 = 69% | 37.4 / 24.7 = 66% | 17.3 / 11.8 = 68% | 27.6 / 22.1 = 80% | 15.5 / 13.1 = 84% | 731.4 / 625.8 = 86% | 81.9 / 59.4 = 73% | 40.8 / 30.5 = 75% |
| B | Q3_K | 62.9 / 41.9 = 67% | 21.0 / 15.0 = 71% | 48.1 / 31.8 = 66% | 21.4 / 15.0 = 70% | 35.5 / 28.5 = 80% | 20.7 / 16.7 = 80% | 958.6 / 819.0 = 85% | 102.3 / 77.3 = 76% | 53.5 / 39.4 = 74% |
| B | Q4_K | 60.4 / 54.4 = 90% | 17.8 / 19.1 = 108% | 44.4 / 41.2 = 93% | 17.5 / 19.1 = 109% | 37.8 / 36.8 = 97% | 19.2 / 21.3 = 111% | 1072.1 / 1071.7 = 100% | 102.9 / 100.8 = 98% | 53.7 / 51.1 = 95% |
| B | Q5_K | 72.9 / 66.2 = 91% | 21.4 / 23.1 = 108% | 52.8 / 50.0 = 95% | 22.0 / 23.1 = 105% | 45.9 / 44.6 = 97% | 25.0 / 25.8 = 103% | 1316.0 / 1309.5 = 100% | 125.4 / 122.9 = 98% | 64.5 / 62.2 = 96% |
| B | Q6_K | 83.1 / 78.7 = 95% | 26.8 / 27.2 = 102% | 60.7 / 59.4 = 98% | 26.7 / 27.2 = 102% | 57.6 / 53.0 = 92% | 29.4 / 30.5 = 104% | 1568.0 / 1562.2 = 100% | 152.0 / 146.3 = 96% | 74.0 / 73.9 = 100% |
| C | IQ4_NL | 64.3 / 54.4 = 85% | 18.9 / 19.1 = 101% | 45.4 / 41.2 = 91% | 19.2 / 19.1 = 100% | 38.7 / 36.8 = 95% | 21.1 / 21.3 = 101% | 1075.9 / 1071.7 = 100% | 111.4 / 100.8 = 90% | 54.2 / 51.1 = 94% |
| C | IQ4_XS | 64.7 / 51.5 = 80% | 20.2 / 18.2 = 90% | 44.8 / 39.0 = 87% | 20.1 / 18.2 = 90% | 38.1 / 34.8 = 91% | 21.4 / 20.2 = 95% | 1029.0 / 1012.2 = 98% | 109.3 / 95.3 = 87% | 52.9 / 48.4 = 91% |
| D | MXFP4 | 63.4 / 51.5 = 81% | 19.0 / 18.2 = 96% | 43.8 / 39.0 = 89% | 19.1 / 18.2 = 95% | 38.2 / 34.8 = 91% | 21.3 / 20.2 = 95% | 1027.4 / 1012.2 = 99% | 107.6 / 95.3 = 89% | 51.0 / 48.4 = 95% |
| E | IQ2_XXS | 56.8 / 25.8 = 45% | 20.3 / 9.6 = 47% | 43.2 / 19.7 = 46% | 20.4 / 9.6 = 47% | 34.4 / 17.7 = 51% | 19.6 / 10.6 = 54% | 910.1 / 492.0 = 54% | 98.3 / 47.0 = 48% | 50.2 / 24.2 = 48% |
| E | IQ2_XS | 60.9 / 28.7 = 47% | 20.5 / 10.6 = 52% | 44.9 / 21.9 = 49% | 20.3 / 10.6 = 52% | 34.1 / 19.6 = 58% | 20.1 / 11.7 = 58% | 917.2 / 551.5 = 60% | 99.6 / 52.5 = 53% | 49.8 / 27.0 = 54% |
| E | IQ2_S | 61.2 / 31.6 = 52% | 20.6 / 11.5 = 56% | 45.3 / 24.1 = 53% | 21.0 / 11.5 = 55% | 34.3 / 21.6 = 63% | 20.0 / 12.8 = 64% | 910.7 / 610.9 = 67% | 101.0 / 58.0 = 57% | 52.3 / 29.8 = 57% |
| E | IQ3_XXS | 62.5 / 37.5 = 60% | 21.3 / 13.5 = 63% | 46.1 / 28.5 = 62% | 21.3 / 13.5 = 63% | 36.1 / 25.5 = 71% | 20.7 / 15.0 = 72% | 954.2 / 729.8 = 76% | 102.9 / 69.1 = 67% | 54.6 / 35.3 = 65% |
| E | IQ3_S | 66.2 / 41.9 = 63% | 21.9 / 15.0 = 68% | 47.4 / 31.8 = 67% | 22.2 / 15.0 = 67% | 37.2 / 28.5 = 77% | 21.3 / 16.7 = 78% | 993.5 / 819.0 = 82% | 106.9 / 77.3 = 72% | 55.8 / 39.4 = 71% |
| E | IQ1_S | 50.8 / 19.9 = 39% | 17.8 / 7.6 = 43% | 37.1 / 15.3 = 41% | 17.9 / 7.6 = 43% | 28.4 / 13.7 = 48% | 16.0 / 8.4 = 52% | 754.2 / 373.1 = 49% | 83.4 / 36.0 = 43% | 42.3 / 18.7 = 44% |
| E | IQ1_M | 57.6 / 22.1 = 38% | 20.1 / 8.4 = 42% | 43.2 / 16.9 = 39% | 20.1 / 8.4 = 42% | 32.5 / 15.2 = 47% | 20.1 / 9.2 = 46% | 887.6 / 417.7 = 47% | 96.1 / 40.1 = 42% | 49.1 / 20.8 = 42% |
| F | Q1_0 | 42.8 / 14.7 = 34% | 16.5 / 5.9 = 36% | 33.1 / 11.4 = 35% | 16.5 / 5.9 = 36% | 25.0 / 10.3 = 41% | 15.1 / 6.5 = 43% | 699.6 / 269.0 = 38% | 77.1 / 26.3 = 34% | 37.5 / 13.9 = 37% |
| F | Q2_0 | 45.6 / 28.0 = 61% | 15.6 / 10.3 = 66% | 32.9 / 21.3 = 65% | 15.6 / 10.3 = 66% | 23.4 / 19.1 = 82% | 13.3 / 11.4 = 86% | 634.8 / 536.6 = 85% | 71.6 / 51.1 = 71% | 36.5 / 26.3 = 72% |
| F | TQ2_0 | 43.9 / 25.8 = 59% | 15.7 / 9.6 = 61% | 31.5 / 19.7 = 63% | 15.6 / 9.6 = 61% | 22.8 / 17.7 = 78% | 13.4 / 10.6 = 79% | 635.0 / 492.0 = 77% | 71.3 / 47.0 = 66% | 34.5 / 24.2 = 70% |

**CUDA**

| class | type | recurrent input projection | recurrent output projection | routed experts gate+up | routed experts and shared down | attention q/k/v projection | attention output projection | readout full head | dense gate+up | dense down |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Q8_0 | - | - | - | - | - | - | - | 89.6 / 58.6 = 65% | 36.2 / 30.1 = 83% |
| A | Q4_0 | 31.9 / 17.6 = 55% | 12.1 / 6.9 = 57% | 20.5 / 12.3 = 60% | 14.0 / 7.5 = 54% | 23.1 / 13.6 = 59% | 12.5 / 6.9 = 55% | 344.1 / 327.5 = 95% | 50.5 / 31.7 = 63% | 24.3 / 16.6 = 68% |
| A | Q4_1 | 39.7 / 19.4 = 49% | 14.5 / 7.5 = 52% | 24.9 / 13.4 = 54% | 15.9 / 8.2 = 52% | 28.3 / 14.9 = 53% | 14.6 / 7.5 = 51% | 401.4 / 363.7 = 91% | - | - |
| B | Q2_K | 43.6 / 10.9 = 25% | 13.8 / 4.6 = 34% | 23.7 / 7.8 = 33% | 15.3 / 5.0 = 33% | 29.2 / 8.6 = 29% | 13.8 / 4.6 = 34% | 453.6 / 191.7 = 42% | 56.1 / 19.1 = 34% | 26.2 / 10.3 = 39% |
| B | Q3_K | 41.9 / 13.8 = 33% | 14.8 / 5.6 = 38% | 23.8 / 9.7 = 41% | 16.4 / 6.1 = 37% | 29.8 / 10.7 = 36% | 14.1 / 5.6 = 40% | 440.3 / 250.5 = 57% | - | - |
| B | Q4_K | 41.1 / 17.6 = 43% | 13.8 / 6.9 = 50% | 22.7 / 12.3 = 54% | 16.9 / 7.5 = 45% | 27.6 / 13.6 = 49% | 13.7 / 6.9 = 50% | 411.5 / 327.5 = 80% | 60.7 / 31.7 = 52% | 26.9 / 16.6 = 62% |
| C | IQ4_NL | 41.5 / 17.6 = 42% | 15.1 / 6.9 = 46% | 23.3 / 12.3 = 53% | 17.3 / 7.5 = 44% | 27.8 / 13.6 = 49% | 14.7 / 6.9 = 47% | 388.1 / 327.5 = 84% | 58.4 / 31.7 = 54% | 26.5 / 16.6 = 63% |
| C | IQ4_XS | 42.5 / 16.7 = 39% | 14.7 / 6.6 = 45% | 25.1 / 11.7 = 46% | 17.7 / 7.2 = 41% | 28.1 / 12.9 = 46% | 14.7 / 6.6 = 45% | 427.0 / 309.4 = 72% | - | - |
| D | MXFP4 | 32.0 / 16.7 = 52% | 12.5 / 6.6 = 53% | 19.6 / 11.7 = 59% | 14.0 / 7.2 = 52% | 23.2 / 12.9 = 56% | 12.2 / 6.6 = 54% | 334.8 / 309.4 = 92% | 49.7 / 30.1 = 60% | 21.2 / 15.8 = 74% |
| E | IQ2_XXS | 29.2 / 8.9 = 30% | 10.6 / 4.0 = 37% | 18.1 / 6.4 = 36% | 12.4 / 4.3 = 34% | 19.9 / 7.0 = 35% | 10.5 / 4.0 = 38% | 290.8 / 150.9 = 52% | 38.9 / 15.4 = 39% | 17.2 / 8.4 = 49% |
| E | IQ2_XS | 49.9 / 9.8 = 20% | 17.7 / 4.3 = 24% | 29.5 / 7.0 = 24% | 19.8 / 4.6 = 23% | 34.9 / 7.7 = 22% | 17.7 / 4.3 = 24% | 564.3 / 169.0 = 30% | - | - |
| E | IQ2_S | 53.8 / 10.7 = 20% | 20.7 / 4.6 = 22% | 29.9 / 7.6 = 25% | 18.4 / 4.9 = 27% | 35.0 / 8.4 = 24% | 20.7 / 4.6 = 22% | 581.6 / 187.1 = 32% | - | - |
| E | IQ3_XXS | 55.3 / 12.5 = 23% | 15.1 / 5.2 = 34% | 32.5 / 8.8 = 27% | 16.9 / 5.6 = 33% | 33.9 / 9.7 = 29% | 15.3 / 5.2 = 34% | 586.8 / 223.4 = 38% | - | - |
| E | IQ3_S | 56.6 / 13.8 = 24% | 17.9 / 5.6 = 31% | 30.2 / 9.7 = 32% | 18.7 / 6.1 = 33% | 34.2 / 10.7 = 31% | 18.1 / 5.6 = 31% | 635.9 / 250.5 = 39% | - | - |
| E | IQ1_S | 61.2 / 7.1 = 12% | 18.2 / 3.4 = 19% | 35.7 / 5.2 = 15% | 21.4 / 3.6 = 17% | 42.5 / 5.7 = 13% | 18.2 / 3.4 = 19% | 738.2 / 114.7 = 16% | 89.4 / 12.0 = 13% | 39.7 / 6.8 = 17% |
| E | IQ1_M | 52.2 / 7.8 = 15% | 17.2 / 3.6 = 21% | 31.5 / 5.7 = 18% | 21.0 / 3.9 = 18% | 35.2 / 6.2 = 18% | 17.2 / 3.6 = 21% | 633.9 / 128.3 = 20% | - | - |
| F | Q1_0 | 29.0 / 5.5 = 19% | 8.7 / 2.8 = 33% | 15.4 / 4.2 = 27% | 9.4 / 3.0 = 32% | 19.4 / 4.5 = 23% | 8.6 / 2.8 = 33% | 249.1 / 83.0 = 33% | 35.3 / 9.1 = 26% | 15.1 / 5.3 = 35% |
| F | Q2_0 | 29.7 / 9.6 = 32% | 9.6 / 4.2 = 44% | 15.5 / 6.9 = 44% | 10.6 / 4.5 = 43% | 20.1 / 7.5 = 38% | 9.8 / 4.2 = 43% | 248.8 / 164.5 = 66% | - | - |
| F | TQ2_0 | 28.1 / 8.9 = 32% | 8.8 / 4.0 = 45% | 14.8 / 6.4 = 43% | 11.0 / 4.3 = 39% | 19.4 / 7.0 = 36% | 8.7 / 4.0 = 46% | 244.7 / 150.9 = 62% | - | - |

**Vulkan**

| class | type | recurrent input projection | recurrent output projection | routed experts gate+up | routed experts and shared down | attention q/k/v projection | attention output projection | readout full head | dense gate+up | dense down |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Q8_0 | - | - | - | - | - | - | - | 74.0 / 63.8 = 86% 4r | 38.3 / 35.0 = 91% |
| A | Q4_0 | 38.5 / 22.5 = 58% | 15.4 / 11.6 = 76% 4r | 30.3 / 17.1 = 56% | 21.2 / 12.3 = 58% | 29.3 / 18.4 = 63% 4r | 15.3 / 11.6 = 76% raw | 2091.6 / 334.9 = 16% 4r | 272.7 / 36.7 = 13% 4r | 24.4 / 21.5 = 88% raw |
| A | Q4_1 | - | - | 24.0 / 18.3 = 76% 4r | 21.3 / 13.0 = 61% | - | - | 399.4 / 371.4 = 93% 4r | - | - |
| B | Q2_K | 39.2 / 15.7 = 40% raw | 16.8 / 9.4 = 56% raw | 28.0 / 12.5 = 45% raw | 25.4 / 9.8 = 38% raw | 37.0 / 13.3 = 36% | 16.5 / 9.4 = 57% | 380.8 / 197.9 = 52% 4r | 267.3 / 24.0 = 9% 4r | 45.9 / 15.1 = 33% raw |
| B | Q3_K | 178.6 / 18.6 = 10% 4r | 17.3 / 10.4 = 60% | 29.2 / 14.5 = 50% | 22.1 / 10.9 = 49% | 30.7 / 15.5 = 51% | 18.3 / 10.4 = 57% | 396.9 / 257.3 = 65% 4r | - | - |
| B | Q4_K | 139.9 / 22.5 = 16% 4r | 43.8 / 11.6 = 27% raw | 25.1 / 17.1 = 68% | 27.4 / 12.3 = 45% raw | 29.6 / 18.4 = 62% | 52.6 / 11.6 = 22% | 2154.8 / 334.9 = 16% 4r | 162.3 / 36.7 = 23% 4r | 33.1 / 21.5 = 65% raw |
| C | IQ4_NL | 39.1 / 22.5 = 57% | 25.2 / 11.6 = 46% raw | 145.0 / 17.1 = 12% 4r | 101.2 / 12.3 = 12% 4r | 29.5 / 18.4 = 62% | 35.2 / 11.6 = 33% | 1440.3 / 334.9 = 23% 4r | 125.9 / 36.7 = 29% 4r | 30.1 / 21.5 = 71% raw |
| C | IQ4_XS | 59.3 / 21.6 = 36% raw | 28.4 / 11.3 = 40% raw | 55.3 / 16.4 = 30% 4r | 26.8 / 12.0 = 45% 4r | 30.1 / 17.7 = 59% | 17.1 / 11.3 = 66% | 410.7 / 316.6 = 77% 4r | - | - |
| D | MXFP4 | 41.8 / 21.6 = 52% raw | 46.4 / 11.3 = 24% raw | 29.8 / 16.4 = 55% | 25.9 / 12.0 = 46% | 30.0 / 17.7 = 59% | 15.8 / 11.3 = 72% | 2319.8 / 316.6 = 14% 4r | 166.1 / 35.0 = 21% 4r | 79.4 / 20.6 = 26% 4r |
| E | IQ2_XXS | 37.4 / 13.7 = 37% | 19.4 / 8.7 = 45% | 153.0 / 11.2 = 7% 4r | 26.4 / 9.0 = 34% | 33.1 / 11.8 = 36% | 46.5 / 8.7 = 19% | 2493.9 / 156.8 = 6% 4r | 137.2 / 20.2 = 15% 4r | 28.4 / 13.2 = 46% |
| E | IQ2_XS | 42.3 / 14.6 = 34% raw | 20.4 / 9.0 = 44% raw | 29.1 / 11.8 = 40% 4r | 21.2 / 9.3 = 44% | 30.9 / 12.5 = 40% | 17.6 / 9.0 = 51% | 433.3 / 175.1 = 40% 4r | - | - |
| E | IQ2_S | 42.2 / 15.5 = 37% raw | 17.6 / 9.3 = 53% | 36.8 / 12.4 = 34% | 22.3 / 9.7 = 43% | 30.2 / 13.2 = 44% | 17.7 / 9.3 = 53% | 414.5 / 193.4 = 47% 4r | - | - |
| E | IQ3_XXS | 60.3 / 17.3 = 29% raw | 17.9 / 9.9 = 55% raw | 31.2 / 13.6 = 44% raw | 27.0 / 10.4 = 38% raw | 32.9 / 14.5 = 44% raw | 21.1 / 9.9 = 47% raw | 417.5 / 229.9 = 55% 4r | - | - |
| E | IQ3_S | 208.2 / 18.6 = 9% 4r | 17.6 / 10.4 = 59% raw | 35.8 / 14.5 = 40% raw | 27.4 / 10.9 = 40% raw | 44.7 / 15.5 = 35% | 22.3 / 10.4 = 46% | 599.1 / 257.3 = 43% 4r | - | - |
| E | IQ1_S | 121.4 / 11.9 = 10% 4r | 16.9 / 8.1 = 48% raw | 29.3 / 10.0 = 34% 4r | 24.2 / 8.3 = 34% | 29.2 / 10.4 = 36% | 15.8 / 8.1 = 51% | 418.8 / 120.3 = 29% 4r | 290.2 / 16.8 = 6% 4r | 28.3 / 11.5 = 41% raw |
| E | IQ1_M | 63.2 / 12.5 = 20% 4r | 18.2 / 8.3 = 46% raw | 30.8 / 10.4 = 34% 4r | 24.3 / 8.6 = 35% | 30.7 / 11.0 = 36% | 17.0 / 8.3 = 49% | 480.4 / 134.0 = 28% 4r | - | - |
| F | Q1_0 | 34.6 / 10.3 = 30% | 19.7 / 7.6 = 38% | 34.0 / 8.9 = 26% | 23.6 / 7.7 = 33% | 28.1 / 9.3 = 33% | 15.6 / 7.6 = 48% | 1788.7 / 88.4 = 5% 4r | 69.7 / 13.8 = 20% raw | 29.5 / 10.0 = 34% raw |
| F | Q2_0 | 36.6 / 14.3 = 39% raw | 19.6 / 8.9 = 46% raw | 26.0 / 11.6 = 45% raw | 24.1 / 9.3 = 38% | 28.6 / 12.3 = 43% | 16.4 / 8.9 = 54% | 334.2 / 170.5 = 51% 4r | - | - |
| F | TQ2_0 | 36.4 / 13.7 = 38% raw | 18.2 / 8.7 = 48% raw | 23.9 / 11.2 = 47% 4r | 21.1 / 9.0 = 43% | 28.6 / 11.8 = 41% | 16.2 / 8.7 = 54% | 325.7 / 156.8 = 48% 4r | - | - |

**CPU**

| class | type | recurrent input projection | recurrent output projection | routed experts and shared gate+up | routed experts and shared down | attention q/k/v projection | attention output projection | readout full head |
|---|---|---|---|---|---|---|---|---|
| A | Q8_0 | 895.1 / 297.3 = 33% | 167.7 / 99.2 = 59% raw | 450.9 / 223.0 = 49% | 210.7 / 111.6 = 53% | 542.1 / 223.0 = 41% | 186.1 / 99.2 = 53% | 11375.7 / 6004.0 = 53% |
| B | Q4_K | 326.2 / 157.5 = 48% | 60.6 / 52.6 = 87% | 209.5 / 118.1 = 56% | 158.0 / 59.2 = 37% | 221.6 / 118.1 = 53% | 68.9 / 52.6 = 76% | 5859.1 / 3178.7 = 54% |
| C | IQ4_NL | 275.7 / 157.5 = 57% raw | 89.0 / 52.6 = 59% | 197.0 / 118.1 = 60% | 167.3 / 59.2 = 35% | 221.5 / 118.1 = 53% | 69.2 / 52.6 = 76% | 6378.9 / 3178.7 = 50% |
| D | MXFP4 | 269.0 / 148.7 = 55% raw | 102.8 / 49.7 = 48% | 177.7 / 111.6 = 63% | 148.8 / 55.9 = 38% | 212.5 / 111.6 = 53% | 91.6 / 49.7 = 54% | 6142.0 / 3002.1 = 49% |
| E | IQ2_XXS | 1321.8 / 72.3 = 5% raw | 522.9 / 24.2 = 5% | 1026.0 / 54.2 = 5% | 640.1 / 27.2 = 4% | 1386.4 / 54.2 = 4% | 554.9 / 24.2 = 4% | 29792.3 / 1457.0 = 5% |
| F | Q1_0 | 186.1 / 39.5 = 21% raw | 55.8 / 13.3 = 24% raw | 132.6 / 29.7 = 22% | 164.4 / 14.9 = 9% | 183.3 / 29.7 = 16% | 55.4 / 13.3 = 24% raw | 2582.4 / 794.8 = 31% raw |
