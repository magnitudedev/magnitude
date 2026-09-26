# Compatibility

**Each backend declares one explicit floor of hardware, drivers, and operating system.**
Anything above the floor is an optional tier that is detected at runtime and never required.
The CPU backend is always available as the fallback.

## Principles

- **Scope:** desktop, laptop, workstation, and datacenter hardware from roughly the last
  7–8 years. Older hardware and phone/mobile GPUs are out of scope.
- **Justified requirements:** every hard requirement must be justified by what the kernels
  need. Any feature a kernel can work without becomes an optional tier, selected per device
  from probed facts.
- **Checked at discovery:** floors are checked when devices are discovered, and a failing
  device reports a typed reason. A device that passes the floor must not later fail for a
  compatibility reason.
- **Fallback between backends:** hardware below one GPU backend's floor may be served by
  another backend (for example, Turing through Vulkan). Otherwise it falls back to CPU.
- **Documented exclusions:** the exclusions listed here are deliberate. A new exclusion
  requires updating this document.

## Summary

| Backend | Platforms | Hardware floor | Driver / OS floor |
| --- | --- | --- | --- |
| Metal | macOS arm64 | Any Apple silicon Mac running macOS 15 | macOS 15 |
| CUDA | Linux x64/arm64, Windows x64 | Compute capability 8.0 (Ampere) | NVIDIA R525 (CUDA 12.0) |
| Vulkan | Linux x64/arm64, Windows x64 | Vulkan 1.3 device meeting the Vulkan floor (below) | Current vendor driver reporting Vulkan 1.3 |
| CPU | All hosts | x86-64-v2 or AArch64 | glibc 2.35 (Linux), macOS 15 |

## Metal

- **macOS 15 or newer.** The backend compiles every library with precise math modes that
  macOS 15 introduced. That makes macOS 15 the floor for the whole Apple build.
- **No GPU family floor.** Every Apple silicon Mac that runs macOS 15 qualifies.
- **Probed, not required:** the Metal Shading Language version (macOS 15 provides 3.2 or
  newer), bf16 arithmetic, simdgroup-matrix element types and multiply-accumulate
  combinations, and scalar collective types. Kernels select among them through device facts.
- **Intel Macs:** CPU only. An x86 build running under Rosetta sees only the x86-64-v2 tier,
  so Apple silicon Macs use the native arm64 build.

## CUDA

- **Compute capability 8.0 or newer.**

  | Compute capability | Hardware |
  | --- | --- |
  | 8.0 | A100, A30 |
  | 8.6 | RTX 30 series, RTX A-series, A10, A40 |
  | 8.7 | Jetson Orin |
  | 8.9 | RTX 40 series, L4, L40 |
  | 9.0 | H100, H200 |
  | 10.0 / 10.3 | B200, B300 |
  | 12.0 | RTX 50 series |
  | 12.1 | GB10 (DGX Spark) |

  The floor exists because the kernels rely on bf16 tensor-core MMA and asynchronous
  global-to-shared copies (`cp.async`), both introduced with sm_80.
- **Excluded architectures:** Turing (RTX 20, GTX 16, T4) and Volta (V100) are excluded from
  CUDA and served by Vulkan. Pascal and older are not served by CUDA.
- **Two compilation routes:**
  - Model kernels are CUDA C++ that the bundled **NVRTC 12.9** compiles on the user's machine
    into CUBIN for the device's exact architecture.
  - Compiler-generated kernels are PTX 7.1, compiled by the driver's JIT.

  Customer machines never need a CUDA toolkit. `libcuda` / `nvcuda.dll` comes from the
  driver.
- **Driver floor: R525 (CUDA 12.0).**
  - NVRTC 12.x CUBIN is guaranteed to load on any driver from the 12.x series (CUDA minor
    version compatibility).
  - Discovery rejects older drivers with a typed reason.
- **Optional tiers:**
  - Driver 12.1+: 32 KiB kernel parameter space.
  - Driver 12.9+ on the Blackwell 10.x family: the `sm_100f` target with tcgen05 and tensor
    memory.
- **Not supported while NVRTC 12.9 is bundled:**
  - Jetson Thor (sm_110).
  - Architectures newer than 12.9's target set, such as Rubin.

  Supporting them requires adding an NVRTC 13 tier, which needs an R580+ driver. That tier
  would be selected per device, so older GPUs keep the R525 floor.

## Vulkan

### Floor

- **Vulkan 1.3** loader and device.
- **Core features:**
  - buffer device address, the Vulkan memory model at device scope, timeline semaphores,
    synchronization2, maintenance4, and pipeline creation cache control
  - subgroup size control with full subgroups
  - integer dot product
- **Types and storage:**
  - 8-, 16-, and 64-bit integer shader types
  - 8- and 16-bit storage buffer access
  - scalar block layout
  - extended subgroup types
- **Subgroups:**
  - basic, vote, arithmetic, ballot, shuffle, shuffle-relative, clustered, and quad
    operations in compute
  - a subgroup width that can be 32, or a fixed width of 64 (see below)
- **Extensions:**
  - `VK_KHR_workgroup_memory_explicit_layout`, with 8-bit, 16-bit, and scalar layouts
  - `VK_EXT_memory_budget`
- **Numerics:**
  - fp32 round-to-nearest-even
  - fp32 signed zero, Inf, and NaN preservation
  - a fused `fma`, either through `VK_KHR_shader_fma` or confirmed by a probe
- **Limits:**
  - 32 KiB of shared memory
  - 1024 invocations per workgroup, with workgroup size at least (1024, 1024, 64)
  - 65535 workgroups on every axis
  - 128 bytes of push constants
- **Queues and memory:**
  - a compute queue with timestamps
  - a device-local heap
  - host-visible coherent memory

### Subgroup model

- Kernels are written against a **logical 32-lane subgroup**:
  - Where the device supports width 32, pipelines request it.
  - On hardware with a fixed width of 32, no request is needed. This covers drivers that
    advertise no required-size stages, such as NVK.
  - On hardware with a fixed width of 64 (AMD GCN), each hardware subgroup holds two logical
    subgroups.
- **Kernel authoring rules:**
  - Use the prelude's lane and subgroup identity helpers and its 32-lane-clustered
    reductions. Never use `gl_Subgroup*` builtins or whole-subgroup reductions directly.
  - Keep lane exchanges inside the logical subgroup.
- **Results are identical across subgroup widths:** fixed-order reductions give identical
  results on 32- and 64-wide hardware.

### Optional tiers

- **fp16 arithmetic,** together with fp16 round-to-nearest-even and preservation.
  - Required only alongside cooperative matrix.
  - Without it, kernels compute in fp32 over 16-bit storage.
- **Cooperative matrix** (`VK_KHR_cooperative_matrix`, subgroup-scope 16×16×16).
- **Shader float atomics, 64-bit shared atomics, and `VK_KHR_shader_fma`.**

### Coverage

| Vendor | Supported | Excluded |
| --- | --- | --- |
| NVIDIA | Maxwell and newer on the proprietary driver (Windows, Linux). Turing and newer on NVK. | Kepler and older |
| AMD | GCN4 (Polaris) and newer on Windows and RADV, including Vega-based Ryzen APUs. RDNA1 and newer run natively 32-wide. | Pre-Polaris GCN |
| Intel | Gen9 (Skylake) and newer on Linux (ANV). On Windows: Gen9/9.5, and Xe-LP/Arc on drivers from 101.4499 (mid-2023) onward. | On Windows only: Ice Lake, Gemini Lake, and DG1 (the driver does not expose 64-bit integers). These run on CPU. |
| Qualcomm | — | Adreno X (Snapdragon X): subgroup width 64/128 and no 8-bit workgroup-memory layout |

Apple GPUs are served by Metal. MoltenVK is not a target.

## CPU

- **Architectures:** x86_64 and AArch64, with 64-bit pointers.
- **x86 floor: x86-64-v2** (SSE4.2, POPCNT).
  - Covers every Intel CPU since Nehalem (2008) and every AMD CPU since Bulldozer (2011).
  - Excludes only virtual machines configured with `qemu64` or `kvm64` CPU models.
- **AArch64 floor: NEON**, present on every ARMv8 CPU.
- **Tiers.** The highest supported tier is detected at runtime.

  | Tier | Adds | Typical hardware |
  | --- | --- | --- |
  | x86v2 | SSE3–4.2, POPCNT | Atom-class Gemini Lake / Jasper Lake; 2017–2020 Pentium and Celeron; x86 under Rosetta |
  | x86v3 | AVX2, FMA, F16C, BMI1/2 | Intel Haswell and newer (including all 12th gen and newer Core, and N100); AMD Zen 1–3 |
  | x86v4 | AVX-512 F/CD/BW/DQ/VL | Intel Skylake-SP/X server and workstation parts |
  | x86v4vnni | AVX-512 VNNI | Intel Ice Lake, Tiger Lake, Rocket Lake, Sapphire Rapids and newer Xeon; AMD Zen 4/5 |
  | neon | NEON | Every AArch64 CPU |

- **v2 machines are supported but slow.** Hosts should prefer native builds: arm64 on Apple
  silicon and Windows on Arm.
- **Planned tiers** (performance on hardware that is already supported):
  - x86: AVX-VNNI, on Intel 12th gen and newer Core, N100, and Zen 5
  - AArch64: `dotprod`, on nearly every ARM CPU since 2018 (Apple M1 and newer, Snapdragon X,
    Graviton, Raspberry Pi 5)
  - AArch64: `i8mm` and FP16 arithmetic
- **Operating systems:**
  - Linux: glibc 2.35 (Ubuntu 22.04-compatible userspace)
  - Windows x64
  - macOS 15

## Pending conformance

- **Unqualified hardware:** the Vulkan paths for 64-wide hardware (AMD GCN), for devices without
  fp16 arithmetic (NVIDIA Pascal/Maxwell), and for NVK have been validated offline but not yet run on
  those devices.
- **Old CUDA drivers:** the R525 floor has not yet been exercised end to end on an R525–R575 driver.
- **Release configuration:** macOS 15 and one artifact per host have not replaced the earlier
  inference packaging yet.
