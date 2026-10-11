# AMD RDNA3 Vulkan host setup

Requirements for measuring the Vulkan backend on an AMD RDNA3 GPU (for example a Radeon PRO V710,
Navi 32, gfx1101) under Linux with Mesa RADV. The same steps apply to other RDNA3 parts.

## Driver: Mesa RADV 26.2 or newer

- The backend needs `VK_KHR_cooperative_matrix`, which older RADV releases lack. Ubuntu 22.04's stock
  RADV (23.2) has no cooperative matrix, and the Mesa PPAs stop at 25.0 for 22.04. On such a
  distribution, build Mesa from source with only the AMD Vulkan driver (`-Dvulkan-drivers=amd`; LLVM is
  not needed) into a separate prefix such as `/opt/mesa-<version>`.
- Select that build with `VK_DRIVER_FILES=<prefix>/share/vulkan/icd.d/radeon_icd.<arch>.json`. Keep
  `VK_DRIVER_FILES` set: otherwise the distribution's ICD also enumerates the same GPU and a second,
  older RADV device appears.

## Vulkan loader: the LunarG Vulkan SDK

- Old system loaders (for example 1.3.204 on Ubuntu 22.04) ignore `VK_DRIVER_FILES`. Install the LunarG
  Vulkan SDK (1.4.x), source its `setup-env.sh`, and make sure its `lib/` comes first on
  `LD_LIBRARY_PATH` so its loader is the one used.
- The SDK also provides `vulkaninfo`, glslang, SPIRV-Tools and the validation layers
  (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` enables them for correctness runs; leave them off for
  timing).

Put the `VK_DRIVER_FILES`, SDK and `LD_LIBRARY_PATH` settings in one environment script and source it
in every shell that builds or runs the backend.

## Kernel module pitfall

- Some cloud images blacklist `amdgpu`. If `/dev/dri/renderD128` is missing, load it with
  `sudo modprobe amdgpu`, and add a boot-time unit or modules-load entry if the image does not load it.
- An out-of-tree (dkms) `amdgpu` needs the kernel's full module set. If automatic updates install a new
  kernel without its matching extra-modules package (on Ubuntu, `linux-modules-extra-<kernel>`), the
  module fails to load with unknown `drm_*` symbols (for example `drm_dp_*` because
  `drm_display_helper` is missing), and the machine boots with no GPU. Fix it with
  `sudo apt install linux-modules-extra-$(uname -r)` followed by a reboot.
- After every reboot or kernel update, check that `/dev/dri/renderD128` exists before measuring.
  Either hold the kernel packages or make sure every new kernel's extra-modules package is installed
  before the reboot.

## Verify with `vulkaninfo`

With the environment script sourced, `vulkaninfo --summary` must list exactly one device for the GPU,
with driver RADV (for example `RADV NAVI32`) and the Mesa version you built, not the distribution's.
For the full facts, run `validation/hardware/vulkan-facts`, which prints one JSON document per device.
Check at least:

- `VK_KHR_cooperative_matrix` with f16 × f16 → f16/f32 and s8/u8 → i32 16×16×16 subgroup-scope shapes;
- subgroup size control, with compute in `requiredSubgroupSizeStages` and sizes 32–64;
- packed int8 dot product acceleration;
- `VK_KHR_shader_fma`;
- `VK_EXT_memory_budget`, and a `DEVICE_LOCAL` heap that is host-visible (ReBAR);
- `maxMemoryAllocationSize` and `maxBufferSize` (4 GiB − 4 on RADV).

## Device facts observed on a Radeon PRO V710 (RADV 26.2.3)

- Vulkan 1.4.354, conformance 1.4.5.3. All of spec §4's floor is present.
- **Subgroups:**
  - sizes 32–64, default 64;
  - compute is in `requiredSubgroupSizeStages`;
  - every subgroup operation plus rotate.
- **Cooperative matrix** (`VK_KHR_cooperative_matrix`), all 16×16×16 at subgroup scope:
  - f16 × f16 → f16 or f32;
  - u8/s8 × u8/s8 → i32/u32, saturating variants included;
  - no bf16 or fp8 (`VK_KHR_shader_bfloat16` and `VK_EXT_shader_float8` are absent).
- **Packed mixed-signedness int8 dot product:** accelerated.
- **Shared memory:** 64 KiB. Push constants: 256 B.
- **Allocation limits:** `maxMemoryAllocationSize` and `maxBufferSize` are both 4 GiB − 4.
- **Memory:**
  - on the 24 GiB part, heap 1 is 25.1 GiB `DEVICE_LOCAL` and fully host-visible (ReBAR);
  - `VK_EXT_memory_budget`, `VK_EXT_memory_priority` and `VK_EXT_external_memory_host` are present;
  - sparse binding and sparse residency buffers are supported;
  - `VK_EXT_pageable_device_local_memory` is absent.
- **Queues:** family 0 is graphics+compute; family 1 is compute-only with 2 queues. Timestamps are 64-bit,
  with a 10 ns period. Calibrated timestamps are available.
- **Float controls:**
  - RTE and SignedZeroInfNanPreserve for f16 and f32;
  - fp32 denorm preserve is supported;
  - independence is `TYPE_32_ONLY`.
- **FMA:**
  - `VK_KHR_shader_fma` is present (f16, f32 and f64).
  - GLSL `fma()` (`GLSL.std.450 Fma`) is **not** reliably fused: next to a `precise` `a*b+c` on the same
    operands, ACO emits `v_mul_f32` + `v_add_f32`.
  - `OpFmaKHR` always gives `v_fma_f32`.
  - glslang cannot emit `OpFmaKHR` from GLSL. See spec §16.1.

## Baselines

For llama.cpp comparisons on the same GPU, use an official llama.cpp Vulkan release binary, and
optionally a HIP build for gfx1101 (a ROCm installation or the ROCm Python wheels, run with the ROCm
libraries on `LD_LIBRARY_PATH`). Run `validation/reference_matrix.sh` on the machine itself, with no
other GPU work running during timing.
