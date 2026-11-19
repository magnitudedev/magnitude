# Compatibility

**Each backend declares one explicit floor of hardware, drivers, and operating system.**
Anything above the floor is an optional tier that is detected at runtime and never required.
The CPU backend serves hosts with no GPU above any backend's floor.

## Principles

- **Scope:** desktop, laptop, workstation, and datacenter hardware from roughly the last
  7–8 years. Older hardware and phone/mobile GPUs are out of scope.
- **Justified requirements:** every hard requirement must be justified by what the kernels
  need. Any feature a kernel can work without becomes an optional tier, selected per device
  from probed facts.
- **Checked at discovery:** floors are checked when devices are discovered, and a failing
  device reports a typed reason. A device that passes the floor must not later fail for a
  compatibility reason.
- **Backends by floor:** hardware below one GPU backend's floor may be served by another
  backend (for example, Turing through Vulkan). Hardware below every GPU floor is served by the
  CPU. A GPU above a floor is never sent to the CPU: a kernel that computes wrong on it is an
  engine bug that fails loudly and is fixed.
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
  - fp32 round-to-nearest-even, declared as `RoundingModeRTE 32` only once the measurement job's
    `rte_32_division` probe has shown the driver forming and rounding it as declared (the NVIDIA
    580 driver's compiler traps on it); until then, and where it does not hold, fp32 rounds by
    the device's default
  - fp32 signed zero, Inf, and NaN preservation
  - discovery reads device properties only; it opens no device and runs no probe
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
    reductions. `seismic-build` refuses `gl_Subgroup*` built-ins and every subgroup operation
    that spans the hardware subgroup in kernel sources.
  - Keep lane exchanges inside the logical subgroup.
- A kernel computes the same function at either width. Its rounding may differ where a form's
  reduction order does; the precision system judges that, not this document.

### Execution model

A kernel may assume only what Vulkan guarantees, plus two assumptions. Vulkan does not run a
subgroup in lockstep, does not say which invocations of a workgroup share a subgroup or in which
lane order, and does not promise that invocations rejoin after a loop they leave at different
iterations, or after some of them exit. CUDA's warp gives all of these, and NVIDIA's Vulkan
driver happens to; AMD's Windows driver does not (#151).

- **The assumptions:**
  - A selection (if/else, switch) entered with subgroup-uniform control flow rejoins at its
    merge: a subgroup operation after it has every invocation. A function call rejoins at its
    return in the same way, whatever returns its body takes. Where the device has
    `VK_KHR_shader_subgroup_uniform_control_flow`, every module that relies on the assumption
    declares `SubgroupUniformControlFlowKHR`, and the driver guarantees it. A module relies on
    it where it runs a subgroup operation or subgroup barrier after invocations that divergent
    control parted have rejoined: at or after a divergent selection's merge, or where an exit
    taken under divergent control leads, with no workgroup barrier between. Elsewhere it is
    relied on because every driver merges structured selections; loops whose invocations leave
    at different iterations are what compilers break, and they are refused, with one exception.
    Where the device has `VK_KHR_shader_maximal_reconvergence`, a subgroup operation may follow
    such a loop, and so may a subgroup barrier: the module declares `MaximallyReconvergesKHR` and
    the driver rejoins the invocations after the loop. Formation accepts the form only in a
    module formed that way. The device library owns the mode (`SEISMIC_HAS_MAXIMAL_RECONVERGENCE`, which `seismic-build`
    refuses in kernel sources). Its `seismic_strided`, `seismic_strided_from` and
    `seismic_strided_bounded` loops are written once for both: the plain loop where the device
    has the mode, otherwise rounds with a uniform trip count and a guard on the bound. Its
    `seismic_rejoin()` is nothing where the device has the mode, otherwise a workgroup barrier.
    A subgroup operation inside a loop whose invocations leave apart is refused either way.
  - A workgroup barrier at a legal point is executed by every invocation of the workgroup
    together, so control is uniform after it, even after a loop its invocations left at
    different iterations. Every implementation counts a workgroup barrier per hardware subgroup.
    A point is legal where control, and every exit before it, is the same for every invocation
    of the workgroup: neither divergent nor steered by which subgroup an invocation is in. A
    subgroup barrier carries no such assumption.
- **Uniform and divergent:** a value is uniform when every invocation of the hardware subgroup
  holds it: constants, loads through uniform pointers from push constants, the argument block,
  the tables buffer and the buffers the launch only reads, the workgroup, grid and subgroup
  counts and ids, whole-subgroup reductions, votes and ballots, a broadcast or shuffle from a
  uniform lane, and what is computed from uniform values alone. Lane ids, invocation ids,
  relative shuffles, scans, clustered reductions and atomics' results are divergent, and so is
  anything computed from them. An undefined value (a variable before it is written) adds
  nothing to what it joins: a kernel whose result depends on one is wrong whatever it holds.
  A whole-subgroup result shows which subgroup combined it (it derives from the lane order,
  below) where its operands derive from the subgroup's id or differ between invocations
  otherwise than by lane. Where they differ by lane alone, such as a lane id or a load at a
  lane's address of memory no store races, every full subgroup combines the same values, and
  the result is the same in every subgroup.
- **Memory:**
  - A load through a uniform pointer from workgroup memory is uniform only where no store that
    may touch its bytes shares its barrier interval: the stretch reachable from the same
    workgroup barrier (one whose semantics order workgroup memory), or from the entry, without
    crossing another. Within an interval a store races a load where one reaches the other in
    it, in either order, or where a branch whose condition may differ between invocations parts
    the ways to them, as different invocations take different branches at once. It races no
    load where a branch the whole workgroup takes alike (its condition, the control it runs
    under and the exits before it uniform), which the interval passes once at most, parts every
    way to the store from every way to the load: every invocation took the same way there, so
    none is at both. Otherwise the invocations of a subgroup see the store or not, and only
    lockstep would make them agree. Workgroup memory is followed by byte: an access at
    a constant offset into its variable (every index the same constant in every invocation)
    touches those bytes alone, all group-memory views being one memory; an access at any other
    offset may touch any byte, so a store at a variable offset races every load.
  - A load from a buffer the launch writes is never uniform: a barrier orders only its own
    workgroup, and other workgroups may store to it at any time. A kernel steered by such a word
    stages it through workgroup memory. What another workgroup stores derives from none of the
    reader's ids or lane order: through a uniform pointer, a load carries what a store derives
    from only where one of its own workgroup shares its barrier interval or may run anywhere
    before it, since no barrier here orders device memory.
  - A buffer the launch only reads is one nothing in the launch writes. Access belongs to the
    launch, and its module is the one source of what it writes: a buffer whose address the
    module never stores through is one it only reads, such as a result or scratch an earlier
    launch of the dispatch wrote. A `&` parameter is one by declaration: the ABI reads its
    address, and the argument words, through the input block (`seismic_input_block_t`), and a
    write through it, or through the tables buffer, is refused. So is the address table of a
    slab or record port the launch writes through: the ABI reads its address through the
    address-table block (`seismic_address_table_block_t`), and a write through it is refused.
  - Which buffer an access reaches is followed through the invocation's registers and
    variables: a value computed from buffers' addresses, as the argument block gives them,
    names every one of them, whatever the computation, so an address rebuilt from another
    buffer's address and a delta names both. No such value is stored to workgroup or device
    memory, so a value read from memory holds no address the launch was given: it is an
    offset where it is added to a named address, and otherwise may point anywhere, except
    where it is read from memory the launch only reads: one read from a `&` buffer points into
    the memory a `&` buffer's tables reach, and one read from an address table into that
    port's own bytes, which the runtime keeps apart from every other binding of the launch
    (so a store through it writes only them). That holds for backed slots: a free slot names a
    zero slab its store shares, which kernels only read by a masked load whose value they
    discard and never store through (formation does not check this), and the host repoints no
    slot a launch in flight reads as backed. A condition (a boolean) holds no address: one
    on an address carries the condition's divergence but not its address. One assumption
    remains: a delta rebuilt from conditions, bit by bit or branch by branch, is taken as an
    offset. No kernel builds one.
  - No two tensors bound to one launch share a byte when either is written. The language makes
    every such pair disjoint, and the runtime refuses a binding that breaks it on every platform,
    in its one binding check, comparing the bytes each binding reaches (a slab view's, the slots
    its site binds).
- **Steering by votes and joins:** hardware-wide votes (the prelude's `seismic_uniform_all` and
  `seismic_uniform_any`) are uniform and may steer control flow; the logical-subgroup helpers on
  64-lane hardware are clustered, so their results are divergent. A value each logical subgroup
  holds alike steers the hardware subgroup through the prelude's `seismic_steer(live, p)`, the
  whole condition of an `if`, `while` or `for`: a vote over its logical subgroups on 64-lane
  hardware, binding `live` to the caller's own predicate for the region it guards. On 32-lane
  hardware the vote is the predicate itself and `live` the literal true: formation requires the
  predicate uniform, then removes the vote from the module it ships, so neither the vote nor
  `live` is left there, and a steered loop exits at its header (seismic-build refuses any other
  placement). `seismic_hardware_first` broadcasts such a value from the first logical subgroup,
  and is the value itself on 32-lane hardware.
- **Lane order:** because the lane order is unspecified, a value derived from the local or
  global invocation id and a value derived from the lane order (a lane or subgroup id, a
  per-lane subgroup result, or a whole-subgroup result that shows which subgroup combined it)
  never meet in one value or one store, nor through the control a store to shared memory runs
  under, nor where invocations that took different paths meet. Shared memory is judged per
  class (workgroup memory; memory the launch writes), not per array, but for the bytes a load
  at a constant offset of workgroup memory reads (Memory, above).
- **Constructs:** a selection is divergent when its condition is. A loop is lane-varying when it
  is left or continued under a divergent condition. A return, `break` or `continue` under a
  divergent condition is a lane-dependent exit. A value a loop carries out derives from the
  lane order when the loop is left under a condition that differs between subgroups. A
  branch whose condition is computed from constants (through values, struct members, calls'
  parameters and variables) takes one way, and the way no invocation takes is not checked; a
  specialization constant is no constant here, since pipeline creation sets it after the
  check, and an undefined condition takes either way.
- **Variables:** a function's variable (a local left in memory, a call's parameter) is each
  invocation's own. A store to it keeps the value stored, whatever control it runs under; the
  control enters where invocations that took different paths meet (a selection's merge, a
  lane-varying loop's exits, the rejoin after a lane-dependent exit), as it does for a value
  chosen there, and what a lane-varying loop stores differs where the loop is left.
- **Enforced at formation:** after cleanup, a copy of each module with its functions inlined
  (except the 32-lane joins) and its locals in SSA form is checked, and formation refuses:
  - a subgroup operation inside a divergent selection or a lane-varying loop, or after a
    lane-varying loop with no workgroup barrier between, or after a lane-dependent exit (until
    the merge the exit rejoins at);
  - a control barrier inside a divergent selection or a lane-varying loop, after a
    lane-dependent exit, or, for a subgroup barrier, after a lane-varying loop with no workgroup
    barrier between (but in a module declaring `MaximallyReconvergesKHR`);
  - a workgroup barrier under control, or after an exit, steered by which subgroup an
    invocation is in;
  - a subgroup operation on a value derived from the local or global invocation id: computed
    from one, chosen by control that depends on one, read from memory such a value is stored to,
    or through an address derived from one;
  - a value or store where local-id and lane-order derivations meet;
  - a subgroup operation whose operand SPIR-V requires to be uniform is divergent: a broadcast's
    or read invocation's lane, a quad broadcast's index, a cooperative-matrix load's or store's
    pointer, layout and stride;
  - a 32-lane join whose argument is divergent;
  - a store of a buffer's address, or of a value computed from one, to workgroup or device
    memory;
  - a function named as a 32-lane join (`seismic_hardware_`) that does not return its one
    argument;
  - a write to a buffer the launch only reads, a module with more than one entry point, and an
    instruction the check does not read.
- **Precision of the check:** a struct's members are followed apart, in values and in variables,
  inside arrays too (each member joined across the array's elements, so one member written
  at a divergent index leaves the others as they were), and a variable holds at each point
  what the stores reaching it put there. A value a lane-varying loop computes alike in every
  iteration (from values outside it, push constants, the argument block, or memory the launch
  only reads at an address the same in every iteration) is uniform after it. A refusal names,
  per kind, how many instructions break the model so, the first, and where what it refuses
  comes from (the built-in load, racing read, subgroup operation, phi where divergent control
  rejoins or load of a variable written there that raised the offending fact, or the loop
  or selection and the condition that steer it).
- **Full subgroups:** a pipeline whose module reads its subgroups (a subgroup operation or a
  subgroup built-in) requires full subgroups, and a launch of it whose workgroup X extent is not
  a multiple of 64 is refused when it is formed, for every device and offline alike, so no
  subgroup it reads is ever partial at either width. A kernel that reads no subgroup takes any X.

### Driver defects

Drivers that meet the floor and still fail on a valid module. Each defect has a shape the kernels
avoid, and formation refuses that shape where it can be recognised.

- **NVIDIA 580, a 64-bit conditional subtraction written as a branch** (`if (x >= m) x -= m;`):
  the compiler crashed forming `attention_prefill`, `attention_append_k8v4` and `import_dense`.
  Kernels write the select, `x -= x >= m ? m : 0ul`. Formation refuses the branch form (an
  `OpPhi` of `x` and `x - m` at the merge of a compare of the two; `seal.rs`).
- **NVIDIA 580 (GB10 on 580.159.03, Ada on 580.95.05), `seismic_sqrt_rn`'s digit steps as a
  loop.** On Ada the numerics oracle's kernel with this loop hung the GPU (utilization 100%, no
  completion in 38 minutes). On GB10 the restoring square root's 28 steps of 64-bit selects,
  written as a constant-count loop (rolled or left to the driver), computed every odd exponent
  as though the preceding parity shift were absent
  (`sqrt(1)` gave `sqrt(2)`, 40820 of the oracle's 65537 cases), in the numerics oracle's kernel
  (`abi/vulkan.rs`) with or without its early return. The same function inlined into another
  kernel computed correctly, and the same steps unrolled (by `[[unroll]]` or written out)
  computed correctly in the failing kernel. The shipped SPIR-V is correct. A neighbouring shape
  computed correctly there: `seismic_narrow_divide`'s 32 steps of 64-bit selects as a kept
  loop, and a constant-count loop of 32-bit selects. The prelude writes the steps
  straight-line. Formation does not recognise the shape: no class is established beyond this
  loop, so the oracle test on GB10 guards it.

### Optional tiers

- **fp16 arithmetic,** together with fp16 round-to-nearest-even and preservation.
  - Required only alongside cooperative matrix.
  - Without it, kernels compute in fp32 over 16-bit storage.
- **Cooperative matrix** (`VK_KHR_cooperative_matrix`, subgroup-scope 16×16×16).
- **Shader float atomics and 64-bit shared atomics.**
- **`VK_KHR_shader_fma`.** With it, `fma` compiles to `OpFmaKHR` and is fused. Without it,
  `fma` is GLSL `fma`, which the driver may fuse or round twice.
- **`VK_KHR_shader_subgroup_uniform_control_flow`.** With it, modules declare
  `SubgroupUniformControlFlowKHR`, which guarantees the execution model's one assumption.

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
  | x86v3 | AVX2, FMA, F16C, BMI1/2 | Intel Haswell through Comet Lake Core; AMD Zen 1–3 |
  | x86v3vnni | AVX-VNNI (VEX, 256-bit) | Intel 12th gen and newer Core, N100 and other Alder Lake-N parts |
  | x86v4 | AVX-512 F/CD/BW/DQ/VL | Intel Skylake-SP/X server and workstation parts (before Cascade Lake) |
  | x86v4vnni | AVX-512 VNNI | Intel Cascade Lake, Ice Lake, Tiger Lake, Rocket Lake, Sapphire Rapids and newer Xeon; AMD Zen 4 and Zen 5 |
  | neon | NEON | Every AArch64 CPU |
  | neondot | `dotprod` (`sdot`) | Nearly every ARM CPU since 2018: Apple M1 and newer, Snapdragon X, Graviton 2+, Ampere Altra, Raspberry Pi 5 |
  | neoni8mm | `i8mm` (`smmla`) | Apple M2 and newer, Snapdragon X, Graviton 3+, NVIDIA Grace and GB10 |

- **v2 machines are supported but slow.** Hosts should prefer native builds: arm64 on Apple
  silicon and Windows on Arm.
- **Planned tiers** (performance on hardware that is already supported):
  - x86: AMX, on Sapphire Rapids and newer Xeon (x86v4vnni until then)
  - AArch64: SME2 (Apple M4) and FP16 arithmetic
- **Operating systems:**
  - Linux: glibc 2.35 (Ubuntu 22.04-compatible userspace)
  - Windows x64
  - macOS 15

## Pending conformance

- **Unqualified hardware:** the Vulkan paths for 64-wide hardware (AMD GCN), for devices without
  fp16 arithmetic (NVIDIA Pascal/Maxwell), and for NVK have been validated offline but not yet run on
  those devices.
- **Old CUDA drivers:** the R525 floor has not yet been exercised end to end on an R525–R575 driver.
