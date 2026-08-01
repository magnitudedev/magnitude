---
applies_to:
  - packages/release/**
  - packages/icn/**
  - packages/icn-protocol/**
  - inference/crates/icn-server/**
  - inference/crates/icn-hardware/**
  - inference/native/**
  - inference/scripts/**
  - .github/workflows/release-build.yml
---

# CUDA compatibility

Magnitude ships self-contained CUDA backend packs across a declared range of NVIDIA drivers and
GPUs. Compatibility is derived from the executable images and runtime closure in a finished pack,
the loaded driver, and the selected physical device. Backend readiness is then proven by executing
the pack. Loading `libcuda`, reporting a CUDA API version, enumerating a GPU, or querying its memory
is discovery, not execution readiness.

The user's machine does not need a CUDA toolkit. Toolkits are release-build inputs. Magnitude owns
and ships each pack's CUDA runtime libraries; the NVIDIA driver stack is a host capability.

## Exact compatibility facts

The following facts are independent and must never be collapsed into one “CUDA version”:

- **Driver API capability** is returned by the loaded `libcuda` through `cuDriverGetVersion`. It is
  not the version of an ambient toolkit installation.
- **Physical compute capability** identifies GPU hardware, such as Ampere 8.0/8.6, Ada 8.9,
  Hopper 9.0, or Blackwell 12.0.
- **PTX ISA version** is the `.version` declared by an embedded PTX module. It identifies the PTX
  language that the active driver JIT must consume.
- **Code-image target** is the PTX virtual target or cubin real target, including whether that
  target is ordinary, family-specific, or architecture-specific.
- **Toolkit and compiler identity** identify the release toolchain that produced an artifact.
  They are provenance and reproducibility facts. Compatibility follows the emitted image, not a
  toolkit name when the emitted image says something more exact.
- **Bundled runtime identity** identifies the CUDA runtime and libraries shipped with the pack and
  their own driver requirements.
- **Loaded provider identity** identifies the actual host library that supplied CUDA, including a
  WSL bridge or a native Linux driver provider.

`nvcc --version`, `CUDA_HOME`, and `/usr/local/cuda` describe a toolkit available for local builds.
They are never runtime compatibility inputs for Magnitude's precompiled packs.

## Artifact contract

Every immutable CUDA backend pack declares:

- host, product tier, artifact digest, native build, and backend-module ABI;
- exact toolkit/compiler closure used to produce it;
- exact bundled runtime-library closure and its driver requirement;
- every device-code image actually embedded in the final backend module; and
- the typed compatibility relation and supported driver-JIT floor for each image.

PTX image state includes:

```text
PtxImage
  PTX ISA major and minor from .version
  producer identity
  target
    Ordinary(virtual compute capability)
    | FamilySpecific(family and virtual compute capability)
    | ArchitectureSpecific(exact architecture)
  minimum supported driver API for this PTX ISA
```

Cubin image state includes:

```text
CubinImage
  target
    Ordinary(real compute capability)
    | FamilySpecific(family and real compute capability)
    | ArchitectureSpecific(exact architecture)
```

Release assembly derives this state from the finished fatbinary with NVIDIA tooling. Requested
CMake architectures are build inputs, not release truth. Publication fails when inspected images
are missing, undeclared, misclassified, or inconsistent with their declared driver requirements.

Toolkit-to-driver tables are used only to establish the conservative supported driver floor for an
emitted PTX ISA or bundled runtime. The artifact records the resulting requirement directly. A
single generic CUDA-major floor cannot represent both runtime minor compatibility and PTX JIT
compatibility.

For example, a CUDA 12.9 artifact containing PTX `.version 8.8` has a supported driver-JIT floor of
CUDA driver API `12090`; the CUDA 12 runtime family floor `12000` is insufficient for that PTX
path. CUDA 12.9's documented corresponding driver floors are Linux `575.51.03` and Windows/WSL
`576.02`.

## Host CUDA provider

CUDA driver discovery loads the canonical platform provider and records the resolved provider path.
On Linux it first resolves `libcuda.so.1` through the system loader. If that fails or resolves an
invalid provider, it searches a bounded, deterministic set of platform roots: WSL bridge and driver
projection roots, the host architecture's standard multiarch and NVIDIA directories, conventional
`/usr/lib{,64}`, NixOS, NVIDIA-container, and CUDA forward-compatibility roots. It accepts only
versioned `libcuda.so.*` files. On Windows it loads `nvcuda.dll` only from the system directory.

The resolver validates required driver symbols, initializes CUDA, enumerates devices and compute
capabilities, and rejects the toolkit stub. It retains the validated provider handle for the process
lifetime and reports its actual path and driver API. Eligibility and CUDA runtime initialization use
this same resolver in every ICN process, so a weaker discovery probe cannot publish CUDA while the
backend uses a different loading rule.

Magnitude never mutates host loader configuration or asks users to create system symlinks. Failure
retains a bounded diagnostic. Toolkit `stubs` directories are not provider roots.

`libcudadebugger.so.1` is not a Magnitude runtime requirement unless the final recursive dependency
closure or an NVIDIA-supported provider contract proves otherwise. Its presence beside WSL CUDA
libraries does not make it an application dependency.

## Static compatibility rules

Compatibility is evaluated for one immutable pack and each exact selected device. It has four
static gates followed by executable validation.

### 1. Driver and device discovery

The loaded driver must initialize CUDA and enumerate the selected physical device. Older PTX cannot
make an old driver recognize newer hardware. Failure is driver or device unavailability, not model
incompatibility.

### 2. Bundled runtime compatibility

The installed driver must meet every requirement of the artifact-owned runtime closure. Missing or
inconsistent artifact-owned libraries are installation defects. Host libraries not named by the
declared capability contract are not searched speculatively.

### 3. Device-image applicability

For ordinary PTX:

```text
ordinaryPtxApplicable(image, device, driver) =
  driver.api >= image.minimumSupportedDriverApiForPtxJit
  AND device.computeCapability >= image.virtualComputeCapability
```

The first comparison is the PTX-language/JIT relationship. The second is the PTX-target/hardware
relationship. They are independent: compiling `compute_80` with CUDA 12.9 may still emit PTX 8.8,
so the old virtual target does not lower the JIT requirement.

Ordinary older PTX targets are forward-compatible to higher physical compute capabilities. They may
omit source paths and instructions enabled only for newer virtual architectures, so hardware
coverage does not imply optimal performance.

Architecture-specific PTX such as `90a` or `120a` requires its exact documented architecture and
never uses a numeric lower-bound comparison. Family-specific PTX is applicable only according to
its declared NVIDIA family relation.

For ordinary desktop cubins, binary applicability requires the same compute-capability major and a
device minor version greater than or equal to the cubin target minor. Architecture-specific and
family-specific cubins use their corresponding exact or family relation. Cubins bypass PTX JIT but
do not bypass runtime, driver-feature, or hardware requirements.

### 4. Pack eligibility

A pack is statically eligible for a selected device only when:

- its host and ABI match;
- its bundled runtime closure is supported; and
- at least one embedded device image is applicable to that device.

Multi-device eligibility is calculated per device. A maximum architecture, minimum architecture,
or uncorrelated list of device numbers cannot prove that every selected device has an applicable
image.

Static eligibility describes NVIDIA's supported configuration. An execution result outside that
range, if best-effort operation is ever permitted, remains a separate fact and does not silently
widen the published support contract.

## Execution readiness

Static eligibility predicts that a pack should execute. It does not prove that provider resolution,
module loading, JIT compilation, kernel launch, or synchronization works in the installed
environment.

After acquisition, ICN:

1. loads the exact module and recursive runtime closure;
2. registers the selected device;
3. launches a bounded kernel through the same backend path used by inference; and
4. synchronizes and validates its result.

Only a successful synchronized canary publishes CUDA as executable. Failures retain their actual
stage and native error, including provider unavailable, stub provider, unsupported PTX version,
JIT compiler unavailable, no applicable binary, module-load failure, kernel-launch failure, and
synchronization failure.

The CUDA fatbinary loader owns final image selection. Magnitude records the set and highest-ranked
applicable declared images, but does not claim to have observed the chosen image unless the native
layer can report it. Successful execution and expected applicable image remain separate facts.

## Release tiers and build matrix

Linux x64 and Linux ARM64 expose the same tier semantics. Host CPU architecture never determines a
CUDA toolkit generation. The initial production matrix is four independent concurrent jobs:

```text
(linux-x64, compatibility)
(linux-x64, modern)
(linux-arm64, compatibility)
(linux-arm64, modern)
```

The compatibility tier uses the oldest validated toolkit that builds the unmodified pinned backend
and passes correctness and representative-performance validation. CUDA 11.8 is the first candidate;
CUDA 12.0 and later are tested in ascending order if it fails. Its required compatibility image is
one ordinary `80-virtual` target. That single image supplies Ampere-and-newer hardware coverage;
additional Ada or Hopper images are optimization work and require measured benefit because each
multiplies CUDA compilation.

The modern tier uses one exact validated toolkit across both hosts. CUDA 12.9 is the initial
toolchain and retains the effective Ampere, Hopper, and Blackwell paths currently requested as
`80-virtual`, `90-virtual`, and `120-virtual`. Release inspection records that the effective
Blackwell image may be architecture-specific, such as `120a`. Ada `89`, additional virtual targets,
and native cubins are added only after measured cold-start or inference benefit justifies their
build-time and size cost.

The compatibility tier exists to serve older supported driver JITs. The modern tier exists to use a
newer compiler and newer architecture source paths. Repeating an older virtual target in a modern
pack does not broaden driver compatibility when every image still declares the modern PTX ISA.

## Pack selection and fallback

Runtime selection is deterministic:

1. Resolve the host CUDA provider, initialize it, and enumerate exact driver/device facts.
2. Consider every pack for the exact host; manifest order has no meaning.
3. Evaluate the complete static rules for each pack and selected device set.
4. Rank eligible packs by explicit tier and applicable image policy.
5. Acquire and validate exactly one selected pack.
6. Require the synchronized execution canary before publishing CUDA as executable.

If the modern pack is not statically eligible, an eligible compatibility pack may run the device
through ordinary older PTX. If no pack is eligible, CUDA is unsupported under the observed
driver/device state and diagnostics identify the failed gate.

Failure after selection is an operational or release-contract result. It never becomes no GPU,
insufficient memory, model incompatibility, or an empty compatible-model set. Another CUDA tier may
be attempted only when it was independently eligible and the failure is classified as pack-specific;
the failed attempt and fallback remain observable. CPU or Vulkan fallback follows explicit product
policy and is never presented as successful CUDA execution.

## Preparation and calibration

After readiness, ICN prepares the kernel families required by calibration and synchronizes them
before measurement. PTX JIT and lazy module work are excluded from timed samples.

Cold bootstrap projects preparation into the shared startup lifecycle:

```ts
{
  _tag: "Starting",
  phase: {
    _tag: "PreparingBackend",
    backend: {
      _tag: "Cuda",
      hardwareLabel: "NVIDIA GeForce RTX 3060"
    }
  }
}
```

The local-inference subsystem owns the underlying state. Startup only projects it while it blocks
initial local readiness. Opaque JIT work reports operation and elapsed time, not fabricated
percentages. Performance calibration failure preserves execution and memory-fit truth while
reporting performance as unavailable.

## Caching and invalidation

The NVIDIA driver may cache JIT-produced native code. That cache is an optimization outside
Magnitude's correctness model and may be absent, unwritable, evicted, or invalidated.

Magnitude may cache successful readiness and calibration independently. Readiness identity includes
at least:

- artifact digest and inspected device-image contract;
- compiler and bundled runtime identities;
- native build and backend ABI;
- loaded provider identity and driver API/version;
- selected device identities and compute capabilities;
- operating-system/host contract; and
- canary method identity.

A behavior-affecting change is a cache miss. Loading, JIT, canary, preparation, and calibration
failures are not persisted as compatibility facts. Cache deletion changes latency and recomputation,
not compatibility.

## Observable state

The system retains distinct typed facts or results for:

- CUDA provider resolution;
- driver initialization and enumerated devices;
- artifact static eligibility and failed gate;
- artifact acquisition and integrity;
- runtime closure and module loading;
- device registration;
- execution canary;
- kernel preparation;
- performance calibration; and
- model runtime compatibility and memory fit.

Hardware presentation may say CUDA acceleration only when the selected backend is executable. A
discovered GPU remains present even when no CUDA pack is eligible or executable. Recommendation
policy consumes compatibility, fit, and performance independently and never infers model
incompatibility from missing performance.

## Acceptance criteria

- Runtime compatibility never depends on an ambient CUDA toolkit installation.
- Final fatbinary inspection, not requested compiler flags, defines the device-image contract.
- PTX eligibility uses the emitted `.version` and its supported driver-JIT floor; CUDA-major runtime
  compatibility cannot substitute for it.
- CUDA 12.9 PTX 8.8 is not represented as supported by driver API `12000`; its supported JIT floor
  is `12090`.
- Ordinary PTX target comparison is independent of PTX ISA comparison and remains directional.
- Architecture-specific and family-specific PTX and cubins never use the ordinary numeric rule.
- WSL CUDA resolution works through the declared WSL provider root without system symlinks.
- The actual loaded provider path and bounded resolution failures remain observable.
- Module loading without a successful synchronized canary never publishes executable CUDA.
- Compatibility and modern tiers have identical semantics on Linux x64 and ARM64 and build in
  parallel.
- Extra PTX targets or cubins require measured benefit; they are not added for nominal coverage.
- CUDA loading, JIT, execution, or calibration failure never becomes model incompatibility, no
  compatible models, or a cached successful empty portfolio.
