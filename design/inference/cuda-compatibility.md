---
applies_to:
  - packages/release/**
  - packages/icn/**
  - packages/icn-protocol/**
  - packages/acn-protocol/src/schemas/acn-health.ts
  - packages/acn/src/icn/**
  - packages/daemon-management/src/acn-jit/**
  - inference/crates/icn-server/**
  - inference/crates/icn-contracts/**
  - inference/scripts/**
  - .github/workflows/release-build.yml
---

# CUDA compatibility and preparation

Magnitude owns the CUDA backend module and its runtime libraries. The host owns the NVIDIA driver.
A locally installed CUDA toolkit is neither required nor consulted at runtime.

CUDA discovery, static artifact compatibility, and successful execution are different facts. A
driver that loads and enumerates a GPU has not yet proven that a shipped backend can execute.

## Host driver resolution

Every ICN process uses one resolver for eligibility and backend loading. Linux first asks the
system loader for `libcuda.so.1`. On WSL it also tries the WSL driver projection under
`/usr/lib/wsl`; native Linux fallbacks are limited to standard multiarch/NVIDIA, NixOS, and
NVIDIA-container driver roots. Toolkit `stubs` and `compat` directories are never candidates.
Windows resolves `nvcuda.dll` from the system directory.

The resolver validates required driver symbols, initializes CUDA, enumerates devices, rejects the
stub provider, retains the loaded handle, and reports its actual path, driver API, compute
capabilities, and hardware labels. Magnitude never changes host loader configuration or creates
system symlinks.

## Shipped artifact contract

Each CUDA pack declares only facts needed for current PTX-only artifacts:

- exact toolkit release and compiler identity;
- every embedded PTX image's `.version`, numeric target, and whether the target is ordinary or
  architecture-specific; and
- the reviewed minimum driver API capable of JIT-compiling each image's PTX ISA.

Release assembly obtains these facts from the finished backend module with NVIDIA tooling and
publishes the inspected result as artifact truth. The configured architecture list is solely a
compiler input and is not duplicated as a second compatibility declaration. An unknown PTX version
has no implicit driver floor and fails publication.

Cubins and additional target relations are not part of this contract until Magnitude actually
ships them.

## Build matrix

Linux x64 and ARM64 build the same two compiler configurations independently:

| Toolkit | PTX targets |
| --- | --- |
| CUDA 11.8 | ordinary `80` |
| CUDA 12.9 | ordinary `80`, ordinary `90`, ordinary `120` |

These four host/toolkit jobs run concurrently on Ubuntu 22.04 for both x64 and SBSA ARM64 so every
Linux pack retains the release userspace ABI baseline. Host CPU architecture never selects a CUDA
toolkit generation.

Release compatibility inspection consumes `cuobjdump` output as a stream and retains only the
distinct PTX image facts. Its memory use must not scale with the textual PTX dump.

Additional PTX targets and native cubins require measured startup or inference benefit because
they increase build time and artifact size.

## Static selection

For each pack on the current host, Magnitude checks:

1. each enumerated CUDA architecture has at least one applicable image; and
2. the installed driver API meets that image's inspected PTX JIT floor.

An ordinary PTX image applies when the physical compute capability is greater than or equal to its
virtual target. An architecture-specific image applies only to its exact numeric architecture.
These device-target comparisons are independent from the driver-to-PTX comparison.

Eligible packs are ordered by architecture specificity, highest applicable target, highest
driver-supported PTX version, toolkit/compiler version, then stable artifact identity. Manifest
order has no meaning. If no CUDA pack is eligible, normal explicit
accelerator/CPU product policy continues without claiming CUDA succeeded.

## Startup preparation

For a selected CUDA installation, ICN starts its persistent planning worker and establishes hardware
calibration before publishing readiness. A valid persisted record is reused. On a miss, that worker
runs the model-free operations. Untimed launches and synchronization prove backend execution and may
trigger PTX JIT before timed samples. The worker remains alive for later assessments, preserving its
process-local CUDA context and modules. The result is persisted and injected into the assessor.

ACN projects this operation through startup state without owning it:

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

The TUI remains responsive and shows the phase under `Starting Magnitude`. It does not fabricate a
percentage for opaque driver JIT work. The NVIDIA driver may reuse its disk JIT cache, but Magnitude
does not depend on or manage that cache. Worker initialization runs on every CUDA ICN startup;
calibration measurement runs only on a cache miss.

Preparation failure is a CUDA execution/startup failure with its retained native diagnostic. It is
never rewritten as insufficient model memory, model incompatibility, unavailable performance, or
`No compatible models`.

## Result integrity

Once preparation succeeds, compatibility and memory capacity remain independent from performance.
Complete execution assessment always includes measured performance. A hardware-calibration, workload, or
estimator failure fails the assessment operation and is never rewritten as incompatibility or a
successful empty ranking result.

## Conformance

- WSL resolves its driver projection without symlinks or toolkit compatibility libraries.
- The final module contains inspectable PTX images whose compatibility metadata is derived from the
  module itself.
- CUDA 11.8 PTX 7.8 and CUDA 12.9 PTX 8.8 use their reviewed driver-JIT floors.
- Ordinary `compute_80` remains applicable to newer physical architectures.
- Architecture-specific targets never use the ordinary numeric-forward rule.
- A CUDA backend is not ready until an isolated synchronized operation succeeds.
- Cold preparation is visible as `PreparingBackend(Cuda, hardwareLabel)` during ACN bootstrap.
- Performance failure cannot create a successful empty ranking result.

## NVIDIA references

- [Minor-version compatibility and its PTX limitation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [CUDA toolkit and corresponding driver versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver)
- [CUDA 11.8 PTX ISA 7.8](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)
- [CUDA 12.9 PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html)
