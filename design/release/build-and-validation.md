---
applies_to:
  - .github/workflows/release-build.yml
  - .github/workflows/release-checks.yml
  - .github/workflows/release-candidate-dry-run.yml
  - packages/release/scripts/assemble.ts
  - packages/release/scripts/build/**
  - packages/release/scripts/matrix.ts
  - packages/release/scripts/validate-host.ts
  - packages/release/src/targets.ts
  - inference/scripts/compile.ts
---

# Release build and validation

Release builds produce the exact archives that may be published. Validation operates on those final
archives, not only on intermediate build outputs.

## Build inputs

- Every job builds one pinned source commit and one Changesets-owned version.
- Version-dependent source is generated in each clean checkout before release code is loaded.
- Planner inputs are generated once and shared by every host build.
- Toolchains, backend features, CUDA targets, and shader compiler versions are explicit release
  inputs. Ambient runner packages must not enable optional native features.

## Linux build baseline

Every Linux host, CPU base, CUDA pack, and Vulkan pack builds on its architecture's Ubuntu 22.04
runner. CUDA 11.8 and CUDA 12.9 use the same userspace baseline.

Ubuntu 22.04's Vulkan headers are older than the Vulkan API types used by the pinned llama.cpp.
Vulkan jobs therefore construct a build-only SDK prefix from Vulkan-Headers 1.4.313 and shaderc
`v2023.8` `glslc`, while linking against Jammy's system Vulkan loader. The headers and shader
compiler are not included in the release and do not become customer dependencies.

## Archive validation

Assembly validates every host base and every legal base-plus-backend composition. For Linux, every
ELF file is inspected with `readelf`; release inputs are never executed through `ldd`.

Assembly rejects:

- the wrong ELF class, machine architecture, or program interpreter;
- glibc requirements above 2.35 or GLIBCXX requirements above 3.4.30.

Archive layout, artifact size and digest, native-build identity, backend ABI, planner-input equality,
and backend compatibility metadata are also validated before the manifest is emitted.

## Execution gates

Each host build extracts and executes its CLI, ACN, and ICN-base archives. It verifies versions,
embedded ripgrep, ICN identity, backend eligibility, readiness, authenticated health, and managed
shutdown with inherited Unix library search paths cleared.

Linux host archives are then downloaded by separate Ubuntu 22.04 consumer jobs for x64 and arm64
and executed again without reusing the build workspace. This catches dependencies accidentally
satisfied by the build job.

The complete candidate gate additionally installs the packed npm package through Node and Bun,
acquires CLI, ACN, and ICN through their production paths from an empty data root, reaches ACN/ICN
readiness and local-model recommendation readiness, shuts down the exact owned processes, and proves
the validated cache works when the artifact endpoint is unavailable.

Pull requests run the complete build and acceptance graph without publishing. A manually dispatched
Linux x64 dry run exercises the CPU-only production path but cannot authorize publication.

## Publication gate

Publication requires the complete configured artifact graph. A runner-only build success, a
host-scoped dry run, static inspection without execution, or execution without final-archive
inspection is insufficient.
