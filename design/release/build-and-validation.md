---
applies_to:
  - .github/workflows/release*.yml
  - packages/release/scripts/**
  - packages/release/src/targets.ts
  - inference/scripts/compile.ts
  - .github/workflows/integrations.yml
  - scripts/*integrations.ts
  - integrations/**/package.json
  - integrations/**/scripts/**
  - packages/sdk/**
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

## Apple build baseline

Apple arm64 and Apple x64 target macOS 13.0. The release configuration passes that floor through
both `MACOSX_DEPLOYMENT_TARGET` and `CMAKE_OSX_DEPLOYMENT_TARGET`, ensuring that Rust, Cargo build
scripts, cc, CMake, Clang, and the linker share one minimum-version contract. The selected SDK may be
newer than macOS 13: newer operating-system APIs must remain weak-linked and availability-guarded,
while Metal kernels and GPU features continue to specialize for the actual runtime device.

The runner image is only a build environment. Changing or advancing that image must not change the
deployment target recorded in release artifacts. Before packaging, the Apple build validates every
executable and native library with Apple's `vtool`, selecting the expected release architecture and
rejecting a missing deployment declaration or a minimum newer than 13.0.

## Archive validation

Assembly validates every host base and every legal base-plus-backend composition. For Linux, every
ELF file is inspected with `readelf`; release inputs are never executed through `ldd`.

Assembly rejects:

- the wrong ELF class, machine architecture, or program interpreter;
- glibc requirements above 2.35 or GLIBCXX requirements above 3.4.30.

Apple compatibility is validated on the Apple build host, using Apple's own Mach-O tooling against
the exact files subsequently passed to the deterministic archive builder. Assembly does not
reimplement Mach-O parsing.

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
readiness and local-model ranking readiness, shuts down the exact owned processes, and proves
the validated cache works when the artifact endpoint is unavailable.

Pull requests run the complete build and acceptance graph without publishing. A manually dispatched
Linux x64 dry run exercises the CPU-only production path but cannot authorize publication.

## Publication gate

Publication requires the complete configured artifact graph. A runner-only build success, a
host-scoped dry run, static inspection without execution, or execution without final-archive
inspection is insufficient.

Harness companion packages have independent, exact package versions. The private SDK and wire
contract are bundled into each companion; they are not separately published. Selected companion
artifacts are verified available before a CLI release advertises them. An explicit integration-only release can publish a main-branch source commit
without rebuilding the CLI/native artifact graph. It must pass the same package tests and clean
consumer acceptance, and must verify the published integrity; an existing version is never replaced.

Integration preparation packs each selected companion once. Acceptance installs those exact
tarballs outside the workspace and loads the extension through the supported harness's native
package and resource loader under Node and Bun. Accepted bytes and their receipt are persisted;
publication does not repack them. Private workspace dependencies cannot escape into the packed
artifact. Shared SDK/wire changes trigger these checks as well as integration changes. Local
acceptance never publishes packages. Prereleases skip plugin release preparation/publication
and preserve the numeric RPC version until stable release preparation.
