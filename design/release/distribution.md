---
applies_to:
  - packages/release/src/contracts.ts
  - packages/release/src/targets.ts
  - packages/release/scripts/assemble.ts
  - packages/release/scripts/build/**
  - packages/cli/package.json
---

# Release distribution

Magnitude distributes one versioned release as an npm package plus a fixed graph of native
artifacts. The release graph is product configuration, not a plugin system.

## Published artifacts

| Artifact | Published for | Contents |
| --- | --- | --- |
| CLI | every host | one `bin/magnitude-cli` executable |
| ACN | every host | one `bin/magnitude-acn` executable with embedded ripgrep |
| ICN base | every host | ICN executable, planner inputs, common runtime libraries, and CPU modules |
| ICN backend pack | compatible hosts | one Metal, CUDA, or Vulkan module family and its redistributable runtime libraries |

Published hosts are Apple arm64, Apple x64, Linux GNU arm64, and Linux GNU x64. Windows artifacts
are not published. Each backend pack names exactly one required ICN base and must have the same
native-build identity and backend-module ABI as that base.

Apple arm64 publishes Metal. Linux arm64 and x64 publish Vulkan plus CUDA 11.8 and CUDA 12.9.
CUDA device-image and driver compatibility is defined by
[CUDA compatibility](../inference/cuda-compatibility.md).

## Release identity

The release manifest identifies one version, source commit, ACN coordination revision, and the
complete native artifact graph. Each artifact record contains its host, kind, filename, byte size,
SHA-256, and the compatibility facts required for runtime selection. ICN records also contain their
native-build identity and backend-module ABI.

The manifest does not describe build provenance or duplicate platform policy. Platform support is
a property of the release target and is enforced while building and accepting the candidate.

## Distribution contract

A conforming release satisfies all of the following:

- Every published artifact is present exactly once and matches its manifest size and SHA-256.
- Every executable and library depends only on artifact-owned files, its host platform contract,
  and the capability dependencies of the selected backend.
- A backend pack composes with exactly its required base and cannot alter the base platform floor.
- Final artifacts pass build-host-independent validation before publication.
- GitHub assets are public before npm is published because the npm launcher acquires those assets.

The concrete host dependency contracts are defined in
[Platform contracts](./platform-contracts.md). Build acceptance is defined in
[Build and validation](./build-and-validation.md). Runtime installation is defined in
[Acquisition](./acquisition.md), native CLI package-manager updates are defined in
[CLI updates](./client-updates.md), and remote publication is defined in
[Publication](./publication.md).
