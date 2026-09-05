---
applies_to:
  - packages/release/src/acquisition*.ts
  - packages/release/src/archive*.ts
  - packages/release/src/artifact-download*.ts
  - packages/release/src/contracts.ts
  - packages/release/src/launcher*.ts
  - packages/launcher/src/wrapper.ts
  - packages/launcher/scripts/build-launcher.ts
  - packages/daemon-management/src/binary.ts
  - packages/icn/src/lifecycle/release-installation.ts
---

# Release acquisition

Runtime acquisition installs only artifacts selected from the version's release manifest.

## Ownership

- The npm launcher acquires CLI.
- The private daemon-management package acquires ACN.
- The ICN lifecycle acquires the ICN base and optional backend pack and composes their installation.

These responsibilities do not overlap.

## Integrity and installation

- The manifest is fetched from the configured GitHub release origin over HTTPS.
- Every downloaded artifact must match the manifest byte size and SHA-256.
- Downloads and extraction are bounded. Range responses must identify the exact requested bytes and
  one consistent representation; unsupported ranges fall back to bounded sequential transfer.
- Archives accept regular files only at validated relative paths.
- Installations are addressed by artifact digest and published atomically only after acquisition
  integrity and executable identity are verified. Published cache entries are trusted on subsequent
  launches; a missing executable is a cache miss.
- A valid cached installation remains usable offline. A missing or invalid installation that cannot
  be repaired fails explicitly.

## Backend composition

Apple arm64 selects Metal. Linux considers compatible CUDA, then compatible Vulkan, then CPU only
when successful capability probes show that no supported accelerator is usable.

Authentication, acquisition, capability probing, ABI validation, module loading, and device
registration are operational failures. They do not silently become CPU fallback. A selected pack
must name the installed base and match its native-build identity and backend-module ABI.
