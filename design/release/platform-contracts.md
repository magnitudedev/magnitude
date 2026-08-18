---
applies_to:
  - packages/release/**
  - packages/icn/src/lifecycle/release-installation.ts
  - packages/icn/src/lifecycle/installation-environment.ts
  - packages/sdk/src/binary.ts
  - inference/**
  - .github/workflows/release-build.yml
---

# Release platform contracts

A platform contract defines what a customer machine may be required to provide. Release artifacts
must not depend on anything else.

## Dependency ownership

Every native dependency belongs to exactly one class:

- **Artifact-owned:** shipped in the selected archive set, integrity-covered, and resolved through
  installation-relative loader paths.
- **Platform-owned:** part of the declared operating-system ABI for that host target.
- **Capability-owned:** supplied by a selected accelerator environment, such as an NVIDIA driver,
  Vulkan loader, or Metal framework.

An unclassified dependency is a release defect. Build tools, SDKs, package-manager prefixes,
compiler runtimes outside the declared platform ABI, ambient search paths, and files from the build
machine are never customer dependencies.

## Common runtime facilities

Every supported host must provide:

- a writable per-user data directory that supports atomic rename and execution of installed native
  files;
- ordinary child-process creation and a long-lived background ACN process;
- local loopback TCP sockets for client, ACN, and ICN communication; and
- DNS, trusted certificate roots, and outbound HTTPS for initial artifact acquisition and repair.

Once a complete installation is cached, release acquisition does not require network access. Model
acquisition has its own network requirements.

Customer systems do not need Rust, Bun, CMake, C/C++ compilers, CUDA toolkits, Vulkan SDKs,
developer headers, OpenSSL packages, OpenMP packages, or build-system package-manager prefixes.
The npm launcher requires its supported Node.js runtime, but acquired native artifacts do not use
Node.js as a native dependency.

## GNU Linux contract

Both `linux-x64-gnu` and `linux-arm64-gnu` target Ubuntu 22.04-compatible userspace.

| Property | x64 | arm64 |
| --- | --- | --- |
| ELF class | ELF64 | ELF64 |
| Machine | x86-64 | AArch64 |
| Interpreter | `/lib64/ld-linux-x86-64.so.2` | `/lib/ld-linux-aarch64.so.1` |
| Maximum glibc requirement | `GLIBC_2.35` | `GLIBC_2.35` |
| Maximum libstdc++ requirement | `GLIBCXX_3.4.30` | `GLIBCXX_3.4.30` |

Linux artifacts may dynamically require their architecture's interpreter listed above and only
these platform libraries:

- `libc.so.6`
- `libdl.so.2`
- `libgcc_s.so.1`
- `libm.so.6`
- `libpthread.so.0`
- `libresolv.so.2`
- `librt.so.1`
- `libstdc++.so.6`
- `libutil.so.1`

A CUDA composition may additionally require driver-owned `libcuda.so.1`. A Vulkan composition may
additionally require loader-owned `libvulkan.so.1`. CPU compositions may require neither. CUDA
toolkit libraries required at runtime are artifact-owned and shipped in the CUDA pack.

Linux executables resolve owned libraries from `../runtime`; libraries and backend modules resolve
from their own directory or `../runtime`. Allowed loader paths are therefore `$ORIGIN` and
`$ORIGIN/../runtime`. Releases must not require `LD_LIBRARY_PATH`.

## Apple contract

Apple artifacts may depend on operating-system libraries and frameworks included with the supported
macOS deployment target. Metal is capability-owned by macOS. Homebrew, MacPorts, Xcode, standalone
SDKs, and developer-tool libraries are not platform dependencies.

The supported macOS floor is macOS 13.0 for both Apple arm64 and Apple x64. Release builds use the
newest selected SDK while compiling and linking every Apple-native image for that floor. The floor
does not select an older Metal implementation: guarded operating-system APIs and actual GPU
capabilities remain runtime decisions, so the same artifact uses newer facilities on newer hosts.
macOS 12 and older are outside the platform contract because the Bun runtime embedded in the CLI and
ACN requires macOS 13 or newer.

Artifact-owned libraries use `@loader_path` or declared installation rpaths and must not require
`DYLD_LIBRARY_PATH`. Every Mach-O image must match its target architecture and must not declare a
deployment target newer than macOS 13.0. The release configuration is the authority for that floor;
a runner label alone is not a support contract. Before packaging, the Apple build validates the
expected architecture and deployment target of every executable and native library with Apple's
`vtool`. Those validated files are the exact inputs to the deterministic archive builder.

## Required guarantees

- Build-host contents cannot introduce a dependency or raise a platform floor.
- Every non-platform dependency is shipped when redistribution permits; otherwise it is an explicit
  capability dependency checked before backend selection.
- Base and accelerator artifacts for one host share the same platform contract.
- Dynamic-loader failure remains distinct from protocol-decoding failure and retains bounded native
  diagnostics.
