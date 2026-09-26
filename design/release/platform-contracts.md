---
applies_to:
  - packages/release/**
  - packages/icn/src/lifecycle/release-installation.ts
  - packages/icn/src/lifecycle/installation-environment.ts
  - inference/scripts/**
  - inference/seismic/backends/cuda/src/nvrtc.rs
  - .github/workflows/release-build.yml
---

# Release platform contracts

A platform contract defines what a customer machine may be required to provide. Release artifacts
must not depend on anything else.

Hardware, driver and operating-system floors are defined once by the inference platform contract,
[`inference/docs/compatibility.md`](../../inference/docs/compatibility.md). Release configuration
and these documents apply those floors; they do not restate or independently change them.

## Dependency ownership

Every native dependency belongs to exactly one class:

- **Artifact-owned:** shipped in the host's artifact, integrity-covered, and resolved through
  installation-relative loader paths.
- **Platform-owned:** part of the declared operating-system ABI for that host target.
- **Capability-owned:** supplied by an accelerator environment, such as an NVIDIA driver, Vulkan
  loader, or Metal framework, and loaded at runtime only when present.

An unclassified dependency is a release defect. Build tools, SDKs, package-manager prefixes,
compiler runtimes outside the declared platform ABI, ambient search paths, and files from the build
machine are never customer dependencies.

## Common runtime facilities

Every supported host must provide:

- a writable per-user data directory that supports atomic rename and execution of installed native
  files;
- ordinary child-process creation and a long-lived background ACN process;
- local loopback TCP sockets for client, ACN, and ICN communication, plus an optional additional ACN listener on a network interface when the user enables network access; and
- DNS, trusted certificate roots, and outbound HTTPS for initial artifact acquisition and repair.

Once a complete installation is cached, release acquisition does not require network access. Model
acquisition has its own network requirements.

Customer systems do not need Rust, Bun, CMake, C/C++ compilers, CUDA toolkits, Vulkan SDKs,
developer headers, OpenSSL packages, OpenMP packages, or build-system package-manager prefixes.
The desktop-bundled CLI needs neither npm nor a separately installed Node.js runtime.

## Inference accelerator dependencies

Each host has one inference artifact with every backend of that host compiled in. There are no
backend packs and no accelerator modules.

- **NVRTC 12.9 is the entire CUDA payload.** It is artifact-owned on Linux and Windows: the two
  standard NVRTC libraries of NVIDIA's pinned redistributable, shipped unmodified with NVIDIA's
  license notice in the installation's `runtime/` directory. There is no CUDA runtime, cuBLAS or
  other toolkit library; kernels are formed on the device and launched through the driver API.
- **Seismic loads NVRTC only from the installation's `runtime/` directory** beside `bin/`. An
  explicit engine-development setting may name another directory for work outside an
  installation layout; it is not part of the installed contract. There is no search-path fallback.
- **Driver and loader libraries are capability-owned.** `libcuda.so.1` / `nvcuda.dll` and
  `libvulkan.so.1` / `vulkan-1.dll` are loaded at runtime when present and are never link-time
  dependencies. A host without them runs the remaining backends.
- **Seismic discovery is the only eligibility authority.** Driver, API, compute-capability and
  feature floors are enforced when devices are discovered. Release tooling, acquisition and the
  service never re-derive them and never select artifacts by accelerator.
- **CPU code targets the architecture baseline.** Release builds compile for baseline `x86_64` and
  `aarch64` with no `target-cpu`; instruction-set tiers are detected at runtime and the engine
  enforces its x86 floor.

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

The inference executable resolves owned libraries from exactly `$ORIGIN/../runtime`.
Redistributed runtime libraries carry no loader path, or only `$ORIGIN` or `$ORIGIN/../runtime`.
Releases must not require `LD_LIBRARY_PATH`.

### Linux graphical desktop

The Electron application additionally requires the distribution's graphical userspace: GLib/GIO,
GTK3, NSS/NSPR, ATK/AT-SPI, D-Bus, Cairo/Pango, CUPS, X11/XCB, xkbcommon, GBM/DRM, expat, udev,
and ALSA libraries, plus util-linux for installation admission and Polkit/pkexec for update authorization. These are package-manager dependencies, including when the application uses
Wayland. They do not become requirements of the headless CLI or inference artifacts. FFmpeg,
Electron, and the bundled rendering libraries are artifact-owned. Package metadata must resolve
the graphical dependencies on each supported distribution without relying on optional recommends
for directly linked libraries. Native consumer checks validate the final installed application's
loader closure and sandbox permissions; a build-host launch is insufficient.

## Windows contract

The inference artifact ships NVRTC and the Microsoft CRT required by its import graph in
`runtime/`, which the managed parent places first on the service's DLL search path. `nvcuda.dll`
and `vulkan-1.dll` are capability-owned and loaded at runtime. NVIDIA publishes the NVRTC
libraries without Authenticode signatures; their integrity is the pinned archive digest verified
at build time, and they are redistributed unmodified.

## Apple contract

Apple artifacts may depend on operating-system libraries and frameworks included with the supported
macOS deployment target. Metal is capability-owned by macOS. Homebrew, MacPorts, Xcode, standalone
SDKs, and developer-tool libraries are not platform dependencies.

The supported macOS floor is macOS 15.0 for both Apple arm64 and Apple x64. The Metal backend
compiles every library with precise math modes that macOS 15 introduced, and the desktop bundles
the service, so the floor applies to the whole Apple product. Release builds use the newest
selected SDK while compiling and linking every Apple-native image for that floor. Guarded
operating-system APIs and actual GPU capabilities remain runtime decisions, so the same artifact
uses newer facilities on newer hosts. macOS 14 and older are outside the platform contract.

Artifact-owned libraries use `@loader_path` or declared installation rpaths and must not require
`DYLD_LIBRARY_PATH`; the Apple inference artifact has none and its executable carries no rpath. Every
Mach-O image must match its target architecture and must not declare a deployment target newer
than macOS 15.0. The application bundle and desktop declare the same minimum system version. The
release configuration is the authority for that floor; a runner label alone is not a support
contract. Before packaging, the Apple build validates the expected architecture and deployment
target of every executable and native library with Apple's `vtool`. Those validated files are the
exact inputs to the deterministic archive builder.

## Required guarantees

- Build-host contents cannot introduce a dependency or raise a platform floor.
- Every non-platform dependency is shipped when redistribution permits; otherwise it is a
  capability dependency loaded at runtime, whose absence only removes the backends that need it.
- Every host's inference artifact contains every backend of that host and nothing selected per
  machine.
- Dynamic-loader failure remains distinct from protocol-decoding failure and retains bounded native
  diagnostics.

Linux application packages must install their payload directories as root-owned mode0755 and
remove group/other write permission from payload files while preserving executable and sandbox
mode bits. Package validation rejects unprotected directories or files before publication; the
privileged updater independently checks installed publisher-trust ancestry before authorizing an update.
