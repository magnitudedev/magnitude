---
applies_to:
  - .github/workflows/release*.yml
  - .github/actions/linux-build-tools/**
  - packages/release/scripts/**
  - packages/release/src/targets.ts
  - scripts/accept-release-candidate.ts
  - inference/scripts/compile.ts
  - .github/workflows/integrations.yml
  - .github/workflows/desktop-native.yml
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
- Compiler-result caches may reuse objects matched by compiler inputs; clean release output,
  final linking, packaging, signing, and independent artifact acceptance remain mandatory.
- The workspace and CI use the same pinned Bun runtime. Runtime changes require native
  Windows pipe acceptance under Node, Bun, and compiled Bun, including unread replies after
  server closure, plus client and lifecycle regressions on the build host. Child compiler
  invocations use the executing build runtime rather than another Bun found through PATH.
- Windows installer helpers are compiled with the native MSVC toolchain for the installer's
  x86 process ABI. Cross-compilation success does not replace this native build check.
- Desktop assembly defaults to the build host or accepts an explicit supported target. Its
  Electron distribution, executable names, native resources and platform metadata all follow
  that target; assembly requires prebuilt matching service and native inputs. Cross-assembly
  does not replace installed-consumer acceptance on the target operating system.
- Desktop resources include the exact headless CLI and service built with the application version.
  Apple signs both compiled runtimes with their required JIT entitlements before notarization.
  Package acceptance executes both version commands; an application update cannot leave its CLI behind.
- Custom macOS installer acceptance runs explicitly in the Apple signing environment using isolated,
  signed and notarized applications. A temporary publisher key and local-only update origin keep
  fixture metadata separate from hosted releases. Public finite installation, foreground startup and Desktop startup must each install a successive
  replacement through the copied helper, verify matching CLI/service versions after each,
  retire prepared state and private helper/transaction storage, and leave each bundle accepted by
  signature, notarization-ticket and Gatekeeper checks. Native unit
  tests or injected verifier tests do not substitute for this gate.

## Linux build baseline

Every Linux host, CPU base, CUDA pack, and Vulkan pack builds on its architecture's Ubuntu 22.04
runner. CUDA 11.8 and CUDA 12.9 use the same userspace baseline.

Build-tool download caches are keyed by Ubuntu version, architecture, and the resolved APT package
plan. APT still resolves and installs dependencies normally; a cache hit never skips installation.
These archives are separate from CUDA downloads and survive CUDA's package-cache cleanup.

Ubuntu 22.04's Vulkan headers are older than the Vulkan API types used by the pinned llama.cpp.
Vulkan jobs therefore construct a build-only SDK prefix from Vulkan-Headers 1.4.313 and shaderc
`v2023.8` `glslc`, while linking against Jammy's system Vulkan loader. The headers and shader
compiler are not included in the release and do not become customer dependencies.

Linux desktop packaging runs its installer tooling under Node and validates the final package,
compressing the DEB once with zstd level 9 after finalizing its payload,
including a root-owned mode-04755 Chromium sandbox helper. Package permissions are a postcondition,
not an assumption about filesystem API calls. Installed-consumer acceptance must exercise ordinary
launch without sandbox-disabling test flags, verify the canonical CLI/application-menu path and
window class, and preserve the matched application/service bytes.
RPM packaging disables build-root rewriting of the prebuilt payload, including stripping and debug
section extraction. The desktop carries Magnitude's license alongside Electron's existing notices.
Each DEB/RPM producer emits a schema-validated artifact record for the final copied package,
including its format-specific identity, host, filename, byte size and SHA-256. That record is build
metadata; it does not replace installed-consumer acceptance or authorize publication.
Linux candidate assembly requires both formats for each selected Linux host. Native package tooling
verifies the embedded package name, version, architecture and sandbox permissions against the
release target; an installer extension or matching checksum alone is insufficient.

## Windows build baseline

Windows CPU and accelerator artifacts target x64 MSVC. CMake receives that target explicitly, including when the
build tools run under x64 emulation on Windows ARM; the build machine's processor must not select
ARM backend variants for an x64 artifact. Desktop, service, and installer builds share the native
host toolchain discovery. The Node import library is verified against the selected Node release's
checksums and remains a build-only input.

The engine package includes the Microsoft C++ runtime DLLs required by its native import graph.
Build validation resolves imports only against the owned payload, the selected toolchain's x64 CRT
redistributable, and Windows system libraries/API sets. An ambient developer PATH or installed VC
redistributable cannot satisfy a missing package dependency. Redistributable DLL imports are checked
recursively; the resulting files use the existing installation-owned runtime directory.
Backend builds initialize the same native compiler environment as host builds and use Ninja.
CUDA pack dependencies are resolved from explicitly selected toolkit DLLs; Vulkan and CUDA driver
libraries are capability-owned only for their respective accelerator compositions. Magnitude-built
backend DLLs are signed before archiving; vendor runtime DLLs retain their vendor signatures.
A manually dispatched Windows-backend-only check builds the same accelerator archives without
publishing or claiming full candidate acceptance. GPU execution validation composes the final base
and accelerator pack on a driver-equipped Windows host without development toolkits, verifies GPU
allocations and owned module paths, and exercises generation, streaming, cancellation, concurrent
admission, model reload, and worker cleanup.
An independent Windows consumer extracts and runs the final archives, checks their metadata,
and exercises engine readiness and parent-loss shutdown before candidate assembly can pass.
Production Windows packaging uses Artifact Signing with an explicit publisher identity. Owned code,
the native CLI launcher, the embedded uninstaller, and the final installer are signed and timestamped before checksums are
recorded. Publisher and signature validation fail the build; missing credentials cannot produce a
production installer. Bundling preserves the signed CLI and service bytes from their archives.
Local unsigned builds carry no production trust claim.
The independent consumer installs and uninstalls the accepted installer under a fresh user profile,
verifies installed registration and CLI versions, compares bundled CLI/service bytes to their
accepted archives, and requires publisher signatures for production inputs.
The consumer also verifies the inference executable and every shipped engine/runtime DLL; engine
code must carry Magnitude's timestamped signature, while Microsoft CRT files retain Microsoft's signature.
Signing credentials belong to the protected Windows signing environment; ordinary pull-request validation is unsigned.

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

## Apple signing and notarization

Trusted release jobs import a Developer ID Application identity and a notarytool API-key profile into
an ephemeral keychain. Untrusted validation jobs use explicit ad-hoc signing and cannot produce
production acceptance receipts. Ad-hoc execution omits Hardened Runtime because it has no team
identity for library validation; production runtime acceptance requires Developer ID. Native libraries and Bun-embedded native files are signed before
embedding; executables use Hardened Runtime and the Bun executables receive JIT entitlements.
Electron nested code is signed from the inside out. Electron receives its JIT entitlement; the
bundled service retains Bun's separate JIT profile. No broad library-validation exception or device
permissions are enabled by default. Framework symlinks remain intact in the platform installer.
The macOS update extraction executable ships inside the sealed application resources and receives
the native-helper entitlement profile, without a JIT entitlement. The packaged update configuration
matches the build's Desktop configuration so foreground preparation uses the same publisher trust.
Developer ID builds compile the Apple Team ID into both Desktop and the CLI; a missing or malformed
identity fails the build. Installed runtime environment variables cannot replace that identity.

Apple must accept the CLI, inference payload, desktop, app, and backend submissions. A rejected or incomplete
submission fails the build and retains diagnostic logs. The app ticket is stapled and validated before
final archiving and checksums. Private receipts bind publisher, commit, submissions, and final native
archive digests. Independent Apple consumer jobs execute the downloaded host archives and verify
signatures and the stapled app. Real login/permission UI acceptance remains a signed macOS test.
Desktop consumer acceptance mounts the final DMG read-only, verifies its sealed bundle and matched
service version, and executes lifecycle tests against the extracted release ICN base. It covers
hidden and concurrent startup, close-to-tray, renderer recovery, full Quit, and owner-crash cleanup.
Installer packaging and acceptance allow bounded retries when macOS reports a busy mounted image;
they retain the attached device identity because an unsuccessful eject may already have removed
the mount path. They never force-detach it, and persistent detach failures still fail the build.
Publication requires a consumer receipt covering the desktop's exact final bytes.
The Mac update ZIP has its own final digest and must be independently consumed alongside the DMG.
Apple consumers extract it, compare its app with the installer payload, verify the sealed signature
and stapled ticket, and execute the extracted app's lifecycle. A DMG-only receipt cannot authorize
publication of an update ZIP. These checks establish payload acceptance, not updater replacement.
Release lifecycle acceptance checks application behavior and native resource ownership, not
presentation copy, fonts, colors, fixed card counts, or page geometry. Visual checks belong in
the separate renderer acceptance fixtures and do not gate a native release.

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
Managed inference starts as its own process-group leader. Parent-channel loss acceptance requires
the watchdog to terminate that group, including workers; it does not expect a graceful zero exit.

Linux host archives are then downloaded by separate Ubuntu 22.04 consumer jobs for x64 and arm64
and executed again without reusing the build workspace. This catches dependencies accidentally
satisfied by the build job.
Those consumers verify installer descriptors against the downloaded bytes, install the DEB through
the package manager, compare the installed service to the accepted ACN executable, and exercise
login configuration, hidden startup, CLI ownership, application-menu launch and full Quit in an
isolated graphical session. Virtual-display execution does not certify physical logout or tray
behavior on every desktop environment.
The same consumers install the accepted RPM in a fresh Fedora userspace with optional package
dependencies disabled, compare the installed service bytes and sandbox permissions, and repeat
the installed desktop lifecycle under an unprivileged user. Container init must reap detached
children so process-exit checks retain their ordinary operating-system meaning. This gate does
not replace native desktop-environment or real logout acceptance.
The container syscall policy must permit Chromium to create its sandbox namespaces while retaining
the default restrictions on other operations. Acceptance never disables Chromium's sandbox.

The complete candidate gate runs the installed desktop’s bundled CLI and service, acquires ICN from an empty data root, reaches ACN/ICN
readiness and local-model ranking readiness, shuts down the exact owned processes, and proves
the bundled CLI works when the artifact endpoint is unavailable.
On Linux this gate runs in a disposable Ubuntu consumer, explicitly installs the candidate DEB,
and passes the acquired candidate ICN installation to the installed desktop lifecycle test. It must
observe a Ready service; an intentionally missing engine only certifies failure handling and cannot
satisfy candidate bootstrap acceptance. Engine readiness does not imply model-serving acceptance.

A manual macOS Intel CPU consumer downloads a selected run's final host archive, verifies its
digest and native identity, installs a selected shipped catalog model through the canonical
catalog operation, and performs real inference with its locked target and draft or projector configuration.
Vision acceptance sends an actual image through the shipped projector and checks the final answer.
It exercises streaming, concurrent requests, prefill, unload and reload on native Intel
hardware, retaining results and server diagnostics without publishing or replacing candidate gates.

Pull requests run the complete build and acceptance graph without publishing. A manually dispatched
Linux x64 dry run exercises the CPU-only production path but cannot authorize publication.

## Publication gate

Publication requires the complete configured artifact graph. A runner-only build success, a
host-scoped dry run, static inspection without execution, or execution without final-archive
inspection is insufficient.

Harness companion packages have independent, exact package versions. The private SDK and wire
contract are bundled into each companion; they are not separately published. Selected companion
artifacts are verified available before any CLI/native release advertises them, including prereleases.
When a merged Version PR changes the prepared plan without changing the already-public CLI version,
its selected plugins are accepted and published independently of the native graph. This still uses
the prepared source, exact tarballs and integrity verification; an existing version is never replaced.

Integration preparation packs each selected companion once. Acceptance installs those exact
tarballs outside the workspace and loads the extension through the supported harness's native
package and resource loader under Node and Bun. Integration acceptance checks only that installation
succeeds and the extension loads without errors. Feature behavior belongs in integration tests.
Accepted bytes and their receipt are persisted;
publication does not repack them. Private workspace dependencies cannot escape into the packed
artifact. Shared SDK/wire changes trigger these checks as well as integration changes. Local
acceptance never publishes packages. Prereleases use the same preparation, acceptance and publication
checks as stable releases. Contract changes advance the RPC allocation in every channel, and each
CLI pins the selected plugin versions from its own release channel.
