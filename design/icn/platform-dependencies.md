---
applies_to:
  - inference/**
  - packages/release/**
  - packages/icn/**
  - packages/sdk/src/binary.ts
  - .github/workflows/release-build.yml
  - .github/workflows/release-checks.yml
---

# ICN platform dependencies

ICN releases have zero implicit runtime dependencies. A published artifact must run on every host
that satisfies its declared platform contract without relying on undeclared packages, files,
environment variables, search paths, toolchains, SDKs, or properties of its build machine.

Native software cannot have zero operating-system dependencies. Kernel and loader ABIs, system
frameworks, and hardware drivers are permitted only as explicit parts of a target contract.

## Dependency ownership

Every dependency reachable from an executable, library, or backend module belongs to exactly one
class:

- **Artifact-owned:** distributed with the artifact, addressed through relative loader paths,
  integrity-covered, and signed where required.
- **Platform ABI:** guaranteed by a declared minimum OS and ABI baseline.
- **Optional capability:** required by an explicitly selected hardware backend, such as Metal,
  Vulkan, or CUDA, and verified before selection.

Anything unclassified is a release defect. Package-manager prefixes, build directories, developer
SDKs, and ambient library search paths are never platform contracts.

Each target contract declares its OS, architecture, CPU baseline, minimum OS and userspace ABI,
permitted system dependencies, artifact-owned closure, and optional capabilities. Distinct
incompatible contracts—such as GNU and musl Linux—are distinct release targets.

## Build and packaging

Native builds use pinned toolchains, SDKs or sysroots, dependency revisions, deployment targets, and
feature flags. Optional integrations are enabled from explicit product configuration, never because
a dependency happens to be installed on the runner. Builds target the oldest platform version the
artifact claims to support.

Release assembly uses declared contents and validates the recursive dependency closure of the final
base-plus-backend installation. Mach-O load commands and deployment targets, ELF interpreters,
needed libraries and symbol-version floors, and PE imports are checked as applicable. Every edge
must resolve to an artifact-owned file or an allowed platform or capability dependency.

Third-party libraries use platform-relative loading (`@rpath` or `@loader_path`, `$ORIGIN`, or
application-local DLL resolution). Final integrity measurement and signing happen after composition
and path rewriting. Vendor drivers that cannot be redistributed remain explicit capability
dependencies rather than becoming artifact dependencies.

ICN executables resolve owned libraries from the sibling `runtime/` directory; backend and runtime
libraries resolve from their own directory and sibling `runtime/`. Linux uses `$ORIGIN`, macOS uses
`@loader_path`, and Windows adds the installation's `runtime/` to the child process `PATH`. Linux and
macOS release execution must not require `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`. Release builds
inspect these embedded paths before archiving, and launchers clear inherited Unix library paths so
ambient toolkit or package-manager libraries cannot override the installation closure.

## Release validation

The extracted final installation is tested independently of its build host on the minimum supported
platform, without package managers, developer tools, or undeclared libraries. Validation proves
binary identity, backend loading, process readiness, and a minimal inference path. Accelerator
operation is additionally proven on representative hardware.

Publication records the target contract, build inputs, dependency graph, bundled-library provenance,
capability requirements, and validation result. It fails for any unresolved or unclassified edge,
build-host path, or deployment and ABI requirement above the declared baseline.

Execution and dynamic-loader failures remain distinct from protocol decoding failures. Missing
artifact dependencies are release defects; unsupported platform contracts make an artifact
ineligible; missing optional capabilities make a backend ineligible before selection.

## Required guarantees

- Released artifacts have no unclassified runtime dependencies or build-host-relative paths.
- Build-host contents cannot enable features or raise the supported platform baseline.
- Every non-platform runtime dependency is bundled and integrity-covered when redistribution permits;
  otherwise it must be an explicit capability dependency.
- Every final base-plus-backend installation passes closure validation and clean minimum-platform
  execution before publication.
