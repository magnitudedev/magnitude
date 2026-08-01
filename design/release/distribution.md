---
applies_to:
  - .github/workflows/changesets.yml
  - .github/workflows/release.yml
  - .github/workflows/release-build.yml
  - .github/workflows/release-checks.yml
  - .github/workflows/publish-npm.yml
  - packages/release/**
  - packages/icn-protocol/**
  - packages/cli/bin/magnitude.js
  - packages/cli/lib/**
  - packages/cli/scripts/prepare-release-runtime.ts
  - packages/cli/package.json
  - packages/sdk/src/binary.ts
  - inference/scripts/compile.ts
  - scripts/accept-release-candidate.ts
---

# Release distribution

Changesets is the sole authority for the CLI version, npm publication, and npm dist-tag. It
maintains one version PR. Merging that PR automatically releases its exact versioned source commit.
Ordinary pushes do not start a normal release. A parameterless manual dispatch resolves the latest
merged Changesets PR for the current version and reconciles that same release. Before publication,
an operator may explicitly pin a later commit on `main` to recover that still-unpublished version.
The recovery commit must descend from the Changesets merge and retain its exact version; it becomes
the source identity for every build, package, manifest, and publication step.

The separate npm workflow completes a release whose GitHub assets are already public while npm is
absent. It checks out the exact release tag and invokes the same Changesets publisher. It never
rebuilds or replaces native assets.

## Artifacts

Magnitude publishes CLI, ACN, and ICN-base archives for Apple arm64 and x64 and Linux GNU arm64
and x64. Windows release artifacts are intentionally disabled until the Windows product path and
release builds are reliable.

CLI and ACN archives contain one executable under `bin/`. ICN bases contain the executable, common
runtime libraries, CPU modules, and the model-planner-input bundle.

Metal, CUDA, and Vulkan are separate backend packs. Each pack contains one backend module family
and only its required redistributable runtime libraries. The supported hosts and packs are fixed
product configuration, not a plugin system.

CUDA compatibility and backend-pack selection are governed by
[CUDA compatibility](../inference/cuda-compatibility.md). Linux x64 and ARM64 each publish CUDA
11.8 output containing ordinary `compute_80` PTX and CUDA 12.9 output containing the current
Ampere, Hopper, and Blackwell paths. An ordinary image may execute on newer GPUs when its PTX ISA
and target rules are satisfied; separate Ada and Hopper targets are not required merely for
coverage. Host architecture does not select a CUDA toolkit generation. The four host/toolkit packs
build independently and concurrently; additional PTX targets and native cubins
require measured startup-latency or runtime-performance justification.

One manifest records every native archive's release identity, filename, size, SHA-256, native-build
identity, backend ABI, and compatibility facts used at runtime. Production acquisition trusts the
GitHub release origin and its HTTPS transport for the manifest. Artifact sizes and hashes prove that
downloaded bytes match that manifest; they do not provide an independent publishing identity.

## Build and validation

Planner-input generation is an explicit operation performed once per release. Every host receives
the same bundle. Ordinary Cargo builds perform no model-data network access.

Each host job builds and extracts its final CLI, ACN, and ICN-base archives. It verifies exact
versions, embedded ripgrep, native installation loading, server readiness, health, and an
authenticated ICN endpoint. Binary identity, backend eligibility, installation, and readiness
records are validated with Effect Schemas generated from the canonical Rust bootstrap protocol;
generated drift fails release checks.

Before archiving, Linux and macOS jobs verify that the ICN executable, runtime libraries, and backend
modules contain only the installation-relative loader paths required by their final `bin/`,
`runtime/`, and `backends/` locations. Their extracted-archive smoke test runs without injected
library search paths and with inherited Unix library paths cleared.

CUDA and Vulkan jobs prove that the configured module and redistributable files were produced and
that their native-build and backend ABI identities match their host base. CUDA release assembly
also inspects the final fatbinary and derives its device-image compatibility facts; requested
compiler targets are not duplicated as a publication contract. Jobs without matching GPU hardware
do not claim device execution.

Pull requests build and validate the complete release artifact graph from the exact proposed commit
using the same reusable build workflow as production. They stop before creating or modifying a
GitHub release or publishing npm. Production consumes those same build jobs and adds the remote
publication steps only after the complete graph succeeds.

A manually dispatched Linux x64 dry run builds only the CPU host artifacts, packs npm, assembles a
host-scoped candidate through the production assembler, and invokes the production candidate gate.
It proves the critical packaged acquisition and isolated-planner path without building unrelated
hosts or accelerator packs. Host-scoped assembly is validation-only and never authorizes
publication; production assembly still requires the complete release graph.

## Runtime ownership

The npm package contains a small Node entry and a bundled Effect acquisition runtime. It acquires
CLI only. SDK acquires ACN. The ICN lifecycle acquires ICN artifacts and constructs an exact local
installation. These responsibilities do not overlap.

All remote native bytes are selected from the release manifest and installed under their digest.
Downloads and extraction are bounded, and every artifact must match its declared size and SHA-256.
Artifacts use up to four validated HTTP byte ranges with dynamically sized work units, independent
part retries, and ordered assembly. Work units keep available workers active without allowing one
retry to repeat more than a bounded maximum range. Range responses must describe the exact
requested bytes and one consistent representation; unsupported ranges fall back to the bounded
sequential transfer. Archives accept regular files at fixed safe paths only. Corrupt or incomplete
downloads and installations are never published and are replaced as complete units.

Apple arm64 always resolves its Metal pack. Other hosts select compatible CUDA, then compatible
Vulkan, then CPU only when successful probes show that no supported accelerator is usable.
Authentication, acquisition, probe, ABI, module-load, and device-registration failures fail
startup; none silently becomes CPU.

## Publication

Before expensive builds, preflight freezes the merge commit and Changesets-owned version, verifies
the npm token, and rejects conflicting GitHub or npm state. Immediately before publication, it
repeats the remote-state check.

Candidate validation packs npm from the released source and exercises its launcher through Node,
npx, Bun, and bunx against the local CLI candidate. Host jobs already validate native archives.
The same candidate gate uses the real SDK to acquire and start its packed ACN and ICN from an empty
data root, proves readiness through an ACN RPC, and requires packaged local-model recommendation
preparation to reach Ready through an isolated native planner. It terminates the exact owned
processes and repeats with the artifact endpoint unavailable to prove the resulting cache is
sufficient.

Publication creates or resumes the exact private GitHub draft, uploads and verifies the complete
candidate, then makes GitHub public. The accepted npm pack must acquire and execute the CLI from
that public release before Changesets publishes npm, because the npm launcher depends on the
GitHub assets. Registry integrity is checked against the accepted npm pack. A lost npm response is
treated as success only when the registry exposes that exact integrity.

An interrupted private draft is discovered through authenticated release listing and can be
retried. Public assets are immutable. The only supported public partial state is
GitHub-public/npm-absent, handled by the npm-only path from the exact tag after the same public CLI
acquisition check. Release reconciliation selects that path automatically, and an exact GitHub/npm
publication is a successful no-op.

## Required guarantees

- Ordinary unrelated pushes cannot start publication, and every published source commit is pinned
  before builds begin.
- Changesets alone controls npm version and dist-tag behavior.
- Every acquired native artifact matches the size and SHA-256 in its release manifest.
- Final host archives execute before publication.
- Accelerator operational failure never becomes CPU fallback.
- Valid cached installations work offline; unavailable repair fails explicitly.
