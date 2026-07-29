---
applies_to:
  - .github/workflows/changesets.yml
  - .github/workflows/release.yml
  - .github/workflows/release-checks.yml
  - .github/workflows/direct-publish-npm.yml
  - packages/release/**
  - packages/cli/bin/magnitude.js
  - packages/cli/lib/**
  - packages/cli/scripts/prepare-release-runtime.ts
  - packages/cli/package.json
  - packages/sdk/src/binary.ts
  - inference/scripts/compile.ts
---

# Release distribution

Changesets is the sole authority for the CLI version, npm publication, and npm dist-tag. It
maintains one version PR. Merging that PR automatically releases its exact merge commit. Ordinary
pushes and manual dispatch do not start a normal release.

The separate direct-npm workflow exists only to complete a release whose GitHub assets are already
public while npm is absent. It checks out the exact release tag and invokes the same Changesets
publisher. It never rebuilds or replaces native assets.

## Artifacts

Magnitude publishes CLI, ACN, and ICN-base archives for Apple arm64 and x64, Linux GNU arm64 and
x64, and Windows MSVC x64.

CLI and ACN archives contain one executable under `bin/`. ICN bases contain the executable, common
runtime libraries, CPU modules, the release catalog lock, and the model-planner-input bundle.

Metal, CUDA, and Vulkan are separate backend packs. Each pack contains one backend module family
and only its required redistributable runtime libraries. The supported hosts and packs are fixed
product configuration, not a plugin system.

One manifest records every native archive's release identity, filename, size, SHA-256, native-build
identity, backend ABI, and compatibility facts used at runtime. Production acquisition trusts the
GitHub release origin and its HTTPS transport for the manifest. Artifact sizes and hashes prove that
downloaded bytes match that manifest; they do not provide an independent publishing identity.

## Build and validation

Catalog hydration is an explicit operation performed once per release. Every host receives the
same lock and planner bundle. Ordinary Cargo builds perform no model-data network access.

Each host job builds and extracts its final CLI, ACN, and ICN-base archives. It verifies exact
versions, embedded ripgrep, native installation loading, server readiness, health, and an
authenticated ICN endpoint.

CUDA and Vulkan jobs prove that the configured module and redistributable files were produced and
that their native-build and backend ABI identities match their host base. Jobs without matching GPU
hardware do not claim device execution.

Pull requests run one representative Linux x64 release build when release-relevant code changes.
An actual release builds the complete artifact graph.

## Runtime ownership

The npm package contains a small Node entry and a bundled Effect acquisition runtime. It acquires
CLI only. SDK acquires ACN. The ICN lifecycle acquires ICN artifacts and constructs an exact local
installation. These responsibilities do not overlap.

All remote native bytes are selected from the release manifest and installed under their digest.
Downloads and extraction are bounded, and every artifact must match its declared size and SHA-256.
Archives accept regular files at fixed safe paths only. Corrupt or incomplete installations are
replaced as complete units.

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

Publication creates or resumes the exact private GitHub draft, uploads and verifies the complete
candidate, then makes GitHub public. The accepted npm pack must acquire and execute the CLI from
that public release before Changesets publishes npm, because the npm launcher depends on the
GitHub assets. Registry integrity is checked against the accepted npm pack. A lost npm response is
treated as success only when the registry exposes that exact integrity.

An interrupted private draft is discovered through authenticated release listing and can be
retried. Public assets are immutable. The only supported public partial state is
GitHub-public/npm-absent, handled by direct npm publication from the exact tag after the same
public CLI acquisition check.

## Required guarantees

- Unrelated commits cannot publish.
- Changesets alone controls npm version and dist-tag behavior.
- Every acquired native artifact matches the size and SHA-256 in its release manifest.
- Final host archives execute before publication.
- Accelerator operational failure never becomes CPU fallback.
- Valid cached installations work offline; unavailable repair fails explicitly.
