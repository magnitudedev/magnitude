---
applies_to:
  - .github/workflows/changesets.yml
  - .github/workflows/release.yml
  - .github/workflows/publish-npm.yml
  - packages/release/scripts/preflight.ts
  - packages/release/scripts/github-release.ts
  - packages/release/scripts/publish.ts
  - packages/release/scripts/prepare-npm.ts
  - packages/release/scripts/verify-npm.ts
  - packages/release/scripts/verify-public-release.ts
  - packages/version/**
---

# Release publication

Changesets is the sole authority for the CLI version, npm publication, and npm dist-tag.

## Source identity

- Merging the Changesets version PR selects the versioned source commit.
- The same version operation advances the checked-in ACN release revision allocation exactly once.
  Re-running without a CLI version change does not advance it.
- Ordinary pushes do not publish.
- Before publication, an operator may explicitly select a later recovery commit on `main` only when
  it descends from the version commit and retains the exact version.
- The selected commit is the source identity for every build, package, manifest, tag, and asset.

## Publication order

1. Preflight pins source and version, verifies credentials, and rejects conflicting GitHub or npm
   state.
2. The complete release graph is built and accepted.
3. Preflight repeats immediately before the commit point.
4. Publication creates or resumes the exact private GitHub draft, uploads the complete candidate,
   verifies it, and makes the release public.
5. The accepted npm package acquires and executes CLI from the public GitHub release.
6. Changesets publishes npm, and registry integrity must equal the accepted package integrity.

## Recovery

Private drafts are retryable. Public assets are immutable. The only supported partial public state
is GitHub-public/npm-absent; the npm-only workflow checks out the exact public tag, verifies the
existing release, repeats public CLI acquisition, and publishes the already-accepted npm package.
It never rebuilds or replaces native assets.

An exact GitHub/npm publication is a successful no-op. An ambiguous npm response counts as success
only when the registry exposes the expected version and exact integrity.
