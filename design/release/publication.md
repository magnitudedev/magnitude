---
applies_to:
  - .changeset/**
  - .github/workflows/changesets.yml
  - .github/workflows/release.yml
  - .github/workflows/register-release.yml
  - packages/release/**
  - packages/version/**
---

# Release preparation and publication

Magnitude ships native desktop installers and their bundled command-line client. Neither the
CLI launcher nor the Pi extension is published to npm. Both package manifests are private.
Changesets versions the private CLI identity package; private-package versioning is enabled,
while other workspace packages, including Pi, are ignored. This keeps desktop versions independent
of any npm registry state. Existing npm releases are not modified or unpublished.

The release package owns contract fingerprinting, public-baseline selection, Changesets
orchestration, private revision allocation, and acceptance. Its checked-in release-plan.json is
the committed record of desktop/CLI version, daemon coordination revision, and RPC allocation.
The plugin selection is empty. Pi's source remains available for local development, but it is
not built or packed by release preparation, installed by normal connections, or a publication gate.

## Identity and preparation

Merging the Changesets version PR selects the versioned source commit. Allocation advances the
private daemon coordination revision once whenever the CLI version changes, prereleases included.
The same CLI version retains its revision. Version generation consumes the plan as data and does
not import release implementation or RPC contracts. Generated identity sources are not committed.

Automatic publication handles the closed, merged Changesets PR in the default-branch context,
so signing environments evaluate the trusted branch. It selects the PR's actual merge commit,
requires the Changesets branch in this repository, and verifies main ancestry before building;
it never checks out an unmerged PR head.

The RPC fingerprint covers procedure identity, encoded payload/success/error schemas, stream
status, replay policy, public health, and transport framing. Irrelevant field order, brands,
class identity, descriptions, and formatting do not affect it. Unsupported constructs fail
preparation. Semantic-break markers cover behavior that structural fingerprinting cannot detect.

The RPC version is monotonic over successive plans: unchanged fingerprint and no semantic-break
marker retain it; a change advances it once. Markers are consumed during allocation. Detect mode
observes without advancing the daemon revision and generates a CLI changeset for an undeclared
RPC change. Verify mode checks the source contract, generated identity, and pinned public baseline.
An allocated stable release awaiting publication prevents another automatic allocation.

The baseline is the public GitHub release manifest. Preparation never queries npm, builds Pi,
allocates Pi versions, or waits for an unpublished npm package.

## Publication

1. Preflight validates the Changesets-selected source/version and exact GitHub state.
2. Build the native artifact graph and validate the installed desktop and bundled CLI.
3. Recheck the source contract and public baseline before publication.
4. Verify the complete Apple Developer ID/notarization receipt graph against the candidate bytes.
5. Upload the accepted native graph and manifest to the exact GitHub release draft, verify its
   asset metadata/digests, and make the release public.
6. Register verified releases for application updates.

Candidate acceptance invokes the installed bundled CLI without npm. It checks that the CLI's
version matches the desktop release even after the candidate artifact endpoint is stopped.
Linux lifecycle acceptance uses the package-owned /usr/bin/magnitude command.

## Recovery

Private GitHub drafts are retryable. Public assets are immutable. An ambiguous publication is
resolved from GitHub state. Draft upload recovery uses the paginated release-asset endpoint,
including incomplete assets omitted from the release object. It preserves uploads only when
their completed state, size, and digest match the accepted candidate, removes incomplete or
mismatched entries, and retries individual network/server failures with bounded backoff.
Each retry reconciles remote state first, including when an upload succeeded but its response
was lost. Failures retain the HTTP status, GitHub request identity, and bounded response details.
Baseline changes require refreshing preparation, not renumbering
already-built candidates. A missing release or exact resumable draft selects the native release
path; an exact public release selects hosted metadata registration. Orphan tags and mismatched
source commits are rejected.

The registration recovery workflow checks out the public tag, verifies the existing release,
and registers hosted metadata without rebuilding or republishing native assets. Neither normal
publication nor recovery requires npm publishing credentials.
