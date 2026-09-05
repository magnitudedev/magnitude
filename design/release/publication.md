---
applies_to:
  - .github/workflows/changesets.yml
  - .github/workflows/release.yml
  - .github/workflows/publish-npm.yml
  - .github/workflows/integrations.yml
  - packages/release/**
  - packages/version/**
  - scripts/*integrations.ts
  - integrations/pi/scripts/**
---

# Release preparation and publication

Changesets assigns package versions. Release preparation derives required changes from the public
contract and shipped plugin contents before Changesets discovers pending changes. Publication
consumes accepted tarballs; generic workspace publication is not an alternate release path.

The release package owns the entire preparation pipeline: contract fingerprinting, public-baseline
selection, Changesets orchestration, private revision allocation, plugin allocation, and acceptance.
Its checked-in `release-plan.json` is the only committed record of release identity: CLI version,
daemon coordination revision, RPC allocation, and plugin selection. RPC schemas are hashed in
memory; no schema snapshot or secondary generated plugin-selection file is persisted. CLI
installation reads its exact selected plugin directly from the plan. Semantic markers live under
`packages/release/rpc-breaks/`. The version package generates every identity constant, the RPC
version included, from the CLI package version and the plan; those generated sources are not
committed. It reads the plan as data and does not import release or the RPC contract.

## Source and version identity

Merging the Changesets version PR selects the versioned source commit. The same version operation
advances the private daemon coordination revision in the plan by one whenever the CLI version
differs from the previous plan's, prereleases included; the same CLI version keeps its revision.
That revision orders processes and is independent of the public application RPC version.

The RPC fingerprint is generated from authoritative RPC declarations: procedure identity, encoded
payload/success/error schemas, stream status, replay policy, public health, and transport framing.
Irrelevant field order, brands, class identity, descriptions, and formatting do not affect it.
Unsupported wire constructs fail preparation. Opaque predicates and transform behavior require a
semantic-break marker when their externally observable behavior changes; generation does not claim
to detect arbitrary function semantics.

The RPC version is a monotonic counter over successive plans, in every channel:

- equal fingerprint and no semantic-break marker: retain the previous plan's version;
- changed fingerprint or a marker: the previous plan's version plus one.

Markers are consumed by allocation, so a recorded break never re-applies. A number never names two
contracts: a reverted contract still takes a fresh number. Several pending changes produce one
increment, and repeated preparation against the same source is idempotent. Prepared source records the baseline, fingerprint, semantic breaks, plugin
artifacts, and exact CLI selection. Preparation always reads the real public baselines, GitHub
releases and npm, so pull-request checks exercise the same allocation without committing it. A
merged plan awaiting publication suppresses further automatic allocations until it is published or
explicitly refreshed; a plugin allocated for publication that npm does not yet serve suppresses
them the same way.

Changesets prerelease mode prepares exactly like stable. The RPC version advances on every contract
change, alpha to alpha included. Plugins bump to prerelease versions, publish under the series'
dist-tag so `latest` stays stable, and the prerelease CLI pins them exactly. The plugin baseline is
the version published under that dist-tag, falling back to `latest`. A plan publishes plugins only
in its own channel: prerelease CLIs publish prerelease plugin versions, stable CLIs stable ones.
`pre exit` lets Changesets fold the series into stable versions and the stable path runs. The
complete pipeline, plugin installation included, is therefore testable in a prerelease series.

## Private SDK and plugin artifacts

The SDK remains private. Pi bundles its reachable workspace code into an ESM distribution and
declares only public Effect/platform dependencies plus host peers. Neither workspace imports nor
private package runtime dependencies may escape the tarball.

A plugin-content fingerprint covers shipped files and install-relevant package fields, including
dependencies and the embedded RPC version. It excludes its own generated metadata, assigned package
version, development tooling, and release notes. SDK-only shipped changes therefore require a
plugin release; unrelated source/test changes do not.

A plugin's baseline is the version npm serves as `latest`, described by the content manifest inside
its tarball. The CLI release manifest is the baseline for the RPC allocation only. A plugin-only
publication therefore becomes its own baseline without a CLI release.

Preparation generates a changeset only for what changed and is not already declared by a human
changeset: the plugin when its content fingerprint differs from npm latest, and the CLI when the RPC
allocation differs from the CLI baseline. A plugin-only change produces a plugin-only Version PR.
Changed plugins receive at least a patch, respecting a larger human Changesets bump. Unchanged
plugins retain the existing published version and artifact bytes. Registry versions are immutable:
an orphan publication with different bytes requires a new version during preparation.

Each CLI embeds exact plugin name, version, RPC version, content fingerprint, and tarball integrity.
That selection is what a fresh installation receives. Compatibility of an installed package is
decided only by its verified content metadata and an RPC version equal to the CLI's; a newer or
older plugin built against the same RPC version is accepted without replacement. User-owned
package/configuration state retains the connection system's transactional preservation rules.
Plan validation requires exactly one selection for every host declared by the plugin-host schema;
it does not identify required hosts through hardcoded package names.

## Publication order

Release vocabulary distinguishes four concrete records: `PluginPackageManifest` describes npm
installation fields; `PluginContentManifest` describes the bundled files and content fingerprint;
`PluginArtifact` identifies the exact selected tarball and its integrity; `PluginAcceptanceReceipt`
binds successful runtime checks to those bytes and the prepared plan. A content fingerprint decides
whether a release is needed; tarball integrity proves which bytes were accepted and published.

1. Preflight selects source/version and rejects conflicting public state. A Version PR that left
   the CLI version unchanged is accepted only when it changed the release plan; it then publishes
   plugins alone.
2. Verify the source contract, pinned public baseline, and bundled plugin fingerprints.
3. Build native artifacts and prepare each selected plugin tarball once; unchanged plugins reuse
   their public tarballs.
4. Accept the exact plugin artifacts through isolated Node and Bun Pi installations. Record the
   acceptance against their integrity and release-plan fingerprint.
5. Recheck the public baseline and source at the commit point.
6. Publish or verify selected plugins first, consuming the accepted tarballs without repacking.
7. Publish the accepted native graph and manifest as the exact GitHub release.
8. The accepted CLI npm tarball acquires and executes that public release, then is published
   directly. Registry integrity must equal the accepted tarball integrity.

The public manifest records RPC and plugin artifact metadata beside the native graph. Ordinary
pushes do not publish. When a merged Version PR leaves the CLI version already public, the release
workflow runs plugin preparation, acceptance, and publication on their own from the same prepared
source; unchanged plugins verify as no-ops. There is no separate manual plugin workflow, and no
path blesses an arbitrary main commit.

## Recovery

Private GitHub drafts are retryable. Public assets and npm versions are immutable. An ambiguous
publication succeeds only if the registry exposes the exact expected integrity. Baseline changes
before publication require refreshing preparation, never renumbering already-built candidates.

GitHub-public/CLI-npm-absent recovery checks out the public tag, verifies the existing release,
repeats acquisition acceptance, and publishes the accepted CLI tarball without rebuilding native
assets. A plugin published before an interrupted main release is either reused byte-for-byte or
superseded with a fresh plugin version during preparation.
