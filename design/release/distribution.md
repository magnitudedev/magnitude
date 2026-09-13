---
applies_to:
  - packages/release/src/contracts.ts
  - packages/release/src/executables.ts
  - packages/release/src/targets.ts
  - packages/release/scripts/assemble.ts
  - packages/release/scripts/build/**
  - packages/release/native/windows-installer.*
  - packages/release/resources/windows/desktop.nsi
  - packages/release/scripts/apple/desktop.ts
  - packages/launcher/package.json
---

# Release distribution

Magnitude distributes one versioned release as an npm package plus a fixed graph of native
artifacts. The release graph is product configuration, not a plugin system.

## Published artifacts

| Artifact | Published for | Contents |
| --- | --- | --- |
| CLI | every host | one `bin/magnitude-cli` executable |
| Desktop | supported graphical hosts | Electron application with its matched service and native ownership addon and, on Unix, transient command helper; macOS uses an explicit DMG installation |
| ACN | Apple hosts | signed, notarized, stapled `Magnitude.app` whose main executable is `magnitude-service` with embedded ripgrep, plus metadata and icon |
| ACN | other hosts | one `bin/magnitude-service` executable with embedded ripgrep |
| ICN base | every host | one `bin/magnitude-inference` executable, planner inputs, common runtime libraries, and CPU modules |
| ICN backend pack | compatible hosts | one Metal, CUDA, or Vulkan module family and its redistributable runtime libraries |

Published hosts are Apple arm64, Apple x64, Linux GNU arm64, and Linux GNU x64. Windows artifacts
are not published. Each backend pack names exactly one required ICN base and must have the same
native-build identity and backend-module ABI as that base.

Apple arm64 publishes Metal. Linux arm64 and x64 publish Vulkan plus CUDA 11.8 and CUDA 12.9.
CUDA device-image and driver compatibility is defined by
[CUDA compatibility](../inference/cuda-compatibility.md).

## Release identity

The release manifest identifies one version, source commit, ACN coordination revision, and the
complete native artifact graph. Each artifact record contains its host, kind, filename, byte size,
SHA-256, and the compatibility facts required for runtime selection. ICN records also contain their
native-build identity and backend-module ABI.

The manifest does not describe build provenance or duplicate platform policy. Platform support is
a property of the release target and is enforced while building and accepting the candidate.

The desktop bundle owns the window, tray, and service lifecycle. Inference artifacts and models
remain outside the app. Its installer preserves the sealed native bundle, including framework
symlinks; runtime archive extraction never installs or interprets a desktop artifact. Signing and
notarization precede final installer checksums. Ad-hoc local builds never imply publisher trust.
Initial desktop installation uses direct platform downloads, with a DMG on macOS and no curl/shell
installer. The transition from standalone daemon installations requires users to stop and disable
their previous installation before opening the new app. No old-service migration helper ships;
user models and settings remain outside the installed bundle.
Each Apple host also produces an update ZIP from the same signed and stapled desktop bundle as
its DMG. The ZIP is a separate desktop artifact covered by the release manifest and acceptance
receipts. It is not an inference/runtime acquisition archive. Producing it does not establish
successful application replacement or relaunch; those remain updater acceptance requirements.

Linux desktop packages use the name `magnitude-desktop` and place the matched application at
`/usr/lib/magnitude-desktop/magnitude`. The application-menu launcher and headless CLI resolve
that same installation through `/usr/bin/magnitude-desktop`, including login startup. This guarded
entry acquires shared installation admission before executing Electron and retains it until exit.
The package manager obtains exclusive admission before replacement/removal and rejects while any
participating user app remains alive. A root-owned installation gate spans the separate maintainer
script lifetimes; launches fail with repair guidance until configuration succeeds. Interrupted
installation never becomes an independently running service. The lock inode survives reinstall.
Package abort hooks must preserve a healthy old installation after a rejected upgrade.
`/usr/bin/magnitude-desktop` is the graphical launch entry; it does not
replace the npm CLI's `magnitude` command. Login registration remains a user preference controlled
by the running application. Package installation does not register an independent daemon or
automatically open a window. Native DEB/RPM consumption and upgrade acceptance precede inclusion
in the published artifact graph.

Windows installer candidates use the desktop's existing application lease and never start or adopt
an independent service. Fresh installation publishes a complete staged payload by same-volume rename.
A private installer-owned scratch container permits recovery after interrupted extraction; cleanup
is relative to retained handles and cannot follow directory redirections. Existing unsafe scratch
permissions are rejected without repair. The installed uninstaller is the sole removal record and
must match its executing self-copy before mutation. Payload removal rejects redirected paths,
preserves unrelated installed files, and retains the exact removal record until required cleanup
succeeds. Interrupted removal can be retried. Candidate packaging does not authorize replacement,
updates, signing claims, or publication before their separate acceptance gates pass.
Windows current-format replacement uses a versioned owned-file inventory generated from the
extraction payload. Native inspection requires a private installation root, exact version and
complete file/directory membership; unknown files, duplicate names, redirected paths and
hard-linked payloads fail inspection without mutation. Replacement retains the previous payload
until the registered version is durably committed. Rerunning setup restores the previous version
before that commit or finishes exact owned-file retirement afterward. The previous installation
is never extraction scratch. Upgrades preserve startup and shortcut preferences. No pre-cutover
installation compatibility is implied, and automatic delivery remains gated on native acceptance.

## Distribution contract

A conforming release satisfies all of the following:

- Every published artifact is present exactly once and matches its manifest size and SHA-256.
- Every executable and library depends only on artifact-owned files, its host platform contract,
  and the capability dependencies of the selected backend.
- A backend pack composes with exactly its required base and cannot alter the base platform floor.
- Final artifacts pass build-host-independent validation before publication.
- GitHub assets are public before npm is published because the npm launcher acquires those assets.

The concrete host dependency contracts are defined in
[Platform contracts](./platform-contracts.md). Build acceptance is defined in
[Build and validation](./build-and-validation.md). Runtime installation is defined in
[Acquisition](./acquisition.md), native CLI package-manager updates are defined in
[CLI updates](./client-updates.md), and remote publication is defined in
[Publication](./publication.md).
