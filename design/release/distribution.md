---
applies_to:
  - packages/release/src/*.ts
  - packages/release/scripts/assemble.ts
  - packages/release/scripts/prepare-installation-distribution.ts
  - packages/release/scripts/build/**
  - packages/release/native/**
  - packages/release/resources/windows/desktop.nsi
  - packages/release/resources/install.*
  - packages/release/scripts/apple/desktop.ts
  - packages/launcher/package.json
  - packages/daemon-management/src/desktop-native/mac-cli-*.ts
---

# Release distribution

Magnitude distributes one versioned release as a fixed graph of native desktop, CLI, and engine
artifacts. The release graph is product configuration, not a plugin system.

## Published artifacts

| Artifact | Published for | Contents |
| --- | --- | --- |
| CLI | every host | one `bin/magnitude-cli` executable |
| Desktop | supported graphical hosts | Electron application with its matched service and native ownership addon and, on Unix, transient command helper; macOS uses an explicit DMG installation |
| ACN | Apple hosts | signed, notarized, stapled `Magnitude.app` whose main executable is `magnitude-service` with embedded ripgrep, plus metadata and icon |
| ACN | other hosts | one `bin/magnitude-service` executable with embedded ripgrep |
| Inference | every host | one complete installation layout: `bin/magnitude-inference` with every backend of the host compiled in, `runtime/` (NVRTC on Linux and Windows; the Microsoft CRT on Windows), and `catalog/` planner inputs |

Release hosts are Apple arm64, Apple x64, Linux GNU arm64, Linux GNU x64, and Windows x64 MSVC.
Windows also ships a per-user desktop installer.

Each host has exactly one inference artifact and there are no backend packs. Apple arm64 includes
Metal and CPU; Apple x64 is CPU-only; Linux and Windows include CUDA, Vulkan and CPU. NVRTC 12.9 is
the only CUDA payload, adding about 40–45 MB compressed to every Linux and Windows artifact.
Backend floors are defined by [the inference platform contract](../../inference/docs/compatibility.md);
the device a service uses is chosen at runtime by Seismic discovery, never by artifact selection.

## Release identity

The release manifest identifies one version, source commit, ACN coordination revision, and the
complete native artifact graph. Each artifact record contains its host, kind, filename, byte size,
and SHA-256. Inference records also contain their native-build identity: the engine build, which a
declared installation and its executable must match.

The manifest does not describe build provenance or duplicate platform policy. Platform support is
a property of the release target and is enforced while building and accepting the candidate.

The desktop bundle owns the window, tray, and service lifecycle. Inference artifacts and models
remain outside the app. Its installer preserves the sealed native bundle, including framework
symlinks; runtime archive extraction never installs or interprets a desktop artifact. Signing and
notarization precede final installer checksums. Ad-hoc local builds never imply publisher trust.
Initial installation supports direct platform downloads and full-application shell installers.
The macOS shell installer verifies the downloaded bootstrap before invoking its finite bundled CLI
installer outside the destination. That entry verifies signed release metadata and bundle contents,
uses shared installation admission and transactions, and registers the command through shared host
code. Installation never starts Desktop or a headless server. The macOS DMG remains available;
both Mac architectures open a compact, styled installer window with explicit drag-to-install
instructions, the app on the left, a directional arrow, and a working Applications shortcut on the right.
On macOS and Linux, opening the new desktop automatically retires verified previous standalone services and their
startup registrations before starting the bundled service. No command or confirmation is required.
User models, caches and settings remain outside the installed bundle and are preserved in place.
Each Apple host also produces an update ZIP from the same signed and stapled desktop bundle as
its DMG. The ZIP is a separate desktop artifact covered by the release manifest and acceptance
receipts. It is not an inference/runtime acquisition archive. Producing it does not establish
successful application replacement or relaunch; those remain updater acceptance requirements.

Linux hosts publish DEB and RPM packages; the x86-64 host also publishes a pacman package for Arch
Linux and its derivatives, which have no official ARM distribution. DEB/RPM metadata uses `~` for
prerelease ordering, and their published filenames retain the SemVer `-` separator because GitHub
rewrites `~` in asset names. Renaming the packaged file does not change its bytes, internal version,
checksum, or installation behavior. Pacman's version comparison sorts `~` after the release it
precedes, so pacman versions drop the prerelease separator instead (`1.2.0beta.1` sorts before
`1.2.0`), and that version is also the pacman filename.

Linux desktop packages use the name `magnitude-desktop` and place the matched application at
`/usr/lib/magnitude-desktop/magnitude`. The application-menu launcher and headless CLI resolve
that same installation through `/usr/bin/magnitude-desktop`, including login startup. This guarded
entry acquires shared installation admission before executing Electron and retains it until exit.
The package manager obtains exclusive admission before replacement/removal and rejects while any
participating user app remains alive. A root-owned installation gate spans the separate maintainer
script lifetimes; launches fail with repair guidance until configuration succeeds. Interrupted
installation never becomes an independently running service. The lock inode survives reinstall.
Pacman ignores package scriptlet failures, so the pacman package performs its exclusive admission in a
PreTransaction hook with AbortOnFail, named to order after other packages' hooks, and completes it in
its post-install, post-upgrade and post-remove scriptlets. Removal deletes the hook's own files before
post-transaction hooks run, which is why completion is a scriptlet. A first installation has no hook
on disk yet; its post-install scriptlet establishes the lock. A failed pacman transaction after
admission leaves the repair gate in place until the package is reinstalled.
Package abort hooks must preserve a healthy old installation after a rejected upgrade. Debian's
`postinst abort-upgrade` and `abort-remove` release the gate after the package manager restores the
old installation; a Debian refusal before gate acquisition does not acquire or clear another owner's gate.
RPM removal refusal retains the repair gate because DNF can still remove automatic dependencies
from the failed transaction. The existing owner continues running, but subsequent startup requires
package reinstallation to restore dependencies and clear the gate.
`/usr/bin/magnitude-desktop` is the graphical launch entry; `/usr/bin/magnitude` resolves the
bundled headless CLI. Login registration remains a user preference controlled
by the running application. Package installation does not register an independent daemon or
automatically open a window. Native DEB/RPM/pacman consumption and upgrade acceptance precede
inclusion in the published artifact graph.

Windows installer candidates use the desktop's existing application lease and never start or adopt
an independent service. The PowerShell bootstrap authenticates a publisher-signed standalone CLI
before using its embedded publisher key to verify the installer release and bytes. It separately
authenticates the installer, waits for silent setup, and refreshes command PATH in the invoking shell.
Both executable signatures require the configured publisher and a timestamp. Fresh installation publishes a complete staged payload by same-volume rename.
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
is never extraction scratch. Uninstall retires that previous tree against its inventory before
changing the current payload, PATH, startup preference or registration. Unknown or mapped previous
files defer removal while the exact uninstaller and recovery registration remain available. Retrying
removal after the obstruction is gone must leave no previous payload that could block a fresh install.
Upgrades preserve startup and shortcut preferences. No pre-cutover
installation compatibility is implied, and automatic delivery remains gated on native acceptance.

## Distribution contract

A conforming release satisfies all of the following:

- Every published artifact is present exactly once and matches its manifest size and SHA-256.
- Every executable and library depends only on artifact-owned files, its host platform contract,
  and capability dependencies loaded at runtime.
- Each host's inference artifact is one complete installation with every backend of that host.
- Final artifacts pass build-host-independent validation before publication.
- GitHub assets are public and verified before hosted update metadata promotes them. No npm packages are published.

The concrete host dependency contracts are defined in
[Platform contracts](./platform-contracts.md). Build acceptance is defined in
[Build and validation](./build-and-validation.md). Runtime installation is defined in
[Acquisition](./acquisition.md), desktop-owned CLI updates are defined in
[CLI updates](./client-updates.md), and remote publication is defined in
[Publication](./publication.md).

Installer distribution preparation consumes authenticated publisher records and emits both scripts
and per-channel target offers into a fresh static hosting directory. It verifies the complete input
batch before writing and rejects duplicate targets or mixed release versions. Preparing these files
does not deploy them or promote a channel. Linux bootstrap requires curl, Python 3 and OpenSSL with
Ed25519 support; Windows bootstrap uses the publisher-signed CLI from the selected release.

## Desktop-owned command registration

The installed desktop exposes its bundled CLI. macOS silently creates the user-owned
`~/.magnitude/bin/magnitude` link on installed-app launch and prepends that directory using marked
shell configuration entries. Shared host code owns command placement and shell configuration for
both Desktop and installation, including removal of only unchanged managed entries. It never
requests administrator authorization. Windows registers a
private native launcher outside the replaceable application tree in the current user's PATH. The
launcher resolves the matched bundled CLI on each invocation and retains foreground command ownership
across a startup update. Launcher publication preserves mapped prior images under distinct retired
names; a later installation removes them after their commands exit. Retrying setup repairs an
interrupted publication. Uninstall defers while retired images are mapped or the command directory
contains unrelated files. Linux retains its package-owned
`/usr/bin/magnitude` link. App replacement keeps the command pointed at the matching bundled
version. No npm launcher is required.

Registration replaces writable existing `magnitude` commands on PATH so the bundled CLI takes
precedence. Protected macOS commands remain untouched and are shadowed by the user PATH entry.
On Windows prior command shims are removed after the new payload and PATH registration are
installed. Other command names and directories are untouched. Removal deletes only exact links
targeting this installation and unchanged managed shell blocks, or a Windows PATH entry recorded
as added by this installer. Other user configuration is preserved. New terminals pick up PATH
changes; existing terminals may retain their old environment. macOS Finder deletion has no uninstall
callback, so the app provides explicit command-link removal. Successful registration is silent.
