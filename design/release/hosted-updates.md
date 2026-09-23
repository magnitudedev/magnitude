---
applies_to:
  - packages/release/src/hosted-update/**
  - packages/release/resources/distribution/**
  - packages/release/scripts/publish-hosted.ts
  - desktop/src/*update*
  - packages/daemon-management/src/desktop-native/*update*
  - packages/daemon-management/src/application-update/**
  - packages/daemon-management/native/mac-update-*
  - cli/src/startup/*update*
  - packages/client-common/src/desktop/update.ts
  - packages/sdk/src/desktop-update.ts
---

# Application updates

Magnitude checks for compatible application updates and supports automatic downloads and manual
checks. Shared application update services own preparation, scheduling, installation identity and
verified transfer independently of Electron. The admitted application owner supplies filesystem
capabilities and retains the update workers for its lifetime. Desktop composes native installation
and relaunch; preparation itself never restarts the application or its service. Installation intent
separates authorization permission from continuation: Desktop retains its visibility choice, while
Caller leaves continuation to the invoking foreground or finite command. Helpers must never launch
Desktop for Caller intent. Handoff admission is distinct from completed installation; the caller
must verify replacement before continuing startup.

One schedule belongs to each owner. Manual checks reset its deadline; resume wakes the same timer.
Automatic-download preference changes are persisted before they affect transfers. Disabling automatic
downloads cancels an automatic transfer and waits for scoped scratch cleanup before another transfer
can begin. A retained prepared update survives owner exit; a recorded failed attempt remains failed
until explicit retry or discard. Observation never initiates a check or transfer.

Updates must match the application platform and release channel. The client verifies release
signatures, downloaded file integrity, and applicable native publisher signatures before
installation. Failed checks or verification must not be reported as successful updates.

macOS staged-bundle verification uses the publisher identity compiled into the installed application;
missing publisher configuration cannot become identifier-only update trust. Native verification checks
sealed resources, nested code, every architecture slice, the required executable architecture, and the
sealed application identifier and release version. The caller retains exclusive staging ownership
through verification and publication. A successful signature check is not installation admission,
archive containment validation or notarization acceptance; those remain separate transaction gates.

Application binaries are distributed through GitHub Releases. Downloads must resolve to trusted
release assets, and interrupted or invalid transfers must not publish a prepared installer.

Transfer scratch storage is separate from the private prepared-update directory. Under exclusive
application or installation admission, Windows startup may retire an older cache with the known
inherited current-user/administrator/system ACL and recognized cache entries. Recovery retains the
directory identity through rename, preserves its contents separately, and creates a fresh private
cache. It never adopts old bytes as a prepared update or rewrites generic private-directory ACLs.
Unknown contents, reparse points, ownership, or permissions require explicit repair. Interrupted
recovery may leave a preserved old cache, but cannot authorize an installation.

Update acceptance covers download, verification, installation, relaunch, and preservation of
application state.
