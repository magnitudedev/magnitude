---
applies_to:
  - packages/release/src/hosted-update/**
  - packages/release/resources/distribution/**
  - packages/release/scripts/publish-hosted.ts
  - desktop/src/*update*
  - packages/daemon-management/src/desktop-native/*update*
  - cli/src/startup/*update*
  - packages/client-common/src/desktop/update.ts
  - packages/sdk/src/desktop-update.ts
---

# Application updates

Magnitude checks for compatible application updates and supports automatic downloads and manual
checks. The desktop application owns download, installation, and relaunch.

Updates must match the application platform and release channel. The client verifies release
signatures, downloaded file integrity, and applicable native publisher signatures before
installation. Failed checks or verification must not be reported as successful updates.

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
