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

Production application binaries are distributed through GitHub Releases. Downloads must resolve
to trusted release assets. Private acceptance builds may instead bind one explicit HTTPS artifact
origin at build time, separate from production service origins. Artifact responses cannot introduce
or change that policy. Private transfers reject redirects and omit installation credentials and
cookies; publisher signatures and file integrity remain required. Runtime environment variables
cannot enable private delivery in a production build. Interrupted or invalid transfers must not
publish a prepared installer.

Update acceptance covers download, verification, installation, relaunch, and preservation of
application state.
