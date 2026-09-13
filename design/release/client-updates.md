---
applies_to:
  - packages/release/src/client-update/**
  - packages/launcher/src/**
  - packages/launcher/scripts/build-launcher.ts
  - cli/src/index.ts
  - cli/src/commands/update.ts
  - cli/src/commands/update-runtime.ts
  - cli/src/update/**
  - cli/src/runtime/**
---

# Client release updates

The installed desktop application owns updates for itself, its bundled service and headless CLI.
Release discovery uses the Magnitude-hosted signed protocol. npm dist-tags and a caller's package
manager are not application update authorities. Harness packages retain their independent npm
publication and installation contracts.

## Headless control

`magnitude update` and `magnitude update check` request a fresh check from the desktop owner.
`magnitude update status` observes current update state without starting the application.
`magnitude update download` explicitly admits a download of the selected offer. It acknowledges
admission and directs the caller to status; download completion belongs to the owner.
`magnitude update install` requires a prepared update and explicitly authorizes stopping the model
and service, installing and restarting the application.

Active update commands may ensure the desktop in the background, independently of inference
readiness. They never open or focus its window. The CLI neither generates another installation
identity nor creates a polling timer, downloads a second release, invokes npm, or independently
replaces the service. Cancellation of a CLI connection does not cancel an admitted download.

Local application control carries the same typed update state as the renderer. A failed command
returns a bounded product error. A check can wait for the bounded network result; other update
commands acknowledge admission. Restart acknowledgement is written before owner shutdown is queued.
Lost mutation replies are not automatically replayed.

## Release channels

Channels derive from the running version: stable follows stable, beta follows beta and stable,
alpha follows alpha, beta and stable. Unknown prerelease labels classify conservatively as stable.
Among compatible newer offers, the highest version wins. Channel selection never installs software.
The hosted release catalog maintains these channels; its manifests bind exact publisher, target,
version and bytes as specified by [Hosted updates](./hosted-updates.md).

## Distribution

The desktop package contains its exact CLI and service. Native app replacement updates them together.
Linux packages own the public CLI executable link as package contents; maintainer scripts do not
rewrite user PATHs. macOS places the headless executable within the signed application resources.
Native publisher verification and scoped owned-service shutdown precede replacement. A platform
without an accepted native transaction reports unavailability rather than falsely acknowledging an
installation.

## Acceptance

- Ordinary CLI startup performs no update check or terminal interaction.
- Status is passive, and active commands preserve background launch intent.
- CLI, Settings and tray address one update owner and one installation key.
- Invalid or failed checks never report up to date.
- Download admission is distinct from successful native staging.
- Install requires Ready and acknowledges before service/application shutdown.
- Packaged CLI and service versions match the application version.
