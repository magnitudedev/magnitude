---
applies_to:
  - packages/release/src/client-update/**
  - packages/launcher/src/**
  - packages/launcher/scripts/build-launcher.ts
  - cli/src/index.ts
  - cli/src/commands/update.ts
  - cli/src/commands/update-runtime.ts
  - cli/src/startup/*update*
  - cli/src/runtime/**
---

# Client release updates

The installed desktop application owns updates for itself, its bundled service and headless CLI.
Release discovery uses the Magnitude-hosted signed protocol. npm dist-tags and a caller's package
manager are not application update authorities. Neither the CLI nor the Pi extension is published
to npm. Normal Pi connections configure its models and skill without installing the extension.

Desktop installation exposes its bundled CLI directly: `~/.magnitude/bin/magnitude` points into the
Mac application, Windows adds the installed resources directory to the user's PATH, and Linux
retains its package-owned /usr/bin/magnitude link. Replacing the desktop at the same location
updates the command's executable without a second package update. On macOS the CLI resolves
its real executable path to locate its enclosing app, so a symlink cannot make it start a different
copy in /Applications. An explicit application override remains authoritative. Existing conflicting commands
must be resolved explicitly; uninstall removes only registration owned by that installation.

## Headless control

`magnitude update` and `magnitude update check` request a fresh check from the application owner.
`magnitude update status` observes current update state without starting the application.
`magnitude update download` explicitly admits a download of the selected offer. It acknowledges
admission and directs the caller to status; download completion belongs to the owner.
`magnitude update install` requires a prepared update. Desktop may stop its model and service to
install and restart; a running Headless owner refuses an explicit install, because it installs by
itself when idle.

No owner installs at startup: Desktop launch and `serve` start normally with a prepared update
retained, and Desktop offers Restart. A running Headless owner installs a downloaded update at an
idle point, when the service reports no working session and no inference request in flight. It
first checks that it may install unattended (the Linux server's sudoers rule, or a writable macOS
application folder); otherwise it keeps serving and prints one line naming `magnitude update
install`, recording no outcome. To install it ends its owner scope, which releases its leases, runs
the platform installer, and on success replaces its own process with the new version (same PID);
on failure it serves the current version again and, keeping the `failed` outcome, drops the failed
download, so the next check can fetch a newer release (or retry this one unless it was withdrawn). It checks the installation lock before stopping
anything: if another process holds or awaits it (the service's own shared lease does not count,
so this reads the kernel's lock table rather than trying the lock), it keeps serving, records `deferred`, reports it with an
immediate update check, and tries again at an idle point 15 minutes later. (A started service
refuses to run while the lock is held, so stopping first would leave it down until the lock frees.) Update checks, scheduled or requested, start only once the
service is ready, so a new version reports `applied` only after it starts; one that fails before
Ready reports `failed`/`startup` and exits non-zero for the service manager to retry. There is no
rollback; withdrawing the release stops it being offered.

Update commands never launch Desktop. With no owner, status observes persisted preparation without
creating state; finite check, download and discard retain maintenance ownership. A finite download
waits for verified preparation and scratch cleanup. Only a running owner retains a polling timer.
The CLI never updates separately from the application. Cancellation of a CLI connection does not
cancel a download admitted by a running owner.

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
- CLI, Settings and tray address one update owner.
- Invalid or failed checks never report up to date.
- Download admission is distinct from successful native staging.
- Install requires Ready and acknowledges before service/application shutdown.
- No owner installs at startup; Headless installs only at an idle point, and never blocks Ready.
- Packaged CLI and service versions match the application version.
