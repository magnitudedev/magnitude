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
  - cli/src/server/serve-updates.ts
  - cli/src/commands/serve-runtime.ts
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

`magnitude update` and `magnitude update check` request a fresh check from the application owner
and wait (up to two minutes) for it to finish, so they print its result. While a check runs, its
state supersedes the previous check's offer. A check that fails prints its failure and exits non-zero.
`magnitude update status` observes current update state without starting the application.
`magnitude update download` explicitly admits a download of the selected offer. It acknowledges
admission and directs the caller to status; download completion belongs to the owner. While a
download is already under way, staging or ready it answers with that state instead of failing.
An offer whose version is held back because it failed here carries the failure (verify, install,
authorization or startup), so `update status` and Settings say that version failed and is not
downloaded automatically, rather than presenting it as an ordinary available update.
`magnitude update install` with nothing downloaded prints the update status (up to date when the
last check found nothing) and asks no owner to install. Otherwise it requires a prepared update.
Desktop may stop its model and service to install and restart; a running Headless owner refuses an
explicit install, because it installs by itself when idle.

No owner installs at startup: Desktop launch and `serve` start normally with a prepared update
retained, and Desktop offers Restart. A macOS or Linux `serve`, the system service (`magnitude
server setup`) or one run by hand alike, installs unattended at an idle point, when the service
reports no working session and no inference request in flight, whenever it may do so without asking
anyone: on Linux when `sudo -n` admits the installer (the sudoers rule that setup writes for the
service, or a person's own password-free sudo), on macOS when the application folder is writable.
A `serve` that cannot, and every Windows `serve`, keeps serving and prints one line per downloaded
version naming `magnitude update install`, recording no outcome. To install the server ends its owner scope, which releases its
leases, runs the platform installer, and on success replaces its own process with the new version
(same PID). On failure it serves the current version again and holds back unattended installation:
no unattended attempt starts for an hour, and when the package itself failed it keeps the `failed`
outcome, drops the download and never downloads or installs that version unattended again, so the
next check can fetch only a newer release. The hold is persisted beside the prepared update, so it
outlives service restarts and the outcome's report; a failure before the installer ran keeps the
download for the next attempt after the hour. A manual check lifts only the hour's wait; the failed
version stays held until a person downloads it explicitly, which may retry it. It checks the installation lock before
stopping anything: if another process holds or awaits it (the service's own shared lease does not
count, so this reads the kernel's lock table rather than trying the lock), it keeps serving,
records `deferred`, reports it with an immediate update check, and holds unattended installation
for 15 minutes. (A started service refuses to run while the lock is held, so stopping first would
leave it down until the lock frees.) Update checks, scheduled or requested, start only once the
service is ready, so a new version reports `applied` only after it starts; one that fails before
Ready records `failed`/`startup`, holds that version back exactly like a failed package (so a
version installed by hand to recover is not updated straight back to it), sends the outcome with an
immediate check whose result it awaits, and exits non-zero for the service manager to retry. While
the running version is the held one, a restart checks again once the hour's wait has passed (and
then holds for another hour), so restarts never storm the update server yet a fixed release still
arrives: when that check leaves a newer download ready, the owner installs it in place of the
version that cannot start instead of exiting. There is no rollback; withdrawing the release stops
it being offered. Desktop likewise starts its scheduled checks only once its service is first Ready or Failed; a
service that fails before Ready records `failed`/`startup` and holds that version back the same way,
and the app, which keeps running, goes on checking hourly.

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
- No owner installs at startup; a Unix `serve` installs only at an idle point, only without a
  password prompt, and never blocks Ready.
- A failed unattended installation, or an installed version that never becomes ready, is never
  retried in a loop: the failed version is not downloaded or installed unattended again, and no
  unattended attempt starts within an hour, across restarts.
- A server whose installed version cannot start still checks hourly and installs a newer release.
- Packaged CLI and service versions match the application version.
- The update server accepts every request field a release sends (such as `owner` and
  `outcome_from`) before that release is published, since it rejects unknown fields.
