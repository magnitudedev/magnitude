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

A running Headless owner accepts status, check, download and discard through application control.
It reports prepared releases with stop-and-start guidance. Install requests fail without stopping
the service or admitting an installer; preparation and its timer end with the owner's scope.

Without a running owner, update observation reads persisted preparation and preferences without
creating an identity, owner or update worker. An explicitly admitted finite preparation command holds
maintenance ownership: check performs one check without an automatic transfer, download checks and
waits through verified durable preparation and scratch retirement, and discard waits for durable
removal. These commands do not modify the automatic-download preference or install an application.
Cancellation retires scoped transfer work; it cannot publish a partial prepared installer. Existing
prepared or failed installation state is retained until explicit installation retry or discard.

An explicit Linux installation without an owner holds maintenance and installation admission while
the installed privileged package helper runs. Terminal authorization may prompt only with an
interactive terminal; unattended execution uses noninteractive authorization. The helper validates
the authorizing user's identity and installed publisher trust. Success requires both successful
package installation and the replacement CLI reporting the prepared version before retained state
is removed. Cancellation retires privileged installation descendants, cannot report success or clear
an attempted installation, and preserves package-manager repair state when replacement was interrupted.
Linux Headless startup considers only unattempted preparation before acquiring shared installation
admission or starting a service. It checks authorization for the exact installed helper without a
prompt; unavailable authorization retains preparation and permits ordinary startup. An admitted
installation must complete and verify replacement before the foreground process executes the new
CLI with the same invocation. Failed or interrupted attempts are never automatically retried.

Updates must match the application platform and release channel. The client verifies release
signatures, downloaded file integrity, and applicable native publisher signatures before
installation. Failed checks or verification must not be reported as successful updates.

macOS staged-bundle verification uses the publisher identity compiled into the installed application;
missing publisher configuration cannot become identifier-only update trust. Native verification checks
sealed resources, nested code, every architecture slice, the required executable architecture, and the
sealed application identifier and release version. The caller retains exclusive staging ownership
through verification and publication. A successful signature check is not installation admission,
archive containment validation or notarization acceptance; those remain separate transaction gates.

macOS archive extraction operates in an empty private staging directory, separately from the live
installation. It permits only one application root, bounded entries and expanded bytes, ordinary
files, directories and contained relative links. It cannot create special files, hard links or
privilege-bearing modes, write through archive-created symlinks, or overwrite duplicate entries.
Framework version links, executable permissions and macOS resource metadata survive extraction.
Dangling or cyclic links fail staging. No partially extracted tree authorizes installation; the
transaction authenticates release metadata against bundled publisher keys before extraction. The
extractor checks the signed digest and byte count against the same retained, no-follow archive
descriptor before and after parsing. It refuses writable-by-others or hard-linked input. A native
lifetime guard contains extraction through cancellation and installer death; staging remains retained
until the extractor retires. The transaction then verifies the resulting signed bundle.

Native macOS transaction filesystem capabilities retain directory descriptors and revalidate their
identities before access. Private journal directories and records cannot carry broader permissions,
extended access grants, symlinks or hard links. A record is bounded, written completely to a new private
file, synchronized, atomically published and synchronized with its parent before acknowledging success.
Bundle exchange is descriptor-relative and requires both expected directory identities; stale requests
cannot exchange the bundles again. An exchange error may occur after namespace mutation and requires
identity reconciliation, never blind retry. These capabilities belong to the finite installer process;
they neither acquire installation exclusion nor authorize mutation on their own.

macOS installation admission uses a shared kernel lease outside the replaceable application bundle.
The stable adjacent lock is owned by the installation owner, readable by other users, and never
unlinked or rewritten. Running owners retain shared admission; an installer requires exclusive
admission and revalidates the retained lock and parent identities before mutation. Unsafe permissions,
extended access grants, substituted paths and observation failures are errors, not contention. The
lease does not establish that an older application version participates in admission; migration must
separately exclude prior-version owners before automatic replacement is enabled.

macOS recovery validates a bounded, schema-checked journal bound to both retained parent identities
and the installation name. Observed bundle identities determine the result: an unexecuted exchange is
abandoned for explicit retry, while an observed valid replacement completes commit without another
exchange. An invalid uncommitted replacement permits rollback only after the displaced old bundle is
verified and restoration intent is durable. Recovery can resume that restoration before or after its
exchange. A committed update never rolls back. Unknown identities, malformed records, failed validation
or incomplete durability require repair before service startup. Completed cleanup may remove the
displaced bundle only after a terminal journal state and successful revalidation of the installed
bundle. Deletion stays within the retained private directory and cannot traverse symbolic links;
partial deletion retains the terminal journal for retry. Cleanup removes and durably synchronizes the
exact terminal receipt last, before another transaction can supersede its installed identity.
Cleanup failure is distinct from an invalid installed bundle. Recovery itself preserves contents.

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
