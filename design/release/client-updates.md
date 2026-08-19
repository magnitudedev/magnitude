---
applies_to:
  - packages/release/src/client-update/**
  - packages/launcher/src/**
  - packages/launcher/scripts/build-launcher.ts
  - packages/storage/src/types/config.ts
  - cli/src/index.tsx
  - cli/src/commands/update.ts
  - cli/src/features/update/**
  - cli/src/runtime/**
  - cli/src/platform/process-exit.ts
  - cli/src/platform/terminal.ts
---

# CLI updates

Magnitude delegates installation updates to the package manager that launched its npm wrapper. The
supported methods are npm, Bun, and pnpm. The launcher detects its manager and passes that context
to the native CLI; an unrecognized native invocation has no automatic update action. Detection
reads only filesystem markers in the installation tree (pnpm's `.modules.yaml`, bun's lockfile
beside the owning `node_modules`; npm leaves no marker and is the conclusion when neither is
present) — never environment hints, which describe whichever tool spawned the process rather than
whichever owns the installation. Magnitude does not silently update, invoke elevated privileges,
or modify a package-manager installation directly.

## Discovery

The npm registry's `latest` dist-tag is the sole version target. A prerelease identifier in a
package version does not select an npm dist-tag, and the updater does not derive a channel from the
installed version. Changesets remains responsible for advancing `latest` during publication.

Every interactive launch fires one discovery check, concurrently with startup, supervised by the
startup scope. There is no check interval or cache expiry: the check is two small requests from
human-frequency launches. The cache holds only the last known answer, so a known update can be
offered instantly without the network. Check failures are silent, leave the prior answer intact,
and never delay or prevent startup.

A registry version is an available update only when it is newer than the running version and the
matching public release manifest contains exactly one CLI artifact for the current host. This
readiness check exists because the launcher acquires the native CLI from the corresponding GitHub
release; an update must never be offered before its binary is downloadable.

The check result is consumed by a race:

- Known from cache → the update prompt precedes all daemon work.
- No installed daemon build (fresh machine, wiped cache) → startup awaits this launch's check
  result before the install sequence begins: an offer always prompts before any download, and a
  multi-minute install of a version about to be replaced never starts. The wait is bounded by the
  check's own ceiling and costs nothing real — installation needs the network regardless.
- Fresh result arrives while daemon startup work is still running (cold spawn — the check
  usually outruns it) → the prompt is shown this launch, before expensive work proceeds.
- Fresh result arrives from daemon readiness on (typical warm start) → one in-session
  notification line, and the now-cached answer prompts first thing next launch. Startup latency
  is never added to wait for the network.

Dismissal state is user-owned and stored separately from the discovery answer. A dismissal
suppresses both the prompt and the notification for exactly that version; a newer version is
eligible again. Source and development builds do not check.

## Startup interaction

When an update is offered, the prompt renders as a phase of the single startup root — before any
daemon work — with three choices: update now, skip this launch, skip until the next version.
Noninteractive launches and launches with an initial prompt never show it.

Accepting an update completes the interactive scope — React unmount, renderer destruction, terminal
restoration, listener removal, supervised-fiber interruption — before invoking the package manager.
A failed update reports the command failure and exits nonzero.

The global `checkForUpdateOnStartup` configuration value defaults behaviorally to true when absent.
Setting it to false disables discovery, prompts, and notifications, but not the explicit update
command.

## Relaunch protocol

After a successful update, the CLI and its launcher complete the update without user action, with
manual restart as the guaranteed floor:

- The launcher sets `MAGNITUDE_LAUNCH_PROTOCOL_VERSION` in the CLI's environment.
- After the package manager succeeds, the CLI exits with the reserved relaunch exit code — but only
  when the environment's protocol version matches its own. On mismatch or absence it prints the
  manual-restart message instead: version skew degrades by definition, never by accident.
- The launcher honors the relaunch code **at most once per process**: it re-runs its own pipeline —
  locate the installation fresh, resolve the now-installed version's binary, spawn it. Any failure
  in that iteration prints "update installed — run `magnitude`" and exits. A second relaunch code
  passes through as an ordinary exit.

The degraded outcome of every relaunch failure — incompatible new release, broken binary, unchanged
version — is exactly the manual-restart behavior, plus one fast failed resolution attempt.

## Package-manager actions

Update actions follow the manager's ordinary global installation command and therefore resolve npm
`latest` at execution time:

- npm runs `npm install -g @magnitudedev/cli`.
- Bun runs `bun install -g @magnitudedev/cli`.
- pnpm runs `pnpm add -g @magnitudedev/cli`.

The visible command and executed command come from the same structured action. Arguments are passed
directly to the executable rather than through a shell. `magnitude update` runs this action directly
without first performing discovery; it is unavailable to development builds or unknown installation
methods.

## Required guarantees

- Startup never waits for update network access, with one deliberate exception: a launch with no
  installed daemon build awaits the version check before downloading.
- Discovery failure never prevents startup or discards the previous known answer.
- An offered version has a matching native CLI artifact for the current host.
- Package-manager execution occurs only after an explicit user choice or `magnitude update`.
- ACN does not start before the update prompt is resolved.
- The package-manager command is run only after terminal restoration.
- The launcher relaunches at most once per process; every relaunch failure degrades to the
  manual-restart message.
- The CLI emits the relaunch exit code only on an exact launch-protocol-version match.
- Discovery and terminal resources cannot outlive the interactive startup scope.
