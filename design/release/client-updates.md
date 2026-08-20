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

### Channels

A release channel is derived from a version's prerelease identifier: none → stable, `alpha` →
alpha, `beta` → beta. The client's channel comes from its own running version, and it admits
update candidates by the candidate's channel:

- stable clients follow only stable releases;
- beta clients follow stable and beta releases;
- alpha clients follow stable, beta, and alpha releases.

A prerelease identifier outside these channels is admitted nowhere and classifies a client
conservatively as stable. Publishing maintains the channels with npm dist-tags: changesets pre
mode publishes under the pre id's tag (`alpha`, `beta`) and moves `latest` only outside pre mode.

The single dist-tags request the check already makes carries every channel's candidate; the
client decodes `latest`, `beta`, and `alpha` and discards nothing it needs. Selection is a pure
function: among admissible candidates newer than the running version, the highest version wins.
Semver's prerelease ordering (`alpha < beta < stable` within a base) makes cross-channel
supersession fall out naturally.

### Checks and the cache

Every interactive launch fires one discovery check, concurrently with startup, supervised by the
startup scope. There is no check interval or cache expiry. A candidate is an available update
only when it is newer than the running version, admissible for this client's channel, and its
public release manifest contains exactly one CLI artifact for the current host — the readiness
check. The check walks the admissible upgrades newest-first and stops at the first that passes
readiness, so the normal case verifies exactly one manifest; a candidate published before its
release assets is skipped, not fatal. Readiness exists because the launcher acquires the native
CLI from the corresponding GitHub release; an update must never be offered before its binary is
downloadable.

The cache (`state/version.json`) stores the selected, readiness-verified candidate of the last
completed check — including the empty result, which erases the cache. A completed check is
authoritative: its selection stands even when it retracts a cached offer (registry rollback);
the cached answer stands in only when the check itself failed. Check failures are silent, leave
the prior answer intact, and never delay or prevent startup. Admissibility, newness, and the
dismissal floor are re-applied when the cache is read: the running binary — and so its channel —
may have changed since the write.

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

Dismissal state is user-owned and stored separately from the discovery answer
(`state/version-dismissal.json`). A dismissal is a floor: it suppresses the prompt and the
notification for every candidate at or below the dismissed version — the user declined the best
available option, so a strictly older one is never surfaced in its place — and anything strictly
newer re-engages. The floor is monotonic (only ever-higher versions can be dismissed, since only
offers above the floor are shown) and self-healing: a completed check whose best candidate falls
below the floor clears it, because the registry retreated past what was dismissed. The explicit
update command ignores dismissals entirely. Source and development builds do not check.

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

Update actions follow the manager's ordinary global installation command, pinned to the exact
selected version — pinning keeps the offer and the installation identical and keeps prerelease
clients on their own channel, where an unpinned install would resolve `latest`:

- npm runs `npm install -g @magnitudedev/cli@<version>`.
- Bun runs `bun install -g @magnitudedev/cli@<version>`.
- pnpm runs `pnpm add -g @magnitudedev/cli@<version>`.

The visible command and executed command come from the same structured action. Arguments are passed
directly to the executable rather than through a shell. `magnitude update` resolves its target the
same way the prompt does — one fresh check, channel-selected, readiness-verified — but ignores
dismissals: asking to update overrides having dismissed. No target means "already up to date"; a
failed check reports itself and exits nonzero. The command is unavailable to development builds or
unknown installation methods.

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
