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

The release package owns channel interpretation and bounded dist-tag discovery for both CLI and
desktop consumers. Each caller verifies its required native artifacts and receives the exact
verified result from the selected candidate. Selection performs no installation or lifecycle action.
The CLI contains only its installation context, CLI-artifact verification and package-manager execution.
Mac desktop selection requires the exact update ZIP identity and filename for the running architecture.
A CLI-only release, installer-only release or archive for another architecture cannot qualify. The
selected value retains the manifest's version, size and digest for subsequent integrity verification.

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

### Explicit checks

Only `magnitude update` checks for CLI updates. Ordinary commands perform no update discovery,
prompts, notifications, or dismissal tracking. There is no CLI startup-update preference or offer
cache. Desktop application updates are a separate distribution concern.
The privileged CLI host supplies the update cache directory; the updater does not depend on
agent storage or independently choose an application profile.

A bounded dist-tags request selects admissible upgrades newest-first. A candidate qualifies only
when its public release manifest contains exactly one CLI artifact for the current host. Candidates
published before their native artifacts are skipped. Registry failure reports an actionable error
and exits nonzero; no qualifying candidate means the CLI is already up to date. Development builds
and unknown installation methods reject the update command before checking.

## Relaunch protocol

After a successful update, the CLI and its launcher complete the update without user action, with
manual restart as the guaranteed floor:

- The launcher sets `MAGNITUDE_LAUNCH_PROTOCOL_VERSION` in the CLI's environment.
- After the package manager succeeds, the CLI exits with the reserved relaunch exit code — but only
  when the environment's protocol version matches its own. On mismatch or absence it prints the
  manual-restart message instead: version skew degrades by definition, never by accident.
- The launcher honors a relaunch request **at most once per process**: it re-runs its own pipeline —
  locate the installation fresh, resolve the now-installed version's binary, and spawn it. An explicit `magnitude update` runs the
  new binary as `magnitude service start`, starting the desktop in the background without opening
  its window. Any failure prints the matching manual command and exits.
  A second relaunch request passes through as an ordinary exit.

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
directly to the executable rather than through a shell. Each explicit command performs a fresh,
channel-selected, readiness-verified check before running the pinned installation command.

## Required guarantees

- Ordinary CLI startup makes no update network request and presents no interaction.
- Every selected version has a matching native CLI artifact for the current host.
- Package-manager execution occurs only through explicit `magnitude update`.
- The launcher handles a post-update request at most once per process; failures provide the
  manual background service-start command.
- The CLI requests launcher follow-up only on an exact launch-protocol-version match.
