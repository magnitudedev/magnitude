---
applies_to:
  - packages/cli/bin/magnitude.js
  - packages/sdk/src/cli-update.ts
  - packages/sdk/src/cli-update.test.ts
  - packages/storage/src/types/config.ts
  - cli/src/index.tsx
  - cli/src/features/update/**
  - cli/src/utils/graceful-shutdown.ts
  - packages/release/src/launcher-install-context.test.ts
---

# CLI updates

Magnitude delegates installation updates to the package manager that launched its npm wrapper. The
supported methods are npm, Bun, and pnpm. The wrapper detects its manager and passes that context to
the native CLI; an unrecognized native invocation has no automatic update action. Magnitude does not
silently update, invoke elevated privileges, restart itself, or modify a package-manager installation
directly.

## Discovery and caching

The npm registry's `latest` dist-tag is the sole version target. A prerelease identifier in a package
version does not select an npm dist-tag, and the updater does not derive a channel from the installed
version. Changesets remains responsible for advancing `latest` during publication.

Update discovery is cached for twenty hours. Startup reads the existing cache without waiting for
the network. A missing or stale cache starts a background refresh, whose result becomes visible on a
later launch. Cache read, decode, and write failures are misses and never prevent CLI startup.

A registry version is eligible for caching as an available update only when it is newer than the
running version and the matching public release manifest contains exactly one CLI artifact for the
current host. This readiness check is required because the npm wrapper acquires its native CLI from
the corresponding GitHub release. An unavailable or invalid release leaves the prior cache intact.

Discovery state records the latest ready version and its check time. User-owned dismissal state is
stored separately so a concurrent background refresh cannot overwrite it. A dismissal suppresses
only that exact version; a newer discovered version is eligible again. Source and development builds
do not check for updates.

## Startup interaction

When the cache contains a newer, non-dismissed version, the interactive CLI presents three choices
before starting ACN:

1. Update now.
2. Skip this launch.
3. Skip until the next version.

Accepting an update exits and restores the terminal before invoking the package manager. A
successful update asks the user to restart Magnitude; the old process never continues into normal
startup. A failed update reports the command failure and exits nonzero without changing this
lifecycle. Noninteractive launches and launches with an initial prompt do not show the startup
prompt.

The global `checkForUpdateOnStartup` configuration value defaults behaviorally to true when absent.
Setting it to false disables cached startup prompts and background refreshes, but does not disable an
explicit update command.

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

- Startup never waits for update network access.
- Update-cache failure never prevents startup or discards a valid previous cache value.
- An offered version has a matching native CLI artifact for the current host.
- Package-manager execution occurs only after an explicit user choice or `magnitude update`.
- ACN does not start before the update prompt is resolved.
- The package-manager command is run only after terminal restoration.
