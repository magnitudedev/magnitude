---
applies_to:
  - cli/src/index.tsx
  - cli/src/commands/interactive.ts
  - cli/src/commands/interactive-runtime.ts
  - cli/src/commands/server.ts
  - cli/src/commands/server-runtime.ts
  - cli/src/runtime/**
  - cli/src/startup/**
  - cli/src/features/update/**
  - cli/src/platform/terminal.ts
  - cli/src/server/acn-connection.ts
  - cli/src/server/acn-instance-manager.ts
  - cli/src/platform/process-exit.ts
  - cli/src/platform/terminal-appearance.ts
  - packages/sdk/src/acn-jit/acn-recovering-client.ts
  - packages/sdk/src/acn-jit/local-acn-require-running-manager.ts
---

# CLI startup sequence

Service startup and update choice are inline terminal work owned by the command runtime. OpenTUI is
not a bootstrap surface. The interactive command creates its renderer only after the update choice,
exact service readiness, and onboarding preflight have completed. Consequently, a warm launch makes
no terminal writes before the application's first frame.

User-facing copy always says **service**. `ACN`, `daemon`, `server`, JIT, ownership, and endpoint
selection remain implementation terms.

## Service acquisition

Every service-backed command explicitly constructs the mechanism appropriate to its operation:

| Commands | Acquisition mechanism | Service absent |
| --- | --- | --- |
| Bare `magnitude` interactive launch | Bootstrapping `AcnInstanceManager` followed by an `AcnConnection` | Install, launch, and await exact readiness inline |
| `magnitude service start` | Explicit platform-service installation/start followed by an observing `AcnConnection` | Install/register/start and await exact readiness inline |
| Hardware, models, downloads, instances, slots, and connection mutations | Existing-service observer followed by an `AcnConnection` | Fail with `Magnitude service is not running. Run \`magnitude service start\`.` |

The terminal adapter contains only terminal and OS operations. It never selects a service-acquisition
mechanism and contains no RPC transport, startup lifecycle, recovery lifecycle, or connection close.
Noninteractive commands do not construct it.

`magnitude update` updates the package and, under a compatible launcher, asks the newly installed
CLI to run `magnitude service start`. Without the launcher protocol, success prints the explicit
service-start command as the degradation floor.

## Interactive launch

```text
shell launcher
  -> resolve native CLI (silent when cached; inline artifact progress when needed)
  -> probe terminal appearance and begin update discovery
  -> resolve an available update inline, if any
  -> construct the bootstrapping instance manager and ACN connection
  -> await exact Ready service occurrence inline
  -> run onboarding preflight
  -> create OpenTUI renderer
  -> first frame is Application
```

The ready check itself is silent. If the service is already warm, no lifecycle phase is printed and
the renderer opens directly into chat. If work is necessary, lifecycle observations drive an inline
active region. Completed phases remain as stable lines; the current phase is replaced in place.
Redirected output uses durable milestone lines and no cursor control codes.

`Checking` never renders. There is no synthetic generic startup phase before an authoritative
observation. A critical startup failure freezes the current progress as a durable transcript,
prints `Magnitude service failed to start:` followed immediately by the underlying error, and exits
nonzero. There is no retry/quit prompt; such a failure is treated as a product error.

## Inline update choice

The prompt uses the detected Magnitude theme and plain terminal input. Numbers are labels, not
shortcuts. Up/down changes the selected row, Enter confirms, Escape skips this launch, and Ctrl-C is
a normal process interruption. There are no j/k or direct-number controls.

```text
Update available! 1.3.0 → 1.4.0

Release notes: https://github.com/magnitudedev/magnitude/releases/tag/@magnitudedev/cli@1.4.0

› 1. Update now (runs `npm install -g @magnitudedev/cli@1.4.0`)
  2. Skip
  3. Skip until next version

Press enter to continue
```

The selector and selected number are theme-accent-colored and bold. The URL uses the theme link
color. Completed checks use the same sea-foam token as Markdown inline code. The URL stays on the label's line and
has no arrow glyph. Confirming erases the active prompt and leaves a short durable summary before
continuing or handing off to the updater.

## Inline service progress

The presenter projects the existing typed lifecycle as one nested service-start operation; it does
not own service decisions. The parent uses a static theme-blue `○` while active and `●` when ready.
Only the current child uses the animated Braille spinner.

| Lifecycle observation | Inline copy |
| --- | --- |
| `Checking` | Nothing |
| `Installing / DownloadingDaemon` | `Downloading Magnitude service... 63% (24.8 MB / 39.4 MB)` when exact bytes are known |
| `Installing / DownloadingInferenceEngine` | `Downloading inference engine... 63% (3.1 GB / 4.9 GB)` when exact bytes are known |
| `Installing / StartingMagnitude` | Child `Starting inference engine` |
| `Starting / WaitingForOwner` | `Waiting for previous Magnitude service` |
| `Starting / ResolvingLocalInference` | No child; absorbed into the parent operation |
| `Starting / LaunchingLocalInference` | Child `Starting inference engine` |
| `Starting / PreparingBackend` | Child `Preparing <backend> backend for <hardware>` |
| `Ready` | Complete the active child and change the parent `○` to `●` |
| `Failed` | Preserve the current progress; outer boundary prints the actual error and exits nonzero |

Service-binary acquisition feeds the same parent presenter before the lifecycle is observable. TTY
output rewrites the nested block with no progress bar; redirected output emits coarse percentage
milestones without ANSI animation.

```text
○ Starting Magnitude service
  ✓ Magnitude service downloaded 100% (39.4 MB / 39.4 MB)
  ✓ Inference engine downloaded 100% (4.9 GB / 4.9 GB)
  ✓ Inference engine started
  ⠹ Preparing CUDA backend for NVIDIA RTX 4090
```

At readiness:

```text
● Magnitude service is ready at 127.0.0.1:10100
  ✓ Magnitude service downloaded 100% (39.4 MB / 39.4 MB)
  ✓ Inference engine downloaded 100% (4.9 GB / 4.9 GB)
  ✓ Inference engine started
  ✓ CUDA backend ready for NVIDIA RTX 4090
```

The launcher uses the same one-line Braille grammar but shows measurements only for the transfer:

```text
⠹ Downloading Magnitude CLI... 63% (24.8 MB / 39.4 MB)
⠹ Verifying Magnitude CLI...
⠹ Installing Magnitude CLI...
✓ Magnitude CLI installed
```

## `magnitude service start`

This command is noninteractive. It installs or refreshes the per-user service definition, starts it,
uses the same lifecycle and inline presenter as interactive startup, and waits for the same exact
readiness guarantee. It then prints:

```text
Magnitude service is ready at 127.0.0.1:10100
```

Update discovery runs concurrently. If an update is available, the command reports the same version
transition and full release-notes URL without prompting:

```text
Update available! 1.3.0 → 1.4.0
Release notes: https://github.com/magnitudedev/magnitude/releases/tag/@magnitudedev/cli@1.4.0
Run `magnitude update` to install it.
```

## Post-start recovery

After the application starts, transport recovery retains the existing single-flight selection and
request retry semantics. Each recovery occurrence uses a fresh lifecycle projection, but it never
re-enters startup UI and never unmounts chat. Active recovery is projected into the shared
notification area; success publishes the ephemeral notice `Reconnected to Magnitude service`.

## Exit and ownership

All startup work is one scoped Effect program. Signals and fatal events use the typed process-exit
path. Package-manager execution begins only after startup terminal resources are restored. The outer
command owns the exit code; nested startup logic does not mutate it. The launcher may honor one
post-update relaunch request, re-inspecting the installation before resolving the new binary.
