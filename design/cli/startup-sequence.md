---
applies_to:
  - cli/src/index.tsx
  - cli/src/commands/interactive.ts
  - cli/src/runtime/**
  - cli/src/features/app-shell/acn-bootstrap.tsx
  - cli/src/features/update/**
  - cli/src/platform/terminal.ts
  - cli/src/platform/process-exit.ts
  - cli/src/platform/terminal-appearance.ts
  - packages/sdk/src/acn-jit/lifecycle.ts
---

# CLI startup sequence

Nothing paints until there is an answer, and an answer is one of three things: the app, real
progress, or a question. Startup is one sequential Effect program owning one scoped terminal
session with one React root; only the Effect program writes presentation state, and every wait
races the typed process-exit request.

Naming discipline: **states** are code identifiers, **titles/labels** are user copy, **phases**
are protocol literals; they are never conflated. The TUI has exactly three states —
`UpdatePrompt | DaemonStartup | Application` — and the renderer (with the alternate screen) is
acquired at the first moment one of them must render, never earlier.

## Startup tree

```
magnitude (shell)
│
├─ LAUNCHER ──────────────────────────────────── surface: plain terminal
│   InstallationInspection   locate installed package        ~1–5ms      silent
│   CliBinaryResolution      resolve CLI binary              ~100–300ms  silent when cached
│     └ download + verify    (fresh install / just updated)  secs–mins   printed progress lines
│
└─ CLI ─────────────────────────────────────────(process boot ~100–200ms)
    AppearanceProbe          terminal color queries          ~1–10ms     silent, concurrent with
    │                        (100ms cap on silent ttys)                  everything below;
    │                                                                   joined before first paint
    UpdateDiscovery          version check                   ~200ms–1s   silent, concurrent with
    │                        (30s ceiling, silent failure)               everything below
    UpdatePrompt ─────────────────────────────── TUI state   user-paced  only when an offer exists
    │    Update → relaunch protocol · Skip / Dismiss → continue
    InstallationProbe        installed daemon build present? ~1–5ms      silent
    │    absent ──▶ await UpdateDiscovery's result (bounded by its ceiling) — an offer
    │              prompts before any download; installs need the network regardless
    PlatformConstruction     manager/runtime wiring          ~10ms       silent
    DaemonCheck              coordination read → probe →     ~10–100ms   silent
    │                        adopt
    │    daemon ready ────────────────▶ SessionConnect
    │    work needed ─────────────────▶ DaemonStartup
    DaemonStartup ────────────────────────────── TUI state
    │    driven by AcnLifecycleState:
    │    · Installing        daemon binary                   secs–mins   bar + % + sizes
    │      "Installing        inference engine                minutes
    │       Magnitude"
    │    · Starting          daemon spawn                    ~0.5–2s     text only
    │      "Starting          inference launch + backend prep ~1–5s
    │       Magnitude"
    │    · Failed            stage message                   —           R retry / Q quit
    │    lifecycle Ready ─────────────▶ SessionConnect
    SessionConnect           client connect + onboarding     warm ~10–100ms   silent
    │                        preflight                       (cold: absorbed above)
    Application ──────────────────────────────── TUI state   the product
```

Discovery race: a result arriving while daemon startup work is still running presents
`UpdatePrompt`; from daemon readiness on, a result presents one notification line, and the cached
answer prompts first thing next launch (`design/release/client-updates.md`).
An install-needed launch does not race at all: the probe holds the install sequence until the
discovery result answers, so an update is always offered before a multi-minute download of the
version it would replace, and an install path never paints `DaemonStartup` only to replace it
with the prompt. Spawn-only cold starts keep the race — the check usually outruns a spawn, and
interrupting one is free.

Warm-launch total ≈ half a second, silent end to end, dominated by two binary boots (the
resolution smoke test and the CLI itself). Cold-cold total is minutes, dominated by the inference
engine download, all narrated under `DaemonStartup`.

## Presentation

Every renderable state, phase, and subphase with its exact user copy. Titles are per substate —
"Installing Magnitude" while installing, "Starting Magnitude" while starting — with all
further variation in the subtext. Internal architecture terms (ACN, ICN, JIT, daemon, platform,
preflight) are never user copy.

### Launcher (plain terminal, no TUI)

| Phase | Copy | Indicator |
| --- | --- | --- |
| InstallationInspection, cached CliBinaryResolution | none | none — silent |
| CLI download (fresh install / just updated) | printed download lines with sizes | printed percentages, from artifact byte progress vs. manifest `bytes` |

### `UpdatePrompt` (TUI)

One centered panel; data source is the discovery result plus `updateActionFor(installMethod)`:

| Element | Exact copy |
| --- | --- |
| Title | "Update available! {current} -> {latest}" (accent bold on "Update available!") |
| Release-notes label | "Release notes:" |
| Release-notes link | `https://github.com/magnitudedev/magnitude/releases/tag/{tag}↗` — own line, underlined, supporting color, link-blue on hover, clickable |
| Option 1 | "1. Update now (runs \`{update command}\`)" |
| Option 2 | "2. Skip" |
| Option 3 | "3. Skip until next version" |
| Footer | "Press Enter to continue" |

Indicator: `›` plus accent bold on the highlighted option; ↑/↓/j/k move, 1–3 select directly,
Enter confirms, Esc/Ctrl-C skip.

### `DaemonStartup` · Installing — title "Installing Magnitude"

Progress bar + percentage under the title on every phase, driven by `Installing.overallProgress`
(0–1, weighted across the installation plan, monotonic across phases). Subtext is the phase
label, plus " · {A} of {B}" sizes when the phase's `AcnStartupProgress` detail is exact and
byte-denominated:

| Phase (protocol literal) | Subtext label | Sizes shown |
| --- | --- | --- |
| `DownloadingDaemon` | "Downloading daemon" | yes, when exact |
| `DownloadingInferenceEngine` | "Downloading inference engine" | yes, when exact |
| `StartingMagnitude` | "Starting Magnitude" | never — time-based synthetic progress (first launch of the freshly installed inference engine; wire value only, never a state name) |

### `DaemonStartup` · Starting — title "Starting Magnitude"

Text only — no bar, no spinner. Subtext is the phase label from `Starting.phase`:

| Phase | Subtext label |
| --- | --- |
| `PreparingAcn` | "Preparing background server" |
| `WaitingForOwner` | "Waiting for previous Magnitude process" |
| `ResolvingLocalInference` | "Preparing local inference" |
| `LaunchingLocalInference` | "Starting local inference" |
| `PreparingBackend { backend, hardwareLabel }` | "Preparing {CPU\|Metal\|CUDA\|Vulkan} backend for {hardwareLabel}" |

### `DaemonStartup` · Failed

Title in failure color by `Failed.stage`; subtext is `Failed.message` verbatim; footer
"R Retry Q Quit":

| Stage | Title |
| --- | --- |
| `InstallDaemon`, `PrepareLocalInference` | "Magnitude failed to install" |
| `LaunchDaemon`, `Connect` | "Magnitude failed to start" |

### `Application`

The product; all further presentation belongs to in-app systems. A discovery result arriving
after this commit presents one notification line
("Update available: {latest} — restart or run `magnitude update`", once per session).

## Data flow

`AcnLifecycleState` (`Checking | Starting | Installing | Ready | Failed`) is the single feed for
`DaemonStartup`. Client-side ensure work (daemon binary download, spawn) reports it directly;
daemon-side work (inference engine download, backend preparation) is projected into it from the
daemon's health state. `Checking` renders nothing by definition — it is `DaemonCheck`, the
question, not the answer. `overallProgress` is normalized against the installation plan so the
bar is monotonic across phases; byte subtext appears only when the underlying detail is exact.

## Appearance

Appearance is probed before the renderer exists, with OpenTUI's standalone palette detector on
the raw process streams (raw mode and stdin flow are the probe's to manage there — no renderer
owns the terminal yet). The probe starts with the session, concurrent with discovery and daemon
work, and settles on the first terminal reply or a 100ms bound on silent ttys — so its answer is
in before anything paints and the first frame is themed from live detection. No appearance state
is persisted. Corrections continue to apply live through the appearance observation runtime
(renderer theme, palette, and focus events).

## Exit

All exit paths — signal, fatal event, in-app quit — resolve one typed exit request. The graceful
path closes the client/platform exactly once, derives exit notices, and unwinds scope finalizers
in LIFO order (root, listeners, appearance observation, renderer and terminal, registry). Notices
print after terminal restoration. The completed command returns one exit code to the outermost CLI
boundary, which explicitly terminates the process. A successful compatible update returns the
reserved relaunch code through the same boundary; no inner runtime mutates process exit state.
