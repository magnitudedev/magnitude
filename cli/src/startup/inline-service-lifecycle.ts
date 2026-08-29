import * as Terminal from "@effect/platform/Terminal"
import type {
  AcnEnsuranceError,
  AcnLifecycleState,
  AcnStartup,
  AcnStartupProgress,
  BinaryAcquisitionEvent,
} from "@magnitudedev/sdk"
import { formatStorageSize } from "@magnitudedev/client-common"
import ansis from "ansis"
import { Clock, Duration, Effect, Exit, Fiber, Option, Stream } from "effect"
import type { CliTheme } from "../types/theme-system"
import { spinnerFrameAt } from "../utils/spinner"
import { serviceAcquisitionChildPhase } from "./inline-service-acquisition"

export const inlineServiceCompletionColor = (theme: CliTheme): string =>
  theme.markdown.inlineCode

const exactProgressCopy = (progress: AcnStartupProgress): string => {
  const percentage = Math.floor(Math.max(0, Math.min(1, progress.completed / Math.max(1, progress.totalBytes))) * 100)
  return `${percentage}% (${formatStorageSize(progress.completed)} / ${formatStorageSize(progress.totalBytes)})`
}

type ChildKey = "service-download" | "inference-download" | "previous-service"
  | "local-inference" | "backend"

interface ChildPhase {
  readonly key: ChildKey
  readonly active: string
  readonly completed: string
  readonly progress: number | null
}

interface ChildRow extends ChildPhase {
  readonly status: "active" | "completed"
}

const completedDownload = (
  subject: "Magnitude service" | "Inference engine",
  progress: Option.Option<AcnStartupProgress>,
  exact: boolean,
): string => {
  if (!exact) return `${subject} downloaded 100%`
  return Option.match(progress, {
    onNone: () => `${subject} downloaded 100%`,
    onSome: ({ totalBytes }) => {
      const total = formatStorageSize(totalBytes)
      return `${subject} downloaded 100% (${total} / ${total})`
    },
  })
}

export const serviceStartupChildPhase = (state: AcnLifecycleState): ChildPhase | null => {
  if (state._tag === "Installing") {
    if (state.phase === "StartingMagnitude") {
      return {
        key: "local-inference",
        active: "Starting inference engine",
        completed: "Inference engine started",
        progress: null,
      }
    }
    const subject = state.phase === "DownloadingDaemon"
      ? "Magnitude service" as const
      : "Inference engine" as const
    const detail = state.detailIsExact
      ? Option.map(state.detail, exactProgressCopy)
      : Option.none()
    const percentage = Math.floor(state.overallProgress * 100)
    const activeSubject = subject === "Magnitude service" ? subject : subject.toLowerCase()
    return {
      key: state.phase === "DownloadingDaemon" ? "service-download" : "inference-download",
      active: `Downloading ${activeSubject}... ${Option.getOrElse(detail, () => `${percentage}%`)}`,
      completed: completedDownload(subject, state.detail, state.detailIsExact),
      progress: state.overallProgress,
    }
  }
  if (state._tag !== "Starting") return null
  if (typeof state.phase !== "string") {
    const backend = `${state.phase.backend._tag} backend for ${state.phase.backend.hardwareLabel}`
    return {
      key: "backend",
      active: `Preparing ${backend}`,
      completed: `${state.phase.backend._tag} backend ready for ${state.phase.backend.hardwareLabel}`,
      progress: null,
    }
  }
  switch (state.phase) {
    case "PreparingAcn":
    case "ResolvingLocalInference":
      return null
    case "WaitingForOwner":
      return {
        key: "previous-service",
        active: "Waiting for previous Magnitude service",
        completed: "Previous Magnitude service stopped",
        progress: null,
      }
    case "LaunchingLocalInference":
      return {
        key: "local-inference",
        active: "Starting inference engine",
        completed: "Inference engine started",
        progress: null,
      }
  }
}

const clearRenderedLines = (lineCount: number): string => lineCount === 0
  ? ""
  : `\u001b[${lineCount}A\r\u001b[0J`

export interface InlineServiceStartupPresenter {
  readonly acquisitionObserver: {
    readonly report: (event: BinaryAcquisitionEvent) => Effect.Effect<void>
  }
  readonly acquisitionSucceeded: Effect.Effect<void>
  readonly run: (startup: AcnStartup) => Effect.Effect<void, AcnEnsuranceError>
  readonly clear: Effect.Effect<void>
}

export const makeInlineServiceStartupPresenter = (
  theme: CliTheme,
  options: {
    readonly showReadyWhenNoWork?: boolean
  } = {},
): Effect.Effect<InlineServiceStartupPresenter, never, Terminal.Terminal> => Effect.gen(function* () {
  const terminal = yield* Terminal.Terminal
  const tty = yield* terminal.isTTY
  const accent = tty ? ansis.hex(theme.status.progress) : (value: string) => value
  const success = tty ? ansis.hex(inlineServiceCompletionColor(theme)) : (value: string) => value
  const display = (text: string) => terminal.display(text).pipe(Effect.orDie)
  const readyLabel = "Magnitude service is ready at 127.0.0.1:10100"
  const rows = new Map<ChildKey, ChildRow>()
  let activeLines = 0
  let visible = false
  let nonTtyParentShown = false
  let nonTtyActiveCopy = ""
  let ready = false

  const clear = Effect.suspend(() => {
    if (!tty || activeLines === 0) return Effect.void
    const erased = clearRenderedLines(activeLines)
    activeLines = 0
    return display(erased)
  })
  const snapshot = (timeMs: number): ReadonlyArray<string> => {
    const parent = ready
      ? `${accent("●")} ${readyLabel}`
      : `${accent("○")} Starting Magnitude service`
    return [parent, ...rows.values().map((row) => row.status === "completed"
      ? `  ${success("✓")} ${row.completed}`
      : `  ${accent(spinnerFrameAt(timeMs))} ${row.active}`)]
  }
  const renderTty = Effect.gen(function* () {
    yield* clear
    const lines = snapshot(yield* Clock.currentTimeMillis)
    activeLines = lines.length
    visible = true
    yield* display(`${lines.join("\n")}\n`)
  })
  const ensureNonTtyParent = Effect.suspend(() => {
    if (nonTtyParentShown) return Effect.void
    nonTtyParentShown = true
    visible = true
    return display("○ Starting Magnitude service\n")
  })
  const completeActive = Effect.gen(function* () {
    const active = [...rows.values()].find((row) => row.status === "active")
    if (active === undefined) return
    rows.set(active.key, { ...active, status: "completed" })
    if (!tty) yield* display(`  ✓ ${active.completed}\n`)
  })
  const activate = (phase: ChildPhase): Effect.Effect<void> => Effect.gen(function* () {
    const active = [...rows.values()].find((row) => row.status === "active")
    if (active !== undefined && active.key !== phase.key) yield* completeActive
    rows.set(phase.key, { ...phase, status: "active" })
    if (tty) return yield* renderTty
    yield* ensureNonTtyParent
    const copy = phase.progress === null
      ? `${phase.key}:${phase.active}`
      : `${phase.key}:${Math.floor(phase.progress * 4)}`
    if (copy === nonTtyActiveCopy) return
    nonTtyActiveCopy = copy
    yield* display(`  ${phase.active}\n`)
  })
  const complete = (key: ChildKey): Effect.Effect<void> => Effect.gen(function* () {
    const row = rows.get(key)
    if (row === undefined || row.status === "completed") return
    rows.set(key, { ...row, status: "completed" })
    if (tty) return yield* renderTty
    yield* display(`  ✓ ${row.completed}\n`)
  })

  const present = (state: AcnLifecycleState): Effect.Effect<void> => Effect.gen(function* () {
    if (state._tag === "Checking") return
    if (state._tag === "Failed") {
      // Keep the startup transcript visible so the fatal error that follows
      // has the complete operation history above it.
      return
    }
    if (state._tag === "Ready") {
      if (!visible && options.showReadyWhenNoWork !== true) return
      yield* completeActive
      ready = true
      if (tty) return yield* renderTty
      yield* display(`● ${readyLabel}\n`)
      return
    }
    const phase = serviceStartupChildPhase(state)
    if (phase !== null) yield* activate(phase)
  })

  const run = (startup: AcnStartup): Effect.Effect<void, AcnEnsuranceError> => Effect.gen(function* () {
    const observations = tty
      ? Stream.merge(
          startup.state.changes,
          Stream.tick(Duration.millis(80)).pipe(Stream.mapEffect(() => startup.state.get)),
        )
      : startup.state.changes
    const updates = yield* observations.pipe(
      Stream.runForEach(present),
      Effect.fork,
    )
    const result = yield* Effect.exit(startup.awaitReady).pipe(
      Effect.onInterrupt(() => clear),
      Effect.ensuring(Fiber.interrupt(updates)),
    )
    yield* present(yield* startup.state.get)
    if (Exit.isFailure(result)) return yield* Effect.failCause(result.cause)
  })

  const acquisitionObserver = {
    report: (event: BinaryAcquisitionEvent): Effect.Effect<void> => Effect.gen(function* () {
      const phase = serviceAcquisitionChildPhase(event)
      if (phase === null) return
      yield* activate({
        key: "service-download",
        ...phase,
      })
    }),
  }
  const acquisitionSucceeded = complete("service-download")

  return {
    acquisitionObserver,
    acquisitionSucceeded,
    run,
    clear,
  }
})
