import { Effect } from "effect"
import type {
  ArtifactInstallationEvent,
  ArtifactInstallationObserver,
} from "./installation-progress"

interface LauncherProgressOutput {
  readonly isTTY?: boolean
  readonly write: (text: string) => unknown
}

export interface LauncherInstallationProgress {
  readonly observer: ArtifactInstallationObserver
  readonly succeeded: Effect.Effect<void>
  readonly failed: Effect.Effect<void>
}

const DECIMAL_MEGABYTE = 1_000_000
const DECIMAL_KILOBYTE = 1_000
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const

const formatBytes = (bytes: number): string =>
  bytes < DECIMAL_MEGABYTE
    ? `${Math.round(bytes / DECIMAL_KILOBYTE)} KB`
    : `${(bytes / DECIMAL_MEGABYTE).toFixed(1)} MB`

const eventProgress = (
  event: ArtifactInstallationEvent,
): { readonly completed: number; readonly total: number } =>
  event._tag === "Downloading"
    ? {
        completed: event.progress.acceptedBytes,
        total: event.progress.totalBytes,
      }
    : {
        completed: event.progress.completedBytes,
        total: event.progress.totalBytes,
      }

const eventLabel = (event: ArtifactInstallationEvent): string => {
  switch (event._tag) {
    case "Downloading":
      return "Downloading Magnitude CLI"
    case "Verifying":
      return "Verifying Magnitude CLI"
    case "Extracting":
      return "Installing Magnitude CLI"
  }
}

export const makeLauncherInstallationProgress = (
  output: LauncherProgressOutput = process.stderr,
): LauncherInstallationProgress => {
  const interactive = output.isTTY === true
  let lastInteractiveUpdate = ""
  let spinnerFrame = 0
  const reportedStages = new Set<ArtifactInstallationEvent["_tag"]>()

  const clear = Effect.sync(() => {
    if (interactive && lastInteractiveUpdate !== "") {
      output.write("\r\u001b[2K")
      lastInteractiveUpdate = ""
    }
  })

  return {
    observer: {
      report: (event) =>
        Effect.sync(() => {
          const label = eventLabel(event)
          const { completed, total } = eventProgress(event)
          if (!interactive) {
            if (!reportedStages.has(event._tag)) {
              reportedStages.add(event._tag)
              output.write(`${label}...\n`)
            }
            return
          }

          const percent = event._tag === "Downloading"
            ? Math.min(100, Math.floor((completed / Math.max(1, total)) * 100))
            : null
          const updateKey = percent === null ? event._tag : `${event._tag}:${percent}`
          if (updateKey === lastInteractiveUpdate) return
          lastInteractiveUpdate = updateKey
          const spinner = SPINNER_FRAMES[spinnerFrame++ % SPINNER_FRAMES.length]!
          const measurement = percent === null
            ? ""
            : ` ${percent}% (${formatBytes(completed)} / ${formatBytes(total)})`
          output.write(
            `\r\u001b[2K${spinner} ${label}...${measurement}`,
          )
        }),
    },
    succeeded: clear.pipe(Effect.zipRight(Effect.sync(() => {
      output.write("✓ Magnitude CLI installed\n")
    }))),
    failed: clear,
  }
}
