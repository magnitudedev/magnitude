import { createRequire } from "node:module"
import { Data, Effect, Option } from "effect"
import { MODEL_TRAY_PHASES, type ModelTrayStatus } from "@magnitudedev/client-common"

class TrayStatusRowUnavailable extends Data.TaggedError("TrayStatusRowUnavailable")<{ readonly message: string }> {}

/**
 * The tray menu's model line as a live native row, which only macOS provides: it keeps updating
 * while the menu is open. Elsewhere the line is the menu item's plain label.
 */
export interface TrayStatusRow {
  /** The next menu built carries the row at `index`. */
  readonly expect: (index: number) => void
  readonly present: (status: typeof ModelTrayStatus.Type) => void
}

interface TrayStatusBindings {
  readonly configureTrayStatus: (phases: ReadonlyArray<string>) => void
  readonly expectTrayStatus: (index: number) => void
  readonly updateTrayStatus: (model: string, phase: string, fraction: number | null, text: string | null) => void
}

/** The native row from the desktop tray addon; without it the tray keeps its plain label. */
export const loadTrayStatusRow = (addonPath: string): Effect.Effect<Option.Option<TrayStatusRow>> =>
  process.platform !== "darwin" ? Effect.succeed(Option.none()) : Effect.try({
    try: () => {
      const bindings = createRequire(import.meta.url)(addonPath) as TrayStatusBindings
      bindings.configureTrayStatus(MODEL_TRAY_PHASES)
      return Option.some<TrayStatusRow>({
        expect: index => bindings.expectTrayStatus(index),
        present: ({ model, phase, detail }) => {
          switch (detail._tag) {
            case "Working": return bindings.updateTrayStatus(model, phase, null, null)
            case "Progress": return bindings.updateTrayStatus(model, phase, detail.fraction, null)
            case "Memory": return bindings.updateTrayStatus(model, phase, null, detail.text)
          }
        },
      })
    },
    catch: error => new TrayStatusRowUnavailable({ message: String(error) }),
  }).pipe(Effect.catchTag("TrayStatusRowUnavailable", error =>
    Effect.logWarning(`The tray's live model row is unavailable: ${error.message}`).pipe(Effect.as(Option.none()))))
