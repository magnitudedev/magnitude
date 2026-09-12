import {
  LAUNCH_PROTOCOL_VERSION,
  LAUNCH_PROTOCOL_VERSION_VARIABLE,
  POST_UPDATE_SERVICE_START_EXIT_CODE,
  updateCommandString,
  type UpdateAction,
} from "@magnitudedev/release"
import { Effect } from "effect"
import type { CliUpdaterShape } from "./updater"

const launcherSpeaksRelaunch = (): boolean =>
  process.env[LAUNCH_PROTOCOL_VERSION_VARIABLE] === String(LAUNCH_PROTOCOL_VERSION)

export const executeUpdate = (
  updater: CliUpdaterShape,
  action: UpdateAction,
): Effect.Effect<number> => {
  const command = updateCommandString(action)
  return Effect.gen(function* () {
    yield* Effect.sync(() => {
      process.stdout.write(`\nUpdating Magnitude CLI via \`${command}\`...\n`)
    })
    yield* updater.runUpdate(action)
    return yield* Effect.sync(() => {
      if (launcherSpeaksRelaunch()) {
        return POST_UPDATE_SERVICE_START_EXIT_CODE
      }
      process.stdout.write("\nCLI update completed. Run `magnitude service start` to connect to the desktop service.\n")
      return 0
    })
  }).pipe(
    Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`\n\`${command}\` failed: ${error.reason}\n`)
      return 1
    })),
  )
}
