import {
  LAUNCH_PROTOCOL_VERSION,
  LAUNCH_PROTOCOL_VERSION_VARIABLE,
  RELAUNCH_EXIT_CODE,
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
  options: { readonly relaunch: boolean } = { relaunch: false },
): Effect.Effect<number> => {
  const command = updateCommandString(action)
  return Effect.gen(function* () {
    yield* Effect.sync(() => {
      process.stdout.write(`\nUpdating Magnitude via \`${command}\`...\n`)
    })
    yield* updater.runUpdate(action)
    return yield* Effect.sync(() => {
      if (options.relaunch && launcherSpeaksRelaunch()) {
        // The launcher re-runs its pipeline and starts the new version.
        return RELAUNCH_EXIT_CODE
      }
      process.stdout.write("\nUpdate ran successfully. Please restart Magnitude.\n")
      return 0
    })
  }).pipe(
    Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`\n\`${command}\` failed: ${error.reason}\n`)
      return 1
    })),
  )
}
