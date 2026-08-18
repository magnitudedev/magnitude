import {
  updateCommandString,
  type UpdateAction,
} from "@magnitudedev/release"
import { Effect } from "effect"
import type { CliUpdaterShape } from "./updater"

export const executeUpdate = (
  updater: CliUpdaterShape,
  action: UpdateAction,
): Effect.Effect<void> => {
  const command = updateCommandString(action)
  return Effect.gen(function* () {
    yield* Effect.sync(() => {
      process.stdout.write(`\nUpdating Magnitude via \`${command}\`...\n`)
    })
    yield* updater.runUpdate(action)
    yield* Effect.sync(() => {
      process.stdout.write("\nUpdate ran successfully. Please restart Magnitude.\n")
    })
  }).pipe(
    Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`\n\`${command}\` failed: ${error.reason}\n`)
      process.exitCode = 1
    })),
  )
}
