import type { Command } from "@commander-js/extra-typings"
import { Effect } from "effect"
import { stopTerminalAcn } from "../platform/terminal"

export const registerStopCommand = (program: Command): void => {
  program
    .command("stop")
    .description("Stop the current Magnitude daemon and release its local models")
    .action(() => Effect.runPromise(stopTerminalAcn.pipe(
      Effect.tap(() => Effect.sync(() => {
        process.stdout.write("Magnitude daemon stopped.\n")
      })),
      Effect.catchAll((error) => Effect.sync(() => {
        process.stderr.write(`Failed to stop Magnitude daemon: ${String(error)}\n`)
        process.exitCode = 1
      })),
    )))
}
