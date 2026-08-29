import { Effect } from "effect"
import { stopLocalAcn } from "../server/acn-instance-manager"

export const runStop = () => Effect.runPromise(stopLocalAcn.pipe(
  Effect.tap(() => Effect.sync(() => {
    process.stdout.write("Magnitude service stopped.\n")
  })),
  Effect.catchAll((error) => Effect.sync(() => {
    process.stderr.write(`Failed to stop Magnitude service: ${String(error)}\n`)
    process.exitCode = 1
  })),
))
