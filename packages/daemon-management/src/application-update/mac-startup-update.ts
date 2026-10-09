import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { basename, dirname, join } from "node:path"
import { acquireUpdateInstallationLease } from "../desktop-native/update-installation-lease"
import { startMacForegroundInstallation } from "./mac-foreground-installation"

/** Startup only completes an interrupted transaction. A prepared update waits for an explicit install. */
export const macStartupUpdateOperation = (bundle: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const receipt = join(dirname(bundle), `.${basename(bundle)}.update`, "transaction.json")
  const recovering = yield* fs.stat(receipt).pipe(Effect.as(true), Effect.catchTag("SystemError", error =>
    error.reason === "NotFound" ? Effect.succeed(false) : Effect.fail(error)))
  return recovering ? Option.some("Recover" as const) : Option.none()
})

/** Runs under application admission, before any service or shared installation lease exists. */
export const prepareMacForegroundStartup = (options: {
  readonly resources: string
  readonly stateDirectory: string
  readonly dataDirectory: string
  readonly version: string
  readonly architecture: "arm64" | "x64"
  readonly arguments: readonly string[]
}) => Effect.gen(function* () {
  const operation = yield* macStartupUpdateOperation(dirname(dirname(options.resources)))
  if (Option.isNone(operation)) return
  yield* acquireUpdateInstallationLease(options.stateDirectory)
  return yield* startMacForegroundInstallation({ ...options, operation: operation.value,
    continuation: { _tag: "Foreground", arguments: options.arguments } })
})
