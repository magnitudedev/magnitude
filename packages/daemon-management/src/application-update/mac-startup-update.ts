import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { basename, dirname, join } from "node:path"
import { acquireUpdateInstallationLease } from "../desktop-native/update-installation-lease"
import { startMacForegroundInstallation } from "./mac-foreground-installation"
import { reconcilePreparedUpdate } from "./prepared-update-installation"

/** A receipt takes precedence over preparation; failed attempts never authorize startup installation. */
export const macStartupUpdateOperation = (bundle: string, version: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const receipt = join(dirname(bundle), `.${basename(bundle)}.update`, "transaction.json")
  const recovering = yield* fs.stat(receipt).pipe(Effect.as(true), Effect.catchTag("SystemError", error =>
    error.reason === "NotFound" ? Effect.succeed(false) : Effect.fail(error)))
  if (recovering) return Option.some("Recover" as const)
  const pending = yield* reconcilePreparedUpdate(version)
  return Option.isSome(pending) && pending.value.installation._tag === "Unattempted" ? Option.some("Install" as const) : Option.none()
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
  const operation = yield* macStartupUpdateOperation(dirname(dirname(options.resources)), options.version)
  if (Option.isNone(operation)) return
  if (operation.value === "Install") {
    const fs = yield* FileSystem.FileSystem
    const writable = yield* fs.access(dirname(dirname(dirname(options.resources))), { writable: true }).pipe(
      Effect.as(true), Effect.catchAll(() => Effect.succeed(false)))
    if (!writable) {
      yield* Effect.sync(() => { process.stderr.write("The prepared update needs permission to replace this application. Quit Magnitude and install the update from an account with access to the application folder.\n") })
      return
    }
  }
  yield* acquireUpdateInstallationLease(options.stateDirectory)
  return yield* startMacForegroundInstallation({ ...options, operation: operation.value,
    continuation: { _tag: "Foreground", arguments: options.arguments } })
})
