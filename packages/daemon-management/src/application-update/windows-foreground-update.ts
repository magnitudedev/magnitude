import { Command } from "@effect/platform"
import { Effect, Option, Scope } from "effect"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { acquireApplicationMaintenance } from "../desktop-native/application-owner"
import { acquireUpdateInstallationLease } from "../desktop-native/update-installation-lease"
import { recoverWindowsUpdateDirectory } from "../desktop-native/windows-update-directory"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { WindowsInstallerVerifier } from "../desktop-native/windows-update-signature"
import { ApplicationUpdateFailed } from "./application-update"
import { reconcilePreparedUpdate } from "./prepared-update-installation"

/** Retain installation admission while releasing application ownership before invoking NSIS. */
export const completeWindowsForegroundUpdate = (options: {
  readonly stateDirectory: string
  readonly resources: string
  readonly dataDirectory: string
  readonly version: string
  readonly automatic: boolean
  readonly launcherProtocol: string | undefined
}) => Effect.scoped(Effect.gen(function* () {
  const scope = yield* Effect.scope
  const store = yield* PreparedUpdateStore
  const prepared = yield* Effect.scoped(Effect.gen(function* () {
    yield* acquireApplicationMaintenance(options.stateDirectory)
    yield* recoverWindowsUpdateDirectory(join(options.resources, "desktop-host.node"), options.dataDirectory)
    const pending = yield* reconcilePreparedUpdate(options.version)
    if (Option.isNone(pending)) {
      if (!options.automatic) return yield* new ApplicationUpdateFailed({ message: "There is no prepared application update to install." })
      return Option.none()
    }
    if (options.automatic && pending.value.installation._tag !== "Unattempted") return Option.none()
    if (options.automatic && options.launcherProtocol !== "1") {
      yield* Effect.sync(() => { process.stderr.write("Use the installed magnitude command on PATH to install the prepared update before serving.\n") })
      return Option.none()
    }
    yield* acquireUpdateInstallationLease(options.stateDirectory).pipe(Scope.extend(scope))
    const release = pending.value.release
    const archive = yield* Effect.gen(function* () {
      const archive = yield* store.verify(release)
      const verifier = yield* WindowsInstallerVerifier
      yield* verifier.verify(archive)
      return archive
    }).pipe(Effect.tapError(error => store.recordFailure(release, error.message)))
    yield* store.recordAttempt(release)
    return Option.some({ archive, release })
  }))
  if (Option.isNone(prepared)) return false
  const { archive, release } = prepared.value
  return yield* Effect.gen(function* () {
    // A cwd inside the application also keeps its directory mapped during replacement.
    yield* Effect.try(() => process.chdir(tmpdir()))
    const code = yield* Command.make(archive, "/S").pipe(Command.workingDirectory(tmpdir()),
      Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
    if (code !== 0) return yield* new ApplicationUpdateFailed({ message: "The Windows installer could not finish. Retry with `magnitude update install`." })
    const version = (yield* Command.make(join(options.resources, "magnitude.exe"), "--version").pipe(Command.string, Effect.timeout("10 seconds"))).trim()
    if (version !== release.version) return yield* new ApplicationUpdateFailed({ message: "The installed application does not report the prepared update version. Retry the installation." })
    yield* store.discard
    return true
  }).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
    Effect.tapError(error => store.recordFailure(release, error.message)))
})).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
