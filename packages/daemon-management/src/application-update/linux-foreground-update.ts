import { Command } from "@effect/platform"
import { Deferred, Effect, Option, Stream } from "effect"
import { join } from "node:path"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { ApplicationUpdateFailed } from "./application-update"

const installedCli = "/usr/lib/magnitude-desktop/resources/magnitude"

/** Caller retains maintenance and installation admission, with no running service or shared installation lease. */
export const completeLinuxForegroundUpdate = (dataDirectory: string, allowAuthorizationPrompt: boolean) => Effect.scoped(Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const pending = yield* store.read
  if (Option.isNone(pending)) return yield* new ApplicationUpdateFailed({ message: "There is no prepared application update to install." })
  const release = pending.value.release
  yield* store.verify(release)
  yield* store.recordAttempt(release)
  const lifetime = yield* Deferred.make<void>()
  yield* Effect.addFinalizer(() => Deferred.succeed(lifetime, undefined))
  return yield* Effect.gen(function* () {
    const status = yield* Command.make("/usr/bin/sudo", ...(allowAuthorizationPrompt ? [] : ["-n"]), "--", installedCli,
      "_install-application-update", join(dataDirectory, "updates", "update.json"), "--parent-stdin").pipe(
      Command.stdin(Stream.fromEffect(Deferred.await(lifetime)).pipe(Stream.drain)), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
    if (status !== 0) return yield* new ApplicationUpdateFailed({
      message: "System authorization or package installation failed. Check the package manager status before retrying `magnitude update install`.",
    })
    const version = (yield* Command.make(installedCli, "--version").pipe(Command.string, Effect.timeout("10 seconds"))).trim()
    if (version !== release.version) return yield* new ApplicationUpdateFailed({ message: "The installed application does not report the prepared update version. Check the package manager status before retrying." })
    yield* store.discard
    return version
  }).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
    Effect.tapError(error => store.recordFailure(release, error.message)))
})).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
