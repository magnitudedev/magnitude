import { Command } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { linuxUpdateFailureReason } from "../desktop-native/linux-update-package"
import { TerminalCommand } from "../desktop-native/terminal-command"
import { ApplicationUpdateFailed } from "./application-update"

const installedCli = "/usr/lib/magnitude-desktop/resources/magnitude"
const ForegroundFailureReason = Schema.Literal("authorization", "verify", "install")
const failureMessage: Record<typeof ForegroundFailureReason.Type, string> = {
  authorization: "System authorization was refused or unavailable. Retry `magnitude update install` from a terminal.",
  verify: "The downloaded update failed verification. Run `magnitude update download` to download it again.",
  install: "The package manager could not install the update. Check the package manager status before retrying `magnitude update install`.",
}
class LinuxForegroundUpdateFailed extends Schema.TaggedError<LinuxForegroundUpdateFailed>()("LinuxForegroundUpdateFailed", {
  reason: ForegroundFailureReason,
}) {
  get message() { return failureMessage[this.reason] }
}
const foregroundFailed = (code: number) => new LinuxForegroundUpdateFailed({ reason: linuxUpdateFailureReason(code) })

/** Caller retains maintenance and installation admission, with no running service or shared installation lease. */
export const completeLinuxForegroundUpdate = (dataDirectory: string, allowAuthorizationPrompt: boolean) => Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const pending = yield* store.read
  if (Option.isNone(pending)) return yield* new ApplicationUpdateFailed({ message: "There is no prepared application update to install." })
  const release = pending.value.release
  yield* store.verify(release).pipe(Effect.tapError(error => store.recordFailure(release, "verify", error.message)))
  yield* store.recordAttempt(release)
  return yield* Effect.gen(function* () {
    // In this terminal's session, so sudo can ask for the password here; the installer watches stdin for our exit.
    const status = yield* (yield* TerminalCommand).run("/usr/bin/sudo", [...(allowAuthorizationPrompt ? [] : ["-n"]), "--", installedCli,
      "_install-application-update", join(dataDirectory, "updates", "update.json"), "--parent-stdin"], { stdin: "lifetime" }).pipe(
      Effect.mapError(() => new LinuxForegroundUpdateFailed({ reason: "authorization" })))
    if (status !== 0) return yield* foregroundFailed(status)
    const version = (yield* Command.make(installedCli, "--version").pipe(Command.string, Effect.timeout("10 seconds"),
      Effect.mapError(() => new LinuxForegroundUpdateFailed({ reason: "install" })))).trim()
    if (version !== release.version) return yield* new LinuxForegroundUpdateFailed({ reason: "install" })
    yield* store.complete(release)
    return version
  }).pipe(Effect.tapError(error => error._tag === "LinuxForegroundUpdateFailed" ? store.recordFailure(release, error.reason, error.message) : Effect.void),
    Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
