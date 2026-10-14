import { Command } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { linuxUpdateFailureReason } from "../desktop-native/linux-update-package"
import { TerminalCommand } from "../desktop-native/terminal-command"
import { ApplicationUpdateFailed } from "./application-update"

const installedCli = "/usr/lib/magnitude-desktop/resources/magnitude"
const ForegroundFailureReason = Schema.Literal("authorization", "verify", "install")
/** `Server` is the system service installing on its own when idle; it never prompts, and `update install` is refused there. */
export type LinuxUpdateInvoker = "Terminal" | "Unattended" | "Server"
const ForegroundInvoker = Schema.Literal("Terminal", "Unattended", "Server")
const terminalMessage: Record<typeof ForegroundFailureReason.Type, string> = {
  authorization: "System authorization was refused or unavailable. Retry `magnitude update install` from a terminal.",
  verify: "The downloaded update failed verification. Run `magnitude update download` to download it again.",
  install: "The package manager could not install the update. Check the package manager status before retrying `magnitude update install`.",
}
const serverMessage: Record<typeof ForegroundFailureReason.Type, string> = {
  authorization: "The server's permission to run the update installer was refused. Run `magnitude server setup` to restore it, then `magnitude update download` to retry.",
  verify: "The downloaded update failed verification. Run `magnitude update download` to download it again.",
  install: "The package manager could not install the update. Check the package manager status, then run `magnitude update download` to retry; the server installs it once it is idle.",
}
class LinuxForegroundUpdateFailed extends Schema.TaggedError<LinuxForegroundUpdateFailed>()("LinuxForegroundUpdateFailed", {
  reason: ForegroundFailureReason,
  invoker: ForegroundInvoker,
}) {
  get message() { return (this.invoker === "Server" ? serverMessage : terminalMessage)[this.reason] }
}

/** Caller retains maintenance and installation admission, with no running service or shared installation lease. */
export const completeLinuxForegroundUpdate = (dataDirectory: string, invoker: LinuxUpdateInvoker) => Effect.gen(function* () {
  const failed = (reason: typeof ForegroundFailureReason.Type) => new LinuxForegroundUpdateFailed({ reason, invoker })
  const store = yield* PreparedUpdateStore
  const pending = yield* store.read
  if (Option.isNone(pending)) return yield* new ApplicationUpdateFailed({ message: "There is no prepared application update to install." })
  const release = pending.value.release
  yield* store.verify(release).pipe(Effect.tapError(error => store.recordFailure(release, "verify", error.message)))
  yield* store.recordAttempt(release)
  return yield* Effect.gen(function* () {
    // In this terminal's session, so sudo can ask for the password here; the installer watches stdin for our exit.
    const status = yield* (yield* TerminalCommand).run("/usr/bin/sudo", [...(invoker === "Terminal" ? [] : ["-n"]), "--", installedCli,
      "_install-application-update", join(dataDirectory, "updates", "update.json"), "--parent-stdin"], { stdin: "lifetime" }).pipe(
      Effect.mapError(() => failed("authorization")))
    if (status !== 0) return yield* failed(linuxUpdateFailureReason(status))
    const version = (yield* Command.make(installedCli, "--version").pipe(Command.string, Effect.timeout("10 seconds"),
      Effect.mapError(() => failed("install")))).trim()
    if (version !== release.version) return yield* failed("install")
    yield* store.complete(release)
    return version
  }).pipe(Effect.tapError(error => error._tag === "LinuxForegroundUpdateFailed" ? store.recordFailure(release, error.reason, error.message) : Effect.void),
    Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
