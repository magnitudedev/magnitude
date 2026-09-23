import { UpdateContinuation } from "../application-update/update-continuation"
import { Command, CommandExecutor } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { spawn } from "node:child_process"
import { win32 } from "node:path"
import { recordPreparedUpdateFailure } from "./prepared-update"
import { UpdateRelease } from "@magnitudedev/release/hosted-update"

const absolutePath = Schema.NonEmptyString.pipe(Schema.filter(path => win32.isAbsolute(path)
  && win32.resolve(path) === path && !path.startsWith("\\\\") && !path.includes("\0")))
export const WindowsUpdateHandoffRequest = Schema.Struct({
  stateDirectory: absolutePath,
  helperDirectory: absolutePath,
  applicationPath: absolutePath,
  dataDirectory: absolutePath,
  release: UpdateRelease,
  continuation: UpdateContinuation,
}).pipe(Schema.filter(request => win32.dirname(request.helperDirectory) === win32.join(request.stateDirectory, "update-helpers")
  && /^helper-[a-f0-9-]{36}$/.test(win32.basename(request.helperDirectory))
  && win32.basename(request.applicationPath) === "Magnitude.exe"))
export type WindowsUpdateHandoffRequest = typeof WindowsUpdateHandoffRequest.Type
export class WindowsUpdateHandoffFailed extends Schema.TaggedError<WindowsUpdateHandoffFailed>()("WindowsUpdateHandoffFailed", {
  message: Schema.String,
}) {}
const failed = () => new WindowsUpdateHandoffFailed({ message: "Could not start the application update installer." })

/** The copied CLI survives replacement; its stdin remains open until the desktop exits. */
export const startWindowsUpdateHandoff = (request: WindowsUpdateHandoffRequest) =>
  Schema.encode(Schema.parseJson(WindowsUpdateHandoffRequest))(request).pipe(Effect.mapError(failed), Effect.flatMap(payload => Effect.async<void, WindowsUpdateHandoffFailed>(resume => {
    const helper = spawn(win32.join(request.helperDirectory, "magnitude.exe"), ["_complete-windows-application-update"], {
      cwd: request.helperDirectory, detached: true, windowsHide: true, stdio: ["pipe", "pipe", "ignore"],
    })
    let admitted = false
    const fail = () => {
      if (admitted) return
      helper.kill()
      resume(Effect.fail(failed()))
    }
    helper.once("error", fail)
    helper.once("exit", fail)
    helper.stdin!.on("error", fail)
    helper.stdout!.on("error", fail)
    let acknowledgement = ""
    helper.stdout!.on("data", (bytes: Buffer) => {
      acknowledgement += bytes.toString("utf8")
      if (acknowledgement === "ready\n") {
        admitted = true
        helper.removeListener("exit", fail)
        helper.unref()
        resume(Effect.void)
      } else if (acknowledgement.length >= 6) fail()
    })
    helper.once("spawn", () => helper.stdin!.write(`${payload}\n`, error => { if (error) fail() }))
    return Effect.sync(() => { helper.kill(); helper.stdin!.destroy(); helper.stdout!.destroy() })
  })), Effect.timeoutFail({ duration: "10 seconds", onTimeout: failed }))

/** Per-user installation needs no elevation. Installer exit, not spawn, is the completion boundary. */
export const completeWindowsUpdateHandoff = (request: WindowsUpdateHandoffRequest) => Effect.gen(function* () {
  if (process.platform !== "win32" || win32.resolve(process.execPath) !== win32.join(request.helperDirectory, "magnitude.exe")) {
    return yield* failed()
  }
  const executor = yield* CommandExecutor.CommandExecutor
  const result = yield* executor.exitCode(Command.make(win32.join(request.dataDirectory, "updates", "magnitude-setup.exe"), "/S")
    .pipe(Command.workingDirectory(request.helperDirectory))).pipe(Effect.either)
  const error = result._tag === "Left" ? Option.some("The application update installer could not be started.")
    : result.right === 0 ? Option.none<string>()
    : Option.some("The application update installer could not finish. Check for updates to retry.")
  if (Option.isSome(error)) yield* recordPreparedUpdateFailure(request.dataDirectory, request.release, error.value)
}).pipe(Effect.mapError(failed))

/** Called only after the helper releases its native installation lease. */
export const relaunchWindowsAfterUpdate = (request: WindowsUpdateHandoffRequest) => {
  const continuation = request.continuation
  if (continuation._tag === "Caller") return Effect.void
  return Effect.async<void, WindowsUpdateHandoffFailed>(resume => {
    const application = spawn(request.applicationPath, continuation.showWindow ? [] : ["--background"], {
      cwd: win32.dirname(request.applicationPath), detached: true, windowsHide: true, stdio: "ignore",
    })
    application.once("error", () => resume(Effect.fail(failed())))
    application.once("spawn", () => { application.unref(); resume(Effect.void) })
})
}
