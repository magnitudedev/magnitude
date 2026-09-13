import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { spawn } from "node:child_process"
import { win32 } from "node:path"
import { SignedUpdateManifest } from "@magnitudedev/release/hosted-update"

const absolutePath = Schema.NonEmptyString.pipe(Schema.filter(path => win32.isAbsolute(path)
  && win32.resolve(path) === path && !path.startsWith("\\\\") && !path.includes("\0")))
export const WindowsUpdateHandoffRequest = Schema.Struct({
  stateDirectory: absolutePath,
  preparedDirectory: absolutePath,
  applicationPath: absolutePath,
  version: Schema.NonEmptyString,
  envelope: SignedUpdateManifest,
  showWindow: Schema.Boolean,
}).pipe(Schema.filter(request => win32.dirname(request.preparedDirectory) === win32.join(request.stateDirectory, "application-updates")
  && /^prepared-[a-f0-9-]{36}$/.test(win32.basename(request.preparedDirectory))
  && win32.basename(request.applicationPath) === "Magnitude.exe"))
export type WindowsUpdateHandoffRequest = typeof WindowsUpdateHandoffRequest.Type
export const WindowsUpdateResult = Schema.Struct({
  request: WindowsUpdateHandoffRequest,
  error: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export class WindowsUpdateHandoffFailed extends Schema.TaggedError<WindowsUpdateHandoffFailed>()("WindowsUpdateHandoffFailed", {
  message: Schema.String,
}) {}
const failed = () => new WindowsUpdateHandoffFailed({ message: "Could not start the application update installer." })

/** The copied CLI survives replacement; its stdin remains open until the desktop exits. */
export const startWindowsUpdateHandoff = (request: WindowsUpdateHandoffRequest) =>
  Schema.encode(Schema.parseJson(WindowsUpdateHandoffRequest))(request).pipe(Effect.mapError(failed), Effect.flatMap(payload => Effect.async<void, WindowsUpdateHandoffFailed>(resume => {
    const helper = spawn(win32.join(request.preparedDirectory, "magnitude-update.exe"), ["_complete-windows-application-update"], {
      detached: true, windowsHide: true, stdio: ["pipe", "pipe", "ignore"],
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
  if (process.platform !== "win32" || win32.resolve(process.execPath) !== win32.join(request.preparedDirectory, "magnitude-update.exe")) {
    return yield* failed()
  }
  const fs = yield* FileSystem.FileSystem
  const executor = yield* CommandExecutor.CommandExecutor
  const result = yield* executor.exitCode(Command.make(win32.join(request.preparedDirectory, "magnitude-setup.exe"), "/S")).pipe(Effect.either)
  const error = result._tag === "Left" ? Option.some("The application update installer could not be started.")
    : result.right === 0 ? Option.none<string>()
    : Option.some("The application update installer could not finish. Check for updates to retry.")
  const relaunch = Effect.async<void, WindowsUpdateHandoffFailed>(resume => {
    const application = spawn(request.applicationPath, request.showWindow ? [] : ["--background"], {
      detached: true, windowsHide: true, stdio: "ignore",
    })
    application.once("error", () => resume(Effect.fail(failed())))
    application.once("spawn", () => { application.unref(); resume(Effect.void) })
  })
  yield* Effect.gen(function* () {
    const path = win32.join(request.stateDirectory, "update-result.json")
    yield* fs.writeFileString(`${path}.tmp`, yield* Schema.encode(Schema.parseJson(WindowsUpdateResult))({ request, error }))
    yield* fs.rename(`${path}.tmp`, path)
    // The reopened desktop removes this completed staging directory after this executable exits.
  }).pipe(Effect.ensuring(relaunch.pipe(Effect.orDie)))
}).pipe(Effect.mapError(failed))
