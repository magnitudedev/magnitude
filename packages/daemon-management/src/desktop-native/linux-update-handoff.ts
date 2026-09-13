import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { spawn } from "node:child_process"
import type { Readable } from "node:stream"
import { basename, dirname, isAbsolute, join, resolve } from "node:path"
import { LINUX_DESKTOP_EXECUTABLE_PATH } from "@magnitudedev/release/executables"
import { LinuxPackageUpdateFailed } from "./linux-update-package"

export const LinuxUpdateResult = Schema.Struct({ version: Schema.NonEmptyString, error: Schema.optionalWith(Schema.String, { as: "Option", exact: true }) })
export const LinuxUpdateHandoffRequest = Schema.Struct({ requestPath: Schema.NonEmptyString, stateDirectory: Schema.NonEmptyString, version: Schema.NonEmptyString, showWindow: Schema.Boolean }).pipe(
  Schema.filter(request => isAbsolute(request.stateDirectory) && resolve(request.stateDirectory) === request.stateDirectory
    && basename(request.requestPath) === "request.json"
    && /^prepared-[a-zA-Z0-9]+$/.test(basename(dirname(request.requestPath)))
    && dirname(dirname(request.requestPath)) === join(request.stateDirectory, "application-updates")),
)
export type LinuxUpdateHandoffRequest = typeof LinuxUpdateHandoffRequest.Type
const cli = "/usr/lib/magnitude-desktop/resources/magnitude"
const failed = () => new LinuxPackageUpdateFailed({ message: "Could not start the application update installer." })

/** The inherited pipe closes when Electron exits, before package installation starts. */
export const startLinuxUpdateHandoff = (request: LinuxUpdateHandoffRequest) =>
  Schema.encode(Schema.parseJson(LinuxUpdateHandoffRequest))(request).pipe(Effect.mapError(failed), Effect.flatMap(payload => Effect.async<void, LinuxPackageUpdateFailed>(resume => {
    const helper = spawn(cli, ["_complete-application-update"], { detached: true, stdio: ["pipe", "ignore", "ignore", "pipe"] })
    let admitted = false
    const fail = () => {
      if (admitted) return
      helper.kill()
      resume(Effect.fail(failed()))
    }
    helper.once("error", fail)
    helper.once("exit", fail)
    helper.stdin!.on("error", fail)
    let acknowledgement = ""
    const ready = helper.stdio[3] as Readable
    ready.on("error", fail)
    ready.on("data", (bytes: Buffer) => {
      acknowledgement += bytes.toString("utf8")
      if (acknowledgement === "ready\n") {
        admitted = true
        helper.removeListener("exit", fail)
        helper.unref()
        resume(Effect.void)
      } else if (acknowledgement.length >= 6) fail()
    })
    helper.once("spawn", () => helper.stdin!.write(`${payload}\n`, error => { if (error) fail() }))
    // Until readiness, cancellation must stop the helper before releasing its lifetime pipe.
    return Effect.sync(() => { helper.kill(); helper.stdin!.destroy(); ready.destroy() })
  })), Effect.timeoutFail({ duration: "10 seconds", onTimeout: failed }))

/** Runs as the desktop user; only the narrow package operation receives Polkit authorization. */
export const completeLinuxUpdateHandoff = (request: LinuxUpdateHandoffRequest) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* CommandExecutor.CommandExecutor
  if (process.platform !== "linux" || process.getuid?.() === 0) return yield* failed()
  const result = yield* executor.exitCode(Command.make("/usr/bin/pkexec", "--disable-internal-agent", cli,
    "_install-application-update", request.requestPath).pipe(Command.stdout("inherit"), Command.stderr("inherit"))).pipe(Effect.either)
  const error = result._tag === "Left" ? Option.some("System authorization could not be started. Try the update again from your desktop session.")
    : result.right === 0 ? Option.none<string>()
    : Option.some(result.right === 126 ? "The update was cancelled at the system authorization prompt."
      : result.right === 127 ? "System authorization was unavailable. Try the update again from your desktop session."
      : "The package manager could not finish the application update. Check its installation status before retrying.")
  const relaunch = Effect.async<void, LinuxPackageUpdateFailed>(resume => {
    const application = spawn(LINUX_DESKTOP_EXECUTABLE_PATH, request.showWindow ? [] : ["--background"], { detached: true, stdio: "ignore" })
    application.once("error", () => resume(Effect.fail(failed())))
    application.once("spawn", () => { application.unref(); resume(Effect.void) })
  })
  yield* Effect.gen(function* () {
    const path = join(request.stateDirectory, "update-result.json")
    yield* fs.writeFileString(`${path}.tmp`, yield* Schema.encode(Schema.parseJson(LinuxUpdateResult))({ version: request.version, error }), { mode: 0o600 })
    yield* fs.rename(`${path}.tmp`, path)
    // Every handoff has a separate immutable staging directory; cleanup cannot remove a later update.
    yield* fs.remove(dirname(request.requestPath), { recursive: true }).pipe(Effect.ignore)
  }).pipe(Effect.ensuring(relaunch.pipe(Effect.orDie)))
}).pipe(Effect.mapError(failed))
