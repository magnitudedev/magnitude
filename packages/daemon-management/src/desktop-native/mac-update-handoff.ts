import { Context, Effect, Schema, type Scope } from "effect"
import { spawn } from "node:child_process"
import { isAbsolute, join, resolve } from "node:path"

const Path = Schema.NonEmptyString.pipe(Schema.filter(path => isAbsolute(path) && resolve(path) === path && !path.includes("\0")))
export const MacUpdateHandoffRequest = Schema.Struct({
  bundle: Path, stateDirectory: Path, helperDirectory: Path, showWindow: Schema.Boolean,
}).pipe(Schema.filter(request => request.helperDirectory.startsWith(join(request.stateDirectory, "update-helpers") + "/")
  && /^helper-[a-f0-9-]{36}$/.test(request.helperDirectory.slice(request.helperDirectory.lastIndexOf("/") + 1))
  && request.bundle.endsWith(".app")))
export type MacUpdateHandoffRequest = typeof MacUpdateHandoffRequest.Type
export class MacUpdateHandoffFailed extends Schema.TaggedError<MacUpdateHandoffFailed>()("MacUpdateHandoffFailed", { message: Schema.String }) {}
export interface MacUpdateHandoff {
  readonly start: (request: MacUpdateHandoffRequest) => Effect.Effect<{ readonly commit: Effect.Effect<void> }, MacUpdateHandoffFailed, Scope.Scope>
}
export const MacUpdateHandoff = Context.GenericTag<MacUpdateHandoff>("desktop/MacUpdateHandoff")
const failed = () => new MacUpdateHandoffFailed({ message: "Could not prepare the application update relaunch." })

/** Acquire before staging. Cancellation stops the helper unless native staging has committed. */
export const startMacUpdateHandoff = (request: MacUpdateHandoffRequest) => Effect.gen(function* () {
  const payload = yield* Schema.encode(Schema.parseJson(MacUpdateHandoffRequest))(request).pipe(Effect.mapError(failed))
  let committed = false
  const helper = yield* Effect.acquireRelease(Effect.try({
    try: () => spawn(join(request.helperDirectory, "magnitude-update"), ["_complete-mac-application-update"], {
      detached: true, stdio: ["pipe", "pipe", "ignore"],
    }),
    catch: failed,
  }), child => Effect.sync(() => { if (!committed) { child.kill(); child.stdin.destroy(); child.stdout.destroy() } }))
  yield* Effect.async<void, MacUpdateHandoffFailed>(resume => {
    const error = () => { cleanup(); resume(failed()) }
    let acknowledgement = ""
    const data = (bytes: Buffer) => {
      acknowledgement += bytes.toString("utf8")
      if (acknowledgement === "ready\n") { cleanup(); resume(Effect.void) }
      else if (acknowledgement.length >= 6) error()
    }
    const cleanup = () => { helper.removeListener("error", error); helper.removeListener("exit", error); helper.stdout.removeListener("data", data) }
    helper.once("error", error); helper.once("exit", error); helper.stdout.on("data", data)
    helper.stdin.on("error", error); helper.stdout.on("error", error)
    helper.stdin.write(`${payload}\n`, writeError => { if (writeError) error() })
    return Effect.sync(cleanup)
  }).pipe(Effect.timeoutFail({ duration: "10 seconds", onTimeout: failed }))
  return { commit: Effect.sync(() => { committed = true; helper.unref() }) }
})

/** The helper inherits only the retiring app's environment; no profile or intent is persisted. */
export const relaunchMacAfterUpdate = (request: MacUpdateHandoffRequest) => Effect.async<void, MacUpdateHandoffFailed>(resume => {
  const child = spawn(join(request.bundle, "Contents/MacOS/Magnitude"), request.showWindow ? [] : ["--background"], { detached: true, stdio: "ignore" })
  child.once("error", () => resume(failed()))
  child.once("spawn", () => { child.unref(); resume(Effect.void) })
})
