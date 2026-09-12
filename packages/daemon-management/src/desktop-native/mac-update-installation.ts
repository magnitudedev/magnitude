import { execFile } from "node:child_process"
import { realpath } from "node:fs/promises"
import { join } from "node:path"
import { Context, Effect, Layer, Schema } from "effect"

export class MacInstallationObservationFailed extends Schema.TaggedError<MacInstallationObservationFailed>()("MacInstallationObservationFailed", {
  message: Schema.String,
}) {}
export interface MacApplicationInstallation {
  readonly isInstalling: (bundle: string) => Effect.Effect<boolean, MacInstallationObservationFailed>
}
export const MacApplicationInstallation = Context.GenericTag<MacApplicationInstallation>("desktop/MacApplicationInstallation")
const Result = Schema.Struct({ code: Schema.Int, stdout: Schema.String })
const command = (executable: string, args: readonly string[]) => Effect.async<typeof Result.Type, MacInstallationObservationFailed>(resume => {
  const child = execFile(executable, [...args], { encoding: "utf8", timeout: 5000, maxBuffer: 128 * 1024 }, (error, stdout) => {
    if (error !== null && typeof error.code !== "number") return resume(new MacInstallationObservationFailed({ message: "Could not inspect the native Magnitude updater. Retry the command." }))
    resume(Effect.succeed({ code: typeof error?.code === "number" ? error.code : 0, stdout }))
  })
  return Effect.sync(() => child.kill())
})

/** A retained, inactive launchd registration is not an installation in progress. */
export const macUpdateJobIsActive = (output: string, executable: string) => Effect.gen(function* () {
  const program = output.match(/^\s*program = (.+)$/m)?.[1]?.trim()
  const state = output.match(/^\s*state = (.+)$/m)?.[1]?.trim()
  if (!program || !state) return yield* new MacInstallationObservationFailed({ message: "The native updater returned an unrecognized job state. Retry after the app update finishes." })
  if (program !== executable) return false
  return state !== "not running"
})

export const NativeMacApplicationInstallation = Layer.succeed(MacApplicationInstallation, {
  isInstalling: bundle => Effect.gen(function* () {
    const canonical = yield* Effect.tryPromise({ try: () => realpath(bundle), catch: () => new MacInstallationObservationFailed({ message: "Could not locate the Magnitude app while checking its update." }) })
    const metadata = yield* command("/usr/bin/plutil", ["-extract", "CFBundleIdentifier", "raw", "-o", "-", "--", join(canonical, "Contents/Info.plist")])
    const identifier = metadata.stdout.trim()
    if (metadata.code !== 0 || !/^[A-Za-z0-9.-]+$/.test(identifier)) return yield* new MacInstallationObservationFailed({ message: "Could not read the installed Magnitude app identity." })
    const job = yield* command("/bin/launchctl", ["print", `gui/${process.getuid!()}/${identifier}.ShipIt`])
    if (job.code === 113) return false
    if (job.code !== 0) return yield* new MacInstallationObservationFailed({ message: "Could not inspect the native Magnitude update job. Retry the command." })
    return yield* macUpdateJobIsActive(job.stdout, join(canonical, "Contents/Frameworks/Squirrel.framework/Resources/ShipIt"))
  }),
})

/** Cold launch waits for the existing installer; it never cancels, replaces, or starts it. */
export const waitForMacApplicationInstallation = (bundle: string) => Effect.gen(function* () {
  const native = yield* MacApplicationInstallation
  while (yield* native.isInstalling(bundle)) yield* Effect.sleep("100 millis")
}).pipe(Effect.timeoutFail({ duration: "2 minutes", onTimeout: () => new MacInstallationObservationFailed({
  message: "Magnitude is still installing an application update. Retry after it finishes; this command has not started another app.",
}) }))
