import { Context, Effect } from "effect"
import { FileSystem } from "@effect/platform"
import { join } from "node:path"
import { app } from "electron"
import { LINUX_DESKTOP_EXECUTABLE_PATH } from "@magnitudedev/release/executables"
import { LoginStartupFailed, type LoginStartupState } from "@magnitudedev/sdk/desktop-host"
import { makeXdgLoginStartup } from "@magnitudedev/daemon-management/desktop-native"

export const WINDOWS_APPLICATION_ID = "dev.magnitude.desktop"

export interface LoginStartup {
  readonly read: Effect.Effect<LoginStartupState, LoginStartupFailed>
  readonly set: (enabled: boolean) => Effect.Effect<LoginStartupState, LoginStartupFailed>
}
export const LoginStartup = Context.GenericTag<LoginStartup>("desktop/LoginStartup")

/** The OS is the preference authority; this adapter retains no settings receipt. */
export const makeLoginStartup = (isolatedProfile: boolean) => Effect.gen(function* () {
  if (isolatedProfile || !app.isPackaged) {
    const message = "Launch at login is disabled in this development or test build. Install Magnitude to enable it."
    return LoginStartup.of({ read: Effect.succeed({ _tag: "Unavailable", message }), set: () => Effect.fail(new LoginStartupFailed({ message })) })
  }
  if (process.platform === "linux") return yield* makeXdgLoginStartup({ executable: LINUX_DESKTOP_EXECUTABLE_PATH })
  const options = process.platform === "darwin" ? { type: "mainAppService" as const } : { path: `"${process.execPath}"`, args: ["--background"] }
  const read = Effect.try({ try: (): LoginStartupState => {
    const settings = app.getLoginItemSettings(options)
    if (process.platform === "darwin") {
      if (settings.status === "requires-approval") return { _tag: "RequiresApproval" }
      return { _tag: settings.status === "enabled" ? "Enabled" : "Disabled" }
    }
    // openAtLogin compares the complete registered command, including --background.
    // launchItems.args omits Chromium switches; use the same named entry only for approval.
    const enabled = settings.openAtLogin && settings.launchItems.some(item => item.name === WINDOWS_APPLICATION_ID && item.scope === "user" && item.enabled)
    return { _tag: enabled ? "Enabled" : "Disabled" }
  }, catch: error => new LoginStartupFailed({ message: `Could not read login startup: ${String(error)}` }) })
  const lock = yield* Effect.makeSemaphore(1)
  return LoginStartup.of({ read, set: enabled => lock.withPermits(1)(Effect.try({
    try: () => app.setLoginItemSettings({ ...options, openAtLogin: enabled }),
    catch: error => new LoginStartupFailed({ message: `Could not change login startup: ${String(error)}` }),
  }).pipe(Effect.zipRight(read), Effect.flatMap(state => state._tag !== (enabled ? "Enabled" : "Disabled") && !(enabled && state._tag === "RequiresApproval")
    ? Effect.fail(new LoginStartupFailed({ message: "The operating system did not apply the login-startup change. Check your system startup settings." }))
    : Effect.succeed(state)))) })
})

/** First-run policy, separate from observational Settings reads. The marker records an
 * attempt, never the user's preference; macOS remains authoritative after that attempt. */
export const initializeLoginStartup = (service: LoginStartup, stateDirectory: string, isolatedProfile: boolean) => Effect.gen(function* () {
  if (isolatedProfile || !app.isPackaged || process.platform !== "darwin" || !app.isInApplicationsFolder()) return
  const fs = yield* FileSystem.FileSystem
  const marker = join(stateDirectory, "login-startup-attempted")
  if (yield* fs.exists(marker)) return
  const settings = yield* Effect.try({
    try: () => app.getLoginItemSettings({ type: "mainAppService" }),
    catch: error => new LoginStartupFailed({ message: `Could not inspect initial login startup: ${String(error)}` }),
  })
  // Claim before registering: a crash or later opt-out must never cause re-enablement.
  yield* fs.writeFileString(marker, "", { flag: "wx", mode: 0o600 })
  if (settings.status === "not-found") yield* service.set(true)
}).pipe(Effect.mapError(error => error instanceof LoginStartupFailed ? error : new LoginStartupFailed({ message: `Could not initialize login startup: ${String(error)}` })))
