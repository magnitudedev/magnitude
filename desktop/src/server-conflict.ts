import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { BrowserWindow, ipcMain, nativeTheme, shell } from "electron"
import { Deferred, Duration, Effect, Option, Runtime, Schedule, Schema } from "effect"
import { join } from "node:path"
import { slate } from "@magnitudedev/client-common"
import { MAC_SERVER_PLIST, installedServerCli, isServerProfileActive, requestApplication } from "@magnitudedev/daemon-management/desktop-native"
import { windowChrome } from "./window-chrome"

export const SERVER_ADDRESS = "http://localhost:10100"
export type ServerConflictDecision = "Proceed" | "Quit"
class ServerRemovalFailed extends Schema.TaggedError<ServerRemovalFailed>()("ServerRemovalFailed", { message: Schema.String }) {}

/**
 * The server set up by `magnitude server setup` is running on this machine: on Linux the service
 * under the server profile, on macOS the LaunchDaemon serving as this person, which the desktop would
 * otherwise take over only for launchd to start it again.
 */
export const isServerRunning = (personalStateDirectory: string) => Effect.gen(function* () {
  if (process.platform === "linux") {
    if (!(yield* isServerProfileActive("linux"))) return false
    const executor = yield* CommandExecutor.CommandExecutor
    return (yield* executor.exitCode(Command.make("/usr/bin/systemctl", "is-active", "--quiet", "magnitude.service"))) === 0
  }
  if (process.platform === "darwin") {
    const fs = yield* FileSystem.FileSystem
    if (!(yield* fs.exists(MAC_SERVER_PLIST))) return false
    const snapshot = yield* requestApplication(join(personalStateDirectory, "application.sock"), "Observe").pipe(Effect.option)
    return Option.isSome(snapshot) && snapshot.value.owner._tag === "Headless"
  }
  return false
}).pipe(Effect.orElseSucceed(() => false), Effect.provide(NodeContext.layer))

/** Removes the server with the platform's administrator authorization; the server keeps its data. */
const removeServer = Effect.gen(function* () {
  const cli = installedServerCli(process.platform)
  if (Option.isNone(cli)) return yield* new ServerRemovalFailed({ message: "Server mode isn't available here." })
  const executor = yield* CommandExecutor.CommandExecutor
  const command = process.platform === "darwin"
    ? Command.make("/usr/bin/osascript", "-e", `do shell script quoted form of "${cli.value}" & " _server-remove" with administrator privileges`)
    : Command.make("/usr/bin/pkexec", cli.value, "_server-remove")
  const code = yield* executor.exitCode(command).pipe(Effect.orElseSucceed(() => -1))
  if (code !== 0) return yield* new ServerRemovalFailed({ message: "The server was not stopped. Authorization is needed to stop it." })
}).pipe(Effect.provide(NodeContext.layer))

/**
 * While the server runs, the whole window is this one screen. Keep opens the server in the browser
 * and quits; Stop removes the server, then the app starts. If the server stops, the app continues.
 */
export const resolveServerConflict = (options: {
  readonly personalStateDirectory: string
  readonly pageUrl: string
  readonly preload: string
  readonly icon: string
}) => Effect.gen(function* () {
  const running = isServerRunning(options.personalStateDirectory)
  if (!(yield* running)) return "Proceed" as const
  const decision = yield* Deferred.make<ServerConflictDecision>()
  yield* Effect.acquireRelease(Effect.sync(() => {
    const dark = nativeTheme.shouldUseDarkColors
    const value = new BrowserWindow({ ...windowChrome(process.platform, dark), width: 720, height: 520, resizable: false, show: false,
      title: "Magnitude", icon: options.icon, backgroundColor: dark ? slate[925] : slate[50],
      webPreferences: { preload: options.preload, contextIsolation: true, nodeIntegration: false, sandbox: false } })
    value.once("ready-to-show", () => value.show())
    value.on("closed", () => { Effect.runFork(Deferred.succeed(decision, "Quit")) })
    void value.loadURL(options.pageUrl)
    return value
  }), value => Effect.sync(() => { if (!value.isDestroyed()) value.destroy() }))
  const runtime = yield* Effect.runtime<never>()
  yield* Effect.acquireRelease(Effect.sync(() => {
    ipcMain.handle("server-conflict:keep", () => Runtime.runPromise(runtime)(Effect.promise(() => shell.openExternal(SERVER_ADDRESS)).pipe(
      Effect.zipRight(Deferred.succeed(decision, "Quit")), Effect.as({ ok: true }))))
    ipcMain.handle("server-conflict:stop", () => Runtime.runPromise(runtime)(removeServer.pipe(
      Effect.zipRight(running.pipe(Effect.repeat({ until: value => !value, schedule: Schedule.spaced(Duration.millis(500)) }), Effect.timeout("30 seconds"), Effect.ignore)),
      Effect.zipRight(Deferred.succeed(decision, "Proceed")), Effect.as({ ok: true as const }),
      Effect.catchAll(error => Effect.succeed({ ok: false as const, message: error.message })))))
  }), () => Effect.sync(() => { ipcMain.removeHandler("server-conflict:keep"); ipcMain.removeHandler("server-conflict:stop") }))
  // The server stopping by itself also lets the app continue.
  yield* running.pipe(Effect.repeat({ until: value => !value, schedule: Schedule.spaced(Duration.seconds(2)) }),
    Effect.zipRight(Deferred.succeed(decision, "Proceed")), Effect.forkScoped)
  return yield* Deferred.await(decision)
}).pipe(Effect.scoped)
