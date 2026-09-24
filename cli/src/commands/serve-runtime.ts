import { dirname, resolve } from "node:path"
import { homedir } from "node:os"
import { fileURLToPath } from "node:url"
import { BunContext } from "@effect/platform-bun"
import { Deferred, Effect, Option, Runtime } from "effect"
import { BunSqliteDriverLayer, bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { applicationNativeHostPath, applicationStateDirectory, nativeHostLayer, resolveApplicationProfile,
  resolveInstalledApplicationRuntime, runHeadlessApplication, type ApplicationRuntime } from "@magnitudedev/daemon-management/desktop-native"
import { isDevelopmentBuild } from "../runtime/environment"
import { runWindowsInstalledUpdate } from "../server/windows-startup-update"
import { initializeServeUpdates, prepareServeStartup } from "../server/serve-updates"

export const runServe = () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const stopped = yield* Deferred.make<void>()
  const effects = yield* Effect.runtime<never>()
  const stop = () => Runtime.runSync(effects)(Deferred.succeed(stopped, undefined))
  const signals = process.platform === "win32" ? ["SIGINT", "SIGTERM", "SIGBREAK"] as const : ["SIGINT", "SIGTERM", "SIGHUP"] as const
  yield* Effect.acquireRelease(Effect.sync(() => { for (const signal of signals) process.on(signal, stop) }),
    () => Effect.sync(() => { for (const signal of signals) process.removeListener(signal, stop) }))
  const runtime: ApplicationRuntime = isDevelopmentBuild()
    ? { _tag: "Development", repository: resolve(dirname(fileURLToPath(import.meta.url)), "../../..") }
    : yield* resolveInstalledApplicationRuntime(process.execPath, process.platform)
  const profile = resolveApplicationProfile({ runtime, home: homedir(), platform: process.platform, acceptance: false, environment: process.env })
  const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: profile.dataDirectory, override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR) })
  const addon = applicationNativeHostPath(runtime, process.platform, process.arch)
  if (yield* runWindowsInstalledUpdate(runtime, profile, stateDirectory, true)) {
    yield* Effect.sync(() => { process.exitCode = 75 })
    return
  }
  yield* runHeadlessApplication({ runtime, profile, stateDirectory, home: homedir(), environment: process.env,
    prepareStartup: prepareServeStartup(runtime, profile, addon, stateDirectory).pipe(
      Effect.provide(process.platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon))),
    initializeUpdates: initializeServeUpdates(runtime, profile, addon),
    updateReady: version => Effect.sync(() => { process.stderr.write(`Magnitude ${version} is ready to install. Stop the server, then run: magnitude serve\n`) }),
    stopping: reason => Effect.sync(() => { process.stderr.write(reason === "DesktopTakeover"
      ? "The desktop app was opened and is taking over. Stopping the headless server.\n"
      : "Stopping the Magnitude server.\n") }),
    stop: Deferred.await(stopped), observe: state => state._tag === "Ready"
      ? Effect.sync(() => { process.stderr.write(`Magnitude is serving at ${profile.endpoint}. Press Ctrl+C to stop.\n`) }) : Effect.void,
  }).pipe(Effect.provide(process.platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon)))
})).pipe(Effect.provide([BunContext.layer, BunSqliteDriverLayer]), Effect.catchAll(error => Effect.sync(() => {
  process.stderr.write(`${error.message}\n`)
  process.exitCode = 1
}))))
