import { dirname, resolve } from "node:path"
import { homedir } from "node:os"
import { fileURLToPath } from "node:url"
import { BunContext } from "@effect/platform-bun"
import { Deferred, Effect, Option, Ref, Runtime, Schema } from "effect"
import { BunSqliteDriverLayer } from "@magnitudedev/storage/sqlite/bun"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { applicationNativeHostPath, applicationStateDirectory, nativeHostLayer, resolveApplicationProfile,
  resolveInstalledApplicationRuntime, runHeadlessApplication, type ApplicationRuntime } from "@magnitudedev/daemon-management/desktop-native"
import { isDevelopmentBuild } from "../runtime/environment"
import { idleInstallationSystem, initializeServeUpdates, makeIdleInstallation, prepareServeStartup } from "../server/serve-updates"
import { resolveServeOwner } from "../server/serve-owner"
import { readServerReach, renderServeReady } from "../server/server-reach"

class ServeRefused extends Schema.TaggedError<ServeRefused>()("ServeRefused", { message: Schema.String }) {}

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
  const { server, owner } = yield* resolveServeOwner
  if (server && owner !== "service") return yield* new ServeRefused({ message: "Magnitude runs as a server on this machine, so it is already serving. "
    + "Run `magnitude status` to see how to reach it, or `magnitude server remove` to stop the server." })
  const profile = resolveApplicationProfile({ runtime, home: homedir(), platform: process.platform, acceptance: false, environment: process.env, server })
  const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: profile.dataDirectory, override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR) })
  const addon = applicationNativeHostPath(runtime, process.platform, process.arch)
  const ready = yield* Ref.make(false)
  // Unix servers install downloaded updates at idle points; Windows serve only prepares them.
  const idle = runtime._tag === "Installed" && process.platform !== "win32"
    ? Option.some(yield* makeIdleInstallation.pipe(Effect.provide(idleInstallationSystem({ runtime, profile, stateDirectory, addon, owner,
      notify: line => Effect.sync(() => { process.stderr.write(`${line}\n`) }) }))))
    : Option.none()
  // A restart request re-admits the owner in place, so settings read at service start take effect.
  for (;;) {
    yield* Ref.set(ready, false)
    const stop = yield* runHeadlessApplication({ runtime, profile, stateDirectory, home: homedir(), environment: process.env,
    prepareStartup: prepareServeStartup(runtime, profile, stateDirectory).pipe(
      Effect.provide(process.platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon))),
    initializeUpdates: initializeServeUpdates(runtime, profile, addon, owner),
    ...(Option.isSome(idle) ? { installWhenIdle: idle.value.installWhenIdle, startupFailed: idle.value.startupFailed } : {}),
    stopping: reason => Effect.sync(() => { process.stderr.write(reason === "DesktopTakeover"
      ? "The desktop app was opened and is taking over. Stopping the headless server.\n"
      : reason === "Restart" ? "Restarting the Magnitude server.\n"
      : reason === "InstallUpdate" ? "Stopping the service to install the downloaded update; clients reconnect when it is back.\n"
      : "Stopping the Magnitude server.\n") }),
    stop: Deferred.await(stopped), observe: state => Ref.getAndSet(ready, state._tag === "Ready").pipe(Effect.flatMap(wasReady =>
      state._tag === "Ready" && !wasReady
        ? readServerReach({ endpoint: profile.endpoint, dataDirectory: profile.dataDirectory, service: owner === "service" }).pipe(
          Effect.flatMap(reach => Effect.sync(() => { process.stderr.write(renderServeReady(reach)) })))
        : Effect.void)),
  }).pipe(Effect.provide(process.platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon)))
    // Installation replaces this process on success; otherwise the current version serves again.
    if (stop === "InstallUpdate" && Option.isSome(idle)) yield* idle.value.install
    else if (stop !== "Restart") return
  }
})).pipe(Effect.provide([BunContext.layer, BunSqliteDriverLayer]), Effect.catchAll(error => Effect.sync(() => {
  process.stderr.write(`${error.message}\n`)
  process.exitCode = 1
}))))
