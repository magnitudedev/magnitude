import { dirname, join, resolve } from "node:path"
import { FileSystem } from "@effect/platform"
import { homedir } from "node:os"
import { fileURLToPath } from "node:url"
import { BunContext } from "@effect/platform-bun"
import { Deferred, Effect, Option, Ref, Runtime, Schema } from "effect"
import { BunSqliteDriverLayer } from "@magnitudedev/storage/sqlite/bun"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { applicationNativeHostPath, applicationOwnershipHeld, applicationStateDirectory, holdContainerData, makeUnixProcessContinuation, nativeHostLayer, observeApplicationOwner, resolveApplicationProfile,
  isLegacyServerMarked, NativeHost, resolveInstalledApplicationRuntime, runHeadlessApplication, servicePortBound, servicePortLingering, servingBlocker, type ApplicationRuntime } from "@magnitudedev/daemon-management/desktop-native"
import { containerImage } from "@magnitudedev/daemon-management/application-update"
import { isDevelopmentBuild } from "../runtime/environment"
import { CLI_VERSION } from "../version"
import { announceAvailableUpdates, announceFailedUpdates, idleInstallationSystem, initializeContainerUpdates, initializeServeUpdates, makeAvailableNotice, makeDownloadedNotice, makeIdleInstallation, makeImageNotice, prepareServeStartup } from "../server/serve-updates"
import { recordModelsBeforeUpdate, recordResidentModels, resumeResidentModels } from "../server/model-resumption"
import { isContainer, renderNodeHint, requireWritableDataDirectory, unopenableRenderNodes } from "../server/container"
import { resolveServeOwner } from "../server/serve-owner"
import { readServerReach, renderServeReady } from "../server/server-reach"
import { runServeLoop } from "../server/serve-turn"
import { serveStopSignals } from "./serve-signals"

class ServeRefused extends Schema.TaggedError<ServeRefused>()("ServeRefused", { message: Schema.String }) {}

export const runServe = () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const stopped = yield* Deferred.make<void>()
  const effects = yield* Effect.runtime<never>()
  const stop = () => Runtime.runSync(effects)(Deferred.succeed(stopped, undefined))
  const signals = yield* serveStopSignals(process.platform)
  yield* Effect.acquireRelease(Effect.sync(() => { for (const signal of signals) process.on(signal, stop) }),
    () => Effect.sync(() => { for (const signal of signals) process.removeListener(signal, stop) }))
  const runtime: ApplicationRuntime = isDevelopmentBuild()
    ? { _tag: "Development", repository: resolve(dirname(fileURLToPath(import.meta.url)), "../../..") }
    : yield* resolveInstalledApplicationRuntime(process.execPath, process.platform)
  const owner = yield* resolveServeOwner(NativeHost.pipe(Effect.flatMap(native => native.isServiceSession),
    Effect.provide(bundledWindowsNative.host), Effect.orElseSucceed(() => false)))
  // A server set up before the CLI-only install is converted by its service's next start; until then nothing serves.
  if (yield* isLegacyServerMarked(process.platform)) return yield* new ServeRefused({ message: "This machine's Magnitude server is being converted into a CLI-only install. "
    + "Its service converts it when it next starts: `sudo systemctl restart magnitude`." })
  const profile = resolveApplicationProfile({ runtime, home: homedir(), platform: process.platform, acceptance: false, environment: process.env })
  const container = yield* isContainer(runtime)
  if (container) yield* requireWritableDataDirectory(profile.dataDirectory)
  const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: profile.dataDirectory, override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR) })
  const addon = applicationNativeHostPath(runtime, process.platform, process.arch)
  if (container) yield* holdContainerData(profile.dataDirectory).pipe(Effect.provide(nativeHostLayer(addon)))
  const ready = yield* Ref.make(false)
  const notify = (line: string) => Effect.sync(() => { process.stderr.write(`${line}\n`) })
  if (container) yield* Effect.forEach(yield* unopenableRenderNodes(), node => notify(renderNodeHint(node)))
  // `serve` installs downloaded updates at idle points when it may install unattended. A Windows desktop
  // installation only prepares them; a container never installs one, because its image is the installation.
  const idle = runtime._tag === "Installed" && (process.platform !== "win32" || runtime.layout === "Standalone") && !container
    ? Option.some(yield* makeIdleInstallation.pipe(Effect.provide(idleInstallationSystem({ runtime, profile, stateDirectory, addon, owner, notify }))))
    : Option.none()
  const updateHooks = Option.isSome(idle) ? { installWhenIdle: idle.value.installWhenIdle, startupFailed: idle.value.startupFailed }
    : runtime._tag === "Installed" && !container ? { updateReady: yield* makeDownloadedNotice(notify) } : {}
  const availableNotice = yield* makeAvailableNotice(notify, runtime._tag === "Installed" && runtime.layout === "Standalone")
  const initializeUpdates = runtime._tag === "Installed" && container
    ? initializeContainerUpdates(runtime, profile, addon, owner, yield* makeImageNotice(notify))
    : initializeServeUpdates(runtime, profile, addon, owner).pipe(Effect.tap(updates => Effect.all([
        announceAvailableUpdates(updates, availableNotice),
        announceFailedUpdates(updates, notify),
      ], { concurrency: "unbounded", discard: true }).pipe(Effect.forkScoped)))
  const fs = yield* FileSystem.FileSystem
  const scope = yield* Effect.scope
  const native = process.platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon)
  /** The desktop app deletes a CLI-only installation it replaces, including the one this process runs from. */
  const replacedByDesktop = runtime._tag === "Installed" && runtime.layout === "Standalone"
    ? FileSystem.FileSystem.pipe(Effect.flatMap(fs => fs.exists(join(runtime.resourcesDirectory,
      process.platform === "win32" ? "magnitude-service.exe" : "magnitude-service"))), Effect.map(exists => !exists), Effect.orElseSucceed(() => false))
    : Effect.succeed(false)
  /**
   * Restarts this command as the desktop's once the desktop has replaced the CLI-only install: Unix execs the
   * `magnitude` command in place, which absorption pointed at the desktop's CLI; Windows exits for its launcher,
   * which re-resolves to the desktop. The native module is loaded now, before absorption deletes its folder.
   */
  const commandLink = join(homedir(), ".magnitude", "bin", "magnitude")
  const continuation = runtime._tag === "Installed" && runtime.layout === "Standalone" && process.platform !== "win32"
    ? yield* makeUnixProcessContinuation(addon).pipe(Effect.option)
    : Option.none()
  const restartAsDesktop = Effect.gen(function* () {
    if (Option.isSome(continuation) && (yield* fs.exists(commandLink).pipe(Effect.orElseSucceed(() => false)))) {
      yield* continuation.value.replace(commandLink, process.argv.slice(2), process.env).pipe(Effect.ignore)
    }
    return yield* Effect.sync(() => process.exit(process.platform === "win32" ? 75 : 1))
  })
  const blocker = servingBlocker(stateDirectory, profile.port).pipe(Effect.provide(native))
  const observeOwner = observeApplicationOwner(stateDirectory).pipe(Effect.provide(native))
  const ownershipHeld = applicationOwnershipHeld(stateDirectory).pipe(Effect.provide(native))
  const lastRecordedDesktop = yield* Ref.make(0)
  yield* runServeLoop({
    stopped,
    blocker,
    observeOwner,
    ownershipHeld,
    portBound: servicePortBound(profile.port),
    portLingering: servicePortLingering(profile.port),
    notify,
    // What this account's desktop app has loaded is loaded again once this server resumes; one that does not answer keeps the last record.
    onDesktopBlocker: Ref.get(lastRecordedDesktop).pipe(Effect.flatMap(recorded => {
      const now = Date.now()
      if (now - recorded < 10_000) return Effect.void
      return Ref.set(lastRecordedDesktop, now).pipe(Effect.zipRight(
        recordResidentModels(profile.endpoint, stateDirectory, "Desktop").pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.ignore),
      ))
    })),
    replacedByDesktop,
    onReplacedByDesktop: notify("The desktop app replaced this CLI-only installation. Restarting with its command.").pipe(Effect.zipRight(restartAsDesktop)),
    onBeforeAttempt: Ref.set(ready, false),
    runAttempt: runHeadlessApplication({
      runtime, profile, stateDirectory, home: homedir(), environment: process.env,
      prepareStartup: prepareServeStartup(runtime, profile, stateDirectory).pipe(
        Effect.provide(process.platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon))),
      initializeUpdates, ...(container ? { container: { image: containerImage(CLI_VERSION) } } : {}),
      ...updateHooks,
      stopping: reason => Effect.sync(() => { process.stderr.write(reason === "DesktopTakeover"
        ? "The desktop app was opened and is taking over. This server will resume when it quits.\n"
        : reason === "Restart" ? "Restarting the Magnitude server.\n"
        : reason === "InstallUpdate" ? "Stopping the service to install the downloaded update; clients reconnect when it is back.\n"
        : "Stopping the Magnitude server.\n") }).pipe(
        // The version that starts after the update loads the same models again.
        Effect.zipRight(reason === "InstallUpdate" ? recordModelsBeforeUpdate(profile.endpoint, stateDirectory).pipe(Effect.provideService(FileSystem.FileSystem, fs)) : Effect.void)),
      stop: Deferred.await(stopped),
      observe: state => Ref.getAndSet(ready, state._tag === "Ready").pipe(Effect.flatMap(wasReady =>
        state._tag === "Ready" && !wasReady
          ? readServerReach({ endpoint: profile.endpoint, dataDirectory: profile.dataDirectory, service: owner === "service" }).pipe(
            Effect.flatMap(reach => Effect.sync(() => { process.stderr.write(renderServeReady(reach)) })),
            Effect.zipRight(resumeResidentModels(profile.endpoint, stateDirectory, notify).pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.forkIn(scope))))
          : Effect.void)),
    }).pipe(Effect.provide(native)),
    onInstallUpdate: Option.isSome(idle) ? idle.value.install : Effect.void,
  })
})).pipe(Effect.provide([BunContext.layer, BunSqliteDriverLayer]), Effect.catchAll(error => Effect.sync(() => {
  process.stderr.write(`${error.message}\n`)
  process.exitCode = 1
}))))
