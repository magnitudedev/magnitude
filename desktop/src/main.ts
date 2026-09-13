import { ApplicationUpdateControlFailed } from "@magnitudedev/sdk/desktop-host"
import { makeRendererRecovery } from "./renderer-recovery"
import { resolveQuitFailure } from "./quit-failure"
import { buildApplicationMenu } from "./application-menu"
import { buildTrayMenu } from "./tray-menu"
import { makeLoginStartup, WINDOWS_APPLICATION_ID } from "./login-startup"
import { ApplicationUpdateFailed, ApplicationUpdateSource, makeApplicationUpdate, unavailableApplicationUpdate } from "./application-update"
import { macUpdateSource } from "./mac-update-source"
import { makeLinuxUpdateSource } from "./linux-update-source"
import { readLinuxUpdateMetadata } from "./update-metadata"
import { NativeMacUpdate, nativeMacUpdate } from "./mac-update-stage"
import { ApplicationUpdateHandoff, makeUpdateHandoff } from "./update-handoff"
import { UpdateClientMetadata } from "@magnitudedev/release/hosted-update"
import { makeUpdateIdentity } from "./update-identity"
import { makeUpdatePreferences, UpdatePreferences } from "./update-preferences"
import { makeUpdateSchedule } from "./update-schedule"
import { readUpdateConfiguration, isUpdateAcceptanceBuild } from "./update-config"
import { NativeTrayFactory, NativeTrayFailed, TrayOwner, TrayOwnerLive } from "./tray-owner"
import { CommandExecutor, FetchHttpClient } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { NodeSqliteDriverLayer } from "@magnitudedev/daemon-management/node"
import { makeHarnessConnectionService, harnessConnectionPaths, harnessExecutableSearchPath } from "@magnitudedev/harness-connections"
import { HttpsUrlSchema, MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import { slate } from "@magnitudedev/client-common"
import { app, autoUpdater, BrowserWindow, dialog, ipcMain, Menu, nativeImage, nativeTheme, powerMonitor, shell, Tray } from "electron"
import { join, resolve, dirname } from "node:path"
import { homedir } from "node:os"
import { fileURLToPath } from "node:url"
import { Cause, Context, Deferred, Effect, Exit, Fiber, Layer, Option, PubSub, Queue, Ref, Runtime, Schema, Schedule, Scope, Stream } from "effect"
import { RpcServer } from "@effect/rpc"
import {
  acquireApplicationOwner, applicationStateDirectory, makeOwnedService, makeUnixOwnedChildSpawner, makeWindowsOwnedChildSpawner, requireServicePort, NativeHost, nativeHostLayer,
  OwnedChildSpawner, serveApplicationControl, serveWindowsApplicationControl, type ApplicationControlOptions,
  LinuxTrayHost, linuxTrayHostLayer, guardedCommandLayer,
  unixPrivateFilePermissions, windowsPrivateFilePermissions,
  adoptLinuxInstallationLease,
  NativeMacApplicationInstallation, nativeMachineIdentity, ApplicationMemory, nativeApplicationMemoryLayer, observeApplicationMemory,
} from "@magnitudedev/daemon-management/desktop-native"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { nativeWindowsPrivatePipesLayer, nativeWindowsJobOwnerLayer, WindowsPipeName } from "@magnitudedev/utils/windows-native"
import { type ApplicationSnapshot, type OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import { HostError, ApplicationAction, InferenceHostRpcs, type Page } from "./desktop-rpc"
import { makeElectronRpcServerLayer } from "./electron-rpc"
import { resolveHarnessEnvironment, harnessCommandExecutor } from "./shell-env"

app.setName("Magnitude")
if (process.platform === "win32") app.setAppUserModelId(WINDOWS_APPLICATION_ID)
const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, "../../..")
const background = process.argv.includes("--background") || (process.platform === "darwin" && app.isPackaged && app.getLoginItemSettings({ type: "mainAppService" }).wasOpenedAtLogin)
const isolatedProfile = isUpdateAcceptanceBuild || !app.isPackaged || process.env.MAGNITUDE_DEV_DATA_DIR !== undefined
const dataDir = process.env.MAGNITUDE_DEV_DATA_DIR ?? join(homedir(), isUpdateAcceptanceBuild ? ".magnitude-update-acceptance" : app.isPackaged ? ".magnitude" : ".magnitude-desktop-dev")
const stateOverride = process.env.MAGNITUDE_DESKTOP_STATE_DIR ?? (isUpdateAcceptanceBuild ? join(dataDir, "desktop") : undefined)
// Chromium can create its profile before native ownership is acquired. Keep it outside the
// protected Windows ownership leaf, which only native acquisition may create.
if (isolatedProfile) app.setPath("userData", process.platform === "win32" ? join(dataDir, "electron") : join(stateOverride ?? join(dataDir, "desktop"), "electron"))
const port = isolatedProfile ? Number(process.env.MAGNITUDE_DEV_PORT ?? (isUpdateAcceptanceBuild ? 11143 : 11101)) : 10100
const endpoint = `http://127.0.0.1:${port}`
const addonPath = app.isPackaged ? join(process.resourcesPath, "desktop-host.node") : join(root, `packages/daemon-management/dist/native/${process.platform}-${process.arch}/desktop-host.node`)
let exiting = false
let canPresentErrors = process.platform !== "win32"
let systemShutdownRequested = false
let earlyQuitRequested = false
let restartLinuxUpdate: (() => Effect.Effect<void, ApplicationUpdateFailed>) | undefined
let discardLinuxUpdate: Effect.Effect<void> = Effect.void
let reopenAfterUpdate = false
let requestQuit: () => void = () => { earlyQuitRequested = true }
app.on("before-quit", event => { if (!exiting) { event.preventDefault(); requestQuit() } })
// OS-requested termination must retire the owned service before Electron exits.
if (process.platform !== "win32") process.on("SIGTERM", () => requestQuit())
app.on("window-all-closed", () => {})
// Electron may emit a late native error after a staging observer has completed. Keep an
// application-lifetime listener so it remains diagnostic rather than an unhandled event.
if (process.platform === "darwin") autoUpdater.on("error", error => console.error("Application update:", error.message))

const program = Effect.scoped(Effect.gen(function* () {
  const native = yield* NativeHost
  if (process.platform === "linux" && app.isPackaged) yield* adoptLinuxInstallationLease(addonPath)
  if (process.platform === "win32") {
    yield* native.requireInteractiveDesktop
    canPresentErrors = true
  }
  const stateDir = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: dataDir, development: !app.isPackaged, override: Option.fromNullable(stateOverride), localAppDataDirectory: native.localAppDataDirectory })
  const owner = yield* acquireApplicationOwner(stateDir, background ? "EnsureRunning" : "ShowWindow")
  if (owner._tag === "Forwarded") { exiting = true; app.quit(); return }
  const handoff = process.platform === "darwin" && app.isPackaged
    ? yield* makeUpdateHandoff(stateDir, dirname(dirname(dirname(process.execPath)))).pipe(Effect.provide(NativeMacApplicationInstallation)) : undefined
  const updateAdmission = handoff ? yield* handoff.inspect(app.getVersion()) : undefined
  // A direct old-bundle launch can race the native installer. Exit before creating any
  // service or window; waiting inside this app would prevent ShipIt from installing it.
  if (updateAdmission?._tag === "Defer") return "Quit" as const
  yield* Effect.promise(() => app.whenReady())
  // A system shutdown can end our process before asynchronous cleanup finishes.
  // Never veto it; native lifetime containment remains the hard fallback.
  if (process.platform !== "win32") {
    const shutdown = () => { systemShutdownRequested = true; requestQuit() }
    powerMonitor.on("shutdown", shutdown)
    yield* Effect.addFinalizer(() => Effect.sync(() => powerMonitor.removeListener("shutdown", shutdown)))
  }
  const loginStartup = yield* makeLoginStartup(isolatedProfile)
  const rendererRecovery = yield* makeRendererRecovery
  const actions = yield* PubSub.unbounded<typeof ApplicationAction.Type>()
  const quit = yield* Queue.sliding<"Quit" | "RestartUpdate">(1)
  const state = yield* Ref.make<OwnedServiceState | null>(null)
  const model = yield* Ref.make({ label: "Model status unavailable", canStop: false })
  const runtime = yield* Effect.runtime<never>()
  const run = (effect: Effect.Effect<unknown>) => { Runtime.runFork(runtime)(effect) }
  requestQuit = () => run(Queue.offer(quit, "Quit"))
  if (earlyQuitRequested) yield* Queue.offer(quit, "Quit")
  const updateConfiguration = yield* readUpdateConfiguration.pipe(Effect.option)
  const updates = Option.isNone(updateConfiguration) ? unavailableApplicationUpdate("Application update configuration is invalid.")
    : isolatedProfile && !updateConfiguration.value.acceptance ? unavailableApplicationUpdate("Application updates are available in the installed Magnitude app.")
    : process.platform === "win32" ? unavailableApplicationUpdate("Application updates are not available in this Windows build.")
    : !app.isPackaged || (process.platform === "darwin" && !handoff) ? unavailableApplicationUpdate("Application update recovery is unavailable in this build.")
    : yield* Effect.gen(function* () {
      const identity = yield* makeUpdateIdentity(dataDir).pipe(
        Effect.provide((process.platform === "win32" ? windowsPrivateFilePermissions(addonPath) : unixPrivateFilePermissions).pipe(Layer.provideMerge(NodeContext.layer))),
      )
      const preferences = yield* makeUpdatePreferences(dataDir).pipe(Effect.provide(NodeContext.layer))
      const { trustedPublishers, origin, storageOrigin } = updateConfiguration.value
      if (process.platform === "linux") {
        const metadata = yield* readLinuxUpdateMetadata(process.resourcesPath, app.getVersion(), process.getSystemVersion()).pipe(Effect.provide(NodeContext.layer))
        const linux = yield* makeLinuxUpdateSource({ origin, storageOrigin, metadata, sign: identity.sign, trustedPublishers,
          userAgent: `Magnitude/${app.getVersion()} ${process.arch} Electron/${process.versions.electron} Linux/${process.getSystemVersion()}`,
          cacheDirectory: join(app.getPath("userData"), "updates"), stateDirectory: stateDir,
        }).pipe(Effect.provide(NodeContext.layer))
        restartLinuxUpdate = () => linux.restart(reopenAfterUpdate)
        discardLinuxUpdate = linux.discard
        return yield* makeApplicationUpdate(linux.previousFailure).pipe(Effect.provideService(ApplicationUpdateSource, linux.source), Effect.provideService(UpdatePreferences, preferences))
      }
      if (!handoff) return unavailableApplicationUpdate("Application update recovery is unavailable in this build.")
      const metadata = yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: app.getVersion(), os: "darwin",
        os_version: process.getSystemVersion(), arch: process.arch, package: "mac-zip" })
      const source = yield* macUpdateSource({ origin, storageOrigin, metadata,
        sign: identity.sign, trustedPublishers, userAgent: `Magnitude/${app.getVersion()} ${process.arch} Electron/${process.versions.electron} macOS/${process.getSystemVersion()}`,
        cacheDirectory: join(app.getPath("userData"), "updates"),
      }).pipe(Effect.provideService(NativeMacUpdate, nativeMacUpdate(autoUpdater)), Effect.provideService(ApplicationUpdateHandoff, handoff))
      return yield* makeApplicationUpdate(updateAdmission?._tag === "Failed" ? Option.some(updateAdmission.message) : Option.none()).pipe(
        Effect.provideService(ApplicationUpdateSource, source), Effect.provideService(UpdatePreferences, preferences))
    }).pipe(Effect.catchAll(() => Effect.succeed(unavailableApplicationUpdate("Application update setup could not be read."))))
  const updateSchedule = yield* makeUpdateSchedule(updates.check)
  const resumeUpdates = () => run(updateSchedule.resume)
  powerMonitor.on("resume", resumeUpdates)
  yield* Effect.addFinalizer(() => Effect.sync(() => powerMonitor.removeListener("resume", resumeUpdates)))

  const nativeTray = Layer.succeed(NativeTrayFactory, { create: Effect.acquireRelease(Effect.try({ try: () => {
    // A monochrome template works in either macOS menu-bar appearance.
    const icon = nativeImage.createFromPath(app.isPackaged ? join(process.resourcesPath, "trayTemplate@2x.png") : join(root, "assets/brand/trayTemplate@2x.png"))
    icon.setTemplateImage(true)
    const result = new Tray(icon)
    result.setToolTip("Magnitude")
    return result
  }, catch: () => new NativeTrayFailed({ message: "Magnitude could not register its tray icon." }) }), value => Effect.sync(() => value.destroy())).pipe(
    Effect.map(value => ({ setMenu: (menu: readonly Electron.MenuItemConstructorOptions[]) => Effect.try({
      try: () => value.setContextMenu(Menu.buildFromTemplate([...menu])),
      catch: () => new NativeTrayFailed({ message: "Magnitude could not update its tray menu." }),
    }) })),
  ) })
  const tray = Context.get(yield* Layer.build(TrayOwnerLive.pipe(Layer.provide(nativeTray))), TrayOwner)
  let window: BrowserWindow
  let pendingPage: Page = "discover"
  let wantsWindow = !background
  const loadRenderer = () => Effect.tryPromise(() => process.env.ELECTRON_RENDERER_URL
    ? window.loadURL(process.env.ELECTRON_RENDERER_URL)
    : window.loadFile(join(here, "../renderer/index.html"))).pipe(
      Effect.catchAll(error => rendererRecovery.loadFailed.pipe(Effect.zipRight(Effect.logError(error)))),
    )
  const show = (page?: Page) => Ref.get(state).pipe(Effect.flatMap(current => current?._tag === "Stopping" || current?._tag === "Stopped" ? Effect.void : Effect.gen(function* () {
    if (page !== undefined) pendingPage = page
    wantsWindow = true
    if (!window) return
    const reload = yield* rendererRecovery.open
    yield* Effect.sync(() => {
      if (window.isMinimized()) window.restore()
      window.show()
      window.focus()
    })
    if (reload) yield* loadRenderer()
    yield* PubSub.publish(actions, { _tag: "Navigate", page: pendingPage })
  })))
  const refreshTray = Effect.gen(function* () {
    const current = yield* Ref.get(state)
    const presentation = yield* Ref.get(model)
    const update = yield* updates.state
    yield* tray.setMenu(buildTrayMenu({ service: current?._tag ?? "Unknown", model: presentation, updateReady: update.transfer._tag === "Ready" }, {
      open: page => run(show(page)),
      stopModel: () => run(PubSub.publish(actions, { _tag: "StopModel" })),
      quit: requestQuit,
      restartUpdate: () => run(updates.requireReady.pipe(Effect.zipRight(Queue.offer(quit, "RestartUpdate")), Effect.catchAll(() => Effect.void))),
    }))
  })
  yield* refreshTray
  yield* updates.changes.pipe(Stream.map(value => value.transfer._tag === "Ready"), Stream.changes,
    Stream.runForEach(() => refreshTray), Effect.forkScoped)
  if (process.platform === "linux") {
    const trayHost = Context.get(yield* Layer.build(linuxTrayHostLayer()), LinuxTrayHost)
    yield* trayHost.changes.pipe(Stream.runForEach(tray.observeHost), Effect.forkScoped)
  }
  const harnessEnvironment = yield* resolveHarnessEnvironment().pipe(Effect.provide(guardedCommandLayer(join(dirname(addonPath), "magnitude-command"))), Effect.forkScoped)
  const spawner = process.platform === "win32" ? yield* Effect.gen(function* () {
    const pipes = yield* Layer.build(nativeWindowsPrivatePipesLayer(addonPath))
    const jobs = yield* Layer.build(nativeWindowsJobOwnerLayer(addonPath))
    return yield* makeWindowsOwnedChildSpawner.pipe(Effect.provide(pipes), Effect.provide(jobs))
  }) : yield* makeUnixOwnedChildSpawner
  const admittedSpawner = yield* requireServicePort(port).pipe(Effect.provideService(OwnedChildSpawner, spawner))
  const service = yield* makeOwnedService({
    executable: app.isPackaged ? join(process.resourcesPath, process.platform === "win32" ? "magnitude-service.exe" : "magnitude-service") : process.env.MAGNITUDE_BUN_PATH ?? "bun",
    arguments: [...(app.isPackaged ? [] : [join(root, "packages/acn/src/binary.ts")]), "serve", "--data-dir", dataDir, "--port", String(port)],
    environment: { ...process.env, MAGNITUDE_NATIVE_HOST: addonPath, ...(app.isPackaged || process.env.MAGNITUDE_ICN_PATH ? {} : { MAGNITUDE_ICN_PATH: join(root, "inference/target/development/installation.json") }) },
  }, MAGNITUDE_RPC_VERSION).pipe(Effect.provideService(OwnedChildSpawner, admittedSpawner))
  const snapshot = Effect.all({ service: service.state, tray: tray.state }).pipe(Effect.map(value => ({ version: 1 as const, pid: process.pid, endpoint, ...value })))
  const snapshots = Stream.zipLatest(service.changes, tray.changes).pipe(Stream.map(([service, tray]) => ({ version: 1 as const, pid: process.pid, endpoint, service, tray })))
  yield* service.changes.pipe(Stream.runForEach(current => Ref.set(state, current).pipe(Effect.zipRight(refreshTray))), Effect.forkScoped)
  const control: ApplicationControlOptions = { snapshot, update: action => Effect.gen(function* () {
    if (action === "check") yield* updateSchedule.check
    if (action === "download") yield* updates.download
    if (action === "install") yield* updates.requireReady
    return { state: yield* updates.state, afterReply: action === "install" ? Queue.offer(quit, "RestartUpdate").pipe(Effect.asVoid) : Effect.void }
  }).pipe(Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message }))), login: action => action === "read" ? loginStartup.read : loginStartup.set(action === "enable"), dispatch: intent => intent === "Quit" ? Queue.offer(quit, "Quit").pipe(Effect.asVoid) : intent === "ShowWindow" ? show() : intent === "Retry" ? service.retry : Effect.void }
  if (process.platform === "win32") {
    const name = yield* Schema.decodeUnknown(WindowsPipeName)(owner.socketPath)
    yield* serveWindowsApplicationControl(name, control).pipe(Effect.provide(nativeWindowsPrivatePipesLayer(addonPath)))
  } else yield* serveApplicationControl(owner.socketPath, control)
  const connections = yield* Effect.cached(Effect.gen(function* () {
    const environment = yield* Fiber.join(harnessEnvironment)
    const executor = yield* harnessCommandExecutor(environment)
    return yield* makeHarnessConnectionService({
      paths: harnessConnectionPaths(isolatedProfile ? join(dataDir, "harness-home") : undefined, environment),
      serviceEndpoint: endpoint,
      detect: connector => connector.detect(harnessExecutableSearchPath(environment.PATH)),
    }).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor))
  }).pipe(Effect.provide([NodeContext.layer, FetchHttpClient.layer, NodeSqliteDriverLayer])))
  const connectionChanges = yield* PubSub.sliding<void>(1)
  const connectionError = (error: { readonly message: string }) => new HostError({ message: error.message })
  const memory = Context.get(yield* Layer.build(nativeApplicationMemoryLayer(addonPath)), ApplicationMemory)
  const machineIdentity = yield* Effect.cached(nativeMachineIdentity(addonPath))
  const handlers = InferenceHostRpcs.toLayer({
    MachineIdentity: () => machineIdentity,
    Memory: () => observeApplicationMemory(memory, () => !!window && !window.isDestroyed() && window.isVisible()),
    ApplicationInfo: () => Effect.sync(() => ({ version: app.getVersion() })),
    Updates: () => updates.changes,
    SetAutoDownload: ({ enabled }) => updates.setAutoDownload(enabled).pipe(Effect.mapError(connectionError), Effect.as({})),
    CheckUpdate: () => updateSchedule.check.pipe(Effect.mapError(connectionError), Effect.as({})),
    DownloadUpdate: () => updates.download.pipe(Effect.mapError(connectionError), Effect.as({})),
    RestartUpdate: () => updates.requireReady.pipe(Effect.mapError(connectionError), Effect.zipRight(Effect.gen(function* () {
      if (!window || window.isDestroyed() || !window.isVisible() || window.isMinimized() || systemShutdownRequested) {
        return yield* new HostError({ message: "Open Magnitude before choosing Restart to update." })
      }
      yield* Queue.offer(quit, "RestartUpdate")
      return {}
    }))),
    Observe: () => snapshots,
    Actions: () => Stream.concat(Stream.succeed({ _tag: "Navigate" as const, page: pendingPage }), Stream.fromPubSub(actions)),
    PresentModel: value => Ref.set(model, value).pipe(Effect.zipRight(refreshTray), Effect.as({})),
    Appearance: ({ preference }) => Effect.sync(() => { nativeTheme.themeSource = preference; window?.setBackgroundColor(nativeTheme.shouldUseDarkColors ? slate[925] : slate[50]); return {} }),
    LoginStartup: () => Stream.repeatEffectWithSchedule(loginStartup.read.pipe(Effect.catchAll(error => Effect.succeed({ _tag: "Unavailable" as const, message: error.message }))), Schedule.spaced("2 seconds")).pipe(Stream.mapError(connectionError)),
    SetLoginStartup: ({ enabled }) => loginStartup.set(enabled).pipe(Effect.mapError(connectionError), Effect.as({})),
    Connections: () => Stream.concat(Stream.succeed(undefined), Stream.merge(Stream.fromPubSub(connectionChanges), Stream.fromSchedule(Schedule.spaced("2 seconds")))).pipe(Stream.mapEffect(() => connections.pipe(Effect.flatMap(service => service.inspect), Effect.map(connections => ({ _tag: "Ready" as const, connections })), Effect.catchAll(error => Effect.succeed({ _tag: "Unavailable" as const, message: error.message }))))),
    Connect: ({ harness, model }) => connections.pipe(Effect.flatMap(service => service.connect(harness, { model, installSkill: true })), Effect.mapError(connectionError), Effect.tap(() => PubSub.publish(connectionChanges, undefined)), Effect.as({})),
    Disconnect: ({ harness }) => connections.pipe(Effect.flatMap(service => service.disconnect(harness)), Effect.mapError(connectionError), Effect.tap(() => PubSub.publish(connectionChanges, undefined)), Effect.as({})),
    Retry: () => service.retry.pipe(Effect.as({})),
    Quit: () => Queue.offer(quit, "Quit").pipe(Effect.as({})),
  })
  yield* RpcServer.layer(InferenceHostRpcs).pipe(Layer.provide(handlers), Layer.provide(makeElectronRpcServerLayer(ipcMain)), Layer.build)
  window = yield* Effect.acquireRelease(Effect.sync(() => {
    const value = new BrowserWindow({ width: 1120, height: 800, minWidth: 800, minHeight: 600, show: false, title: "Magnitude", icon: app.isPackaged ? join(process.resourcesPath, "application-icon.png") : join(root, "assets/brand/application-icon.png"), backgroundColor: nativeTheme.shouldUseDarkColors ? slate[925] : slate[50], webPreferences: { preload: join(here, "../preload/preload.mjs"), contextIsolation: true, nodeIntegration: false, sandbox: false, backgroundThrottling: false } })
    value.on("close", event => { if (!exiting) { event.preventDefault(); value.hide() } })
    value.webContents.on("render-process-gone", () => run(Effect.gen(function* () {
      yield* Ref.set(model, { label: "Model status unavailable", canStop: false })
      const current = yield* Ref.get(state)
      if (exiting || current?._tag === "Stopping" || current?._tag === "Stopped") return
      const retry = yield* rendererRecovery.crashed
      if (!retry) yield* Ref.set(model, { label: "Window unavailable · Open Magnitude to retry", canStop: false })
      yield* refreshTray
      if (retry) yield* loadRenderer()
    })))
    value.webContents.on("did-fail-load", (_event, code, _description, _url, isMainFrame) => {
      if (!isMainFrame || code === -3 || exiting) return
      run(rendererRecovery.loadFailed.pipe(
        Effect.zipRight(Ref.set(model, { label: "Window unavailable · Open Magnitude to retry", canStop: false })),
        Effect.zipRight(refreshTray),
      ))
    })
    value.webContents.session.webRequest.onHeadersReceived((details, callback) => callback({ responseHeaders: {
      ...details.responseHeaders,
      "Content-Security-Policy": [`default-src 'self'; script-src 'self'${app.isPackaged ? "" : " 'unsafe-inline'"}; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:* http://localhost:* ws://localhost:*`],
    } }))
    value.webContents.on("will-navigate", event => event.preventDefault())
    value.webContents.setWindowOpenHandler(({ url }) => {
      const source = Schema.decodeUnknownEither(HttpsUrlSchema)(url)
      if (source._tag === "Right") run(Effect.tryPromise(() => shell.openExternal(source.right)).pipe(
        Effect.catchAll(() => Effect.sync(() => dialog.showErrorBox("Could not open model source", "Open your browser and try the source link again."))),
      ))
      return { action: "deny" }
    })
    value.webContents.on("console-message", (_event, level, message) => console.log(`[renderer:${level}] ${message}`))
    value.webContents.on("preload-error", (_event, path, error) => console.error(path, error))
    return value
  }), value => Effect.sync(() => value.destroy()))
  Menu.setApplicationMenu(Menu.buildFromTemplate(buildApplicationMenu(process.platform, {
    open: page => run(show(page)), quit: requestQuit,
  })))
  yield* loadRenderer()
  // Initial activation belongs to launch intent. Subsequent Dock activation is an explicit Open.
  const activate = () => run(show())
  app.on("activate", activate)
  yield* Effect.addFinalizer(() => Effect.sync(() => app.removeListener("activate", activate)))
  if (wantsWindow) yield* show()
  for (;;) {
    const intent = yield* Queue.take(quit)
    reopenAfterUpdate = BrowserWindow.getAllWindows().some(window => window.isVisible())
    yield* updates.close
    const stopped = yield* service.shutdown.pipe(Effect.either)
    if (stopped._tag === "Right") return systemShutdownRequested ? "Quit" as const : intent
    if (systemShutdownRequested) yield* Effect.logError(stopped.left.message)
    else {
      const retry = yield* resolveQuitFailure(stopped.left.message, {
        showDialog: options => dialog.showMessageBox(options),
        forceQuit: () => { exiting = true; app.exit(1) },
      })
      if (retry) yield* Queue.offer(quit, "Quit")
    }
  }
})).pipe(Effect.provide(nativeHostLayer(addonPath)), Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))
Effect.runPromiseExit(program).then(Exit.match({
  onSuccess: intent => {
    exiting = true
    if (intent !== "RestartUpdate") {
      if (process.platform === "linux" && !systemShutdownRequested) void Effect.runPromise(discardLinuxUpdate.pipe(Effect.timeoutOption("1 second"))).finally(() => app.quit())
      else app.quit()
      return
    }
    if (process.platform === "linux" && restartLinuxUpdate) {
      void Effect.runPromise(restartLinuxUpdate()).then(() => app.quit(), error => {
        dialog.showErrorBox("Magnitude could not restart for its update", String(error))
        app.quit()
      })
      return
    }
    try { autoUpdater.quitAndInstall() }
    catch (error) {
      dialog.showErrorBox("Magnitude could not restart for its update", String(error))
      app.quit()
    }
  },
  onFailure: cause => {
    const message = Cause.pretty(cause)
    console.error(message)
    if (!canPresentErrors) { exiting = true; app.exit(1); return }
    void app.whenReady().then(() => {
      if (!background && !systemShutdownRequested) dialog.showErrorBox("Magnitude could not continue", message)
      exiting = true
      app.exit(1)
    })
  },
}))
