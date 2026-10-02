import { windowChrome, windowControlColors } from "./window-chrome"
import { makeMacCliRegistration } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateControlFailed } from "@magnitudedev/sdk/desktop-host"
import { makeRendererRecovery } from "./renderer-recovery"
import { nativeQuitFailureDecision, resolveQuitFailure } from "./quit-failure"
import { buildApplicationMenu } from "./application-menu"
import { buildTrayMenu, MODEL_STATUS_ITEM } from "./tray-menu"
import { loadTrayStatusRow } from "./tray-status"
import { initializeLoginStartup, makeLoginStartup, WINDOWS_APPLICATION_ID } from "./login-startup"
import { ApplicationUpdateFailed, ApplicationUpdateSource, makeApplicationUpdate, unavailableApplicationUpdate, readLinuxUpdateMetadata, makeUpdateIdentity, makeUpdateSchedule, makeLinuxUpdateSource, makeWindowsUpdateSource, hostedUpdateSource, startMacForegroundInstallation, macStartupUpdateOperation } from "@magnitudedev/daemon-management/application-update"
import { PreparedUpdateInstaller, reconcilePreparedUpdate, installPreparedUpdate, type UpdateInstallationIntent } from "@magnitudedev/daemon-management/application-update"
import { isNewerVersion } from "@magnitudedev/release"
import { ReleaseTarget, UpdateClientMetadata } from "@magnitudedev/release/hosted-update"
import { makeAppearancePreferences, makeUpdatePreferences, UpdatePreferences } from "@magnitudedev/daemon-management/desktop-native"
import { readUpdateConfiguration, isUpdateAcceptanceBuild } from "./update-config"
import { NativeTrayFactory, NativeTrayFailed, TrayOwner, TrayOwnerLive, type TrayMenu } from "./tray-owner"
import { CommandExecutor, FetchHttpClient } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { NodeSqliteDriverLayer } from "@magnitudedev/storage/sqlite/node"
import { HttpsUrlSchema } from "@magnitudedev/sdk"
import { slate } from "@magnitudedev/client-common"
import { DESKTOP_APP_ORIGIN, handleAppProtocol, resolveRendererDir } from "./app-protocol"
import { developmentRelaunchExitCode, rendererServedByDevServer } from "./development-relaunch"
import { app, BrowserWindow, dialog, ipcMain, Menu, nativeImage, nativeTheme, powerMonitor, shell, Tray } from "electron"
import { join, resolve, dirname } from "node:path"
import { homedir } from "node:os"
import { fileURLToPath } from "node:url"
import { Cause, Context, Deferred, Effect, Exit, Fiber, Layer, Option, PubSub, Queue, Ref, Runtime, Schema, Schedule, Scope, Stream } from "effect"
import { RpcServer } from "@effect/rpc"
import {
  acquireApplicationOwner, applicationStateDirectory, isUpdateInstallationActive, NativeHost, nativeHostLayer,
  resolveApplicationProfile, applicationNativeHostPath, makeApplicationService, type ApplicationRuntime,
  serveApplicationControl, serveWindowsApplicationControl, type ApplicationControlOptions,
  LinuxTrayHost, linuxTrayHostLayer,
  unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory,
  nativeWindowsInstallerVerifier,
  adoptLinuxInstallationLease, acquireMacApplicationInstallationLease, nativeMacUpdateAdmission,
  acquireApplicationMaintenance, acquireUpdateInstallationLease, MacApplicationInstallation, PreparedUpdateStore, makePreparedUpdateStore, NativeMacApplicationInstallation,
} from "@magnitudedev/daemon-management/desktop-native"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { nativeWindowsPrivatePipesLayer, WindowsPipeName } from "@magnitudedev/utils/windows-native"
import { type ApplicationSnapshot, type OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import { HostError, ApplicationAction, InferenceHostRpcs, ModelTrayPresentation, type HostNotice, type Page } from "./desktop-rpc"
import type { QuitFailureDecision } from "@magnitudedev/client-common/application/contracts"
import { makeElectronRpcServerLayer } from "./electron-rpc"
import { resolveHarnessEnvironment } from "@magnitudedev/harness-connections"
import { guardedCommandLayer } from "@magnitudedev/utils/guarded-command"
import { MAGNITUDE_VERSION } from "@magnitudedev/version"

app.setName("Magnitude")
if (process.platform === "win32") app.setAppUserModelId(WINDOWS_APPLICATION_ID)
const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, "../../..")
const background = process.argv.includes("--background") || (process.platform === "darwin" && app.isPackaged && app.getLoginItemSettings({ type: "mainAppService" }).wasOpenedAtLogin)
const applicationRuntime: ApplicationRuntime = app.isPackaged
  ? { _tag: "Installed", resourcesDirectory: process.resourcesPath }
  : { _tag: "Development", repository: root }
const profile = resolveApplicationProfile({ runtime: applicationRuntime, home: homedir(), platform: process.platform,
  acceptance: isUpdateAcceptanceBuild, environment: process.env })
const { isolated: isolatedProfile, dataDirectory: dataDir, port, endpoint } = profile
const stateOverride = process.env.MAGNITUDE_DESKTOP_STATE_DIR
// Chromium can create its profile before native ownership is acquired. Keep it outside the
// protected Windows ownership leaf, which only native acquisition may create.
app.setPath("userData", join(dataDir, "electron"))
app.setPath("sessionData", join(dataDir, "electron"))
const addonPath = applicationNativeHostPath(applicationRuntime, process.platform, process.arch)
let exiting = false
let canPresentErrors = process.platform !== "win32"
let systemShutdownRequested = false
let earlyQuitRequested = false
let restartPreparedUpdate: ((intent: UpdateInstallationIntent) => Effect.Effect<"Started" | "Deferred", ApplicationUpdateFailed>) | undefined
let startupUpdateStarted = false
let startupMacUpdate = false
let macUpdateOperation: "Install" | "Recover" = "Install"
let startupUpdateDeferred = false
let reopenAfterUpdate = false
let requestQuit: () => void = () => { earlyQuitRequested = true }
app.on("before-quit", event => { if (!exiting) { event.preventDefault(); requestQuit() } })
// OS-requested termination must retire the owned service before Electron exits.
if (process.platform !== "win32") process.on("SIGTERM", () => requestQuit())
app.on("window-all-closed", () => {})
const program = Effect.scoped(Effect.gen(function* () {
  const native = yield* NativeHost
  if (process.platform === "linux" && app.isPackaged) yield* adoptLinuxInstallationLease(addonPath)
  if (process.platform === "win32") {
    yield* native.requireInteractiveDesktop
    canPresentErrors = true
  }
  const stateDir = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: dataDir, override: Option.fromNullable(stateOverride) })
  const owner = yield* acquireApplicationOwner(stateDir, { _tag: "Desktop", intent: background ? "EnsureRunning" : "ShowWindow" })
  if (owner._tag === "Forwarded") { exiting = true; app.quit(); return }
  if (yield* isUpdateInstallationActive(stateDir)) return "Quit" as const
  if (process.platform === "darwin" && app.isPackaged) {
    startupUpdateDeferred = yield* Effect.gen(function* () {
      const installation = yield* MacApplicationInstallation
      return yield* installation.isInstalling(dirname(dirname(dirname(process.execPath))))
    }).pipe(Effect.provide(NativeMacApplicationInstallation))
    yield* acquireMacApplicationInstallationLease(dirname(dirname(dirname(process.execPath)))).pipe(
      Effect.provide(nativeMacUpdateAdmission(addonPath)))
  }
  yield* Effect.promise(() => app.whenReady())
  yield* Effect.sync(() => handleAppProtocol(resolveRendererDir(here)))
  const preferenceWrites = yield* Effect.makeSemaphore(1)
  const appearance = yield* makeAppearancePreferences(dataDir).pipe(Effect.provide(NodeContext.layer))
  const initialAppearance = yield* appearance.read.pipe(Effect.catchAll(error =>
    Effect.logWarning(error.message).pipe(Effect.as("system" as const))))
  nativeTheme.themeSource = initialAppearance
  // A system shutdown can end our process before asynchronous cleanup finishes.
  // Never veto it; native lifetime containment remains the hard fallback.
  if (process.platform !== "win32") {
    const shutdown = () => { systemShutdownRequested = true; requestQuit() }
    powerMonitor.on("shutdown", shutdown)
    yield* Effect.addFinalizer(() => Effect.sync(() => powerMonitor.removeListener("shutdown", shutdown)))
  }
  const loginStartup = yield* makeLoginStartup(isolatedProfile)
  yield* initializeLoginStartup(loginStartup, stateDir, isolatedProfile).pipe(Effect.provide(NodeContext.layer), Effect.catchAll(error => Effect.logWarning(error.message)))
  const rendererRecovery = yield* makeRendererRecovery
  const actions = yield* PubSub.unbounded<typeof ApplicationAction.Type>()
  const quit = yield* Queue.sliding<"Quit" | "RestartUpdate" | "Relaunch">(1)
  const state = yield* Ref.make<OwnedServiceState | null>(null)
  const unavailableModel = (label: string): typeof ModelTrayPresentation.Type => ({ label, status: Option.none(), canStop: false })
  const model = yield* Ref.make(unavailableModel("Model status unavailable"))
  const runtime = yield* Effect.runtime<never>()
  const run = (effect: Effect.Effect<unknown>) => { Runtime.runFork(runtime)(effect) }
  requestQuit = () => run(Queue.offer(quit, "Quit"))
  if (earlyQuitRequested) yield* Queue.offer(quit, "Quit")
  const updateConfiguration = yield* readUpdateConfiguration.pipe(Effect.option)
  const updates = Option.isNone(updateConfiguration) ? unavailableApplicationUpdate("Application update configuration is invalid.")
    : isolatedProfile && !updateConfiguration.value.acceptance ? unavailableApplicationUpdate("Application updates are available in the installed Magnitude app.")
    : !app.isPackaged ? unavailableApplicationUpdate("Application update recovery is unavailable in this build.")
    : yield* Effect.gen(function* () {
      const privateFiles = (process.platform === "win32" ? windowsPrivateFilePermissions(addonPath) : unixPrivateFilePermissions).pipe(Layer.provideMerge(NodeContext.layer))
      if (process.platform === "win32") {
        const retired = yield* recoverWindowsUpdateDirectory(addonPath, dataDir).pipe(Effect.provide(NodeContext.layer))
        if (retired) yield* Effect.logWarning("An older update cache was preserved separately. Download the update again.")
      }
      const identity = yield* makeUpdateIdentity(dataDir).pipe(Effect.provide(privateFiles))
      const preferences = yield* makeUpdatePreferences(dataDir).pipe(Effect.provide(NodeContext.layer))
      const { trustedPublishers, origin } = updateConfiguration.value
      const metadata = process.platform === "linux"
        ? yield* readLinuxUpdateMetadata(process.resourcesPath, app.getVersion(), process.getSystemVersion()).pipe(Effect.provide(NodeContext.layer))
        : yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: app.getVersion(), os: process.platform === "win32" ? "windows" : "darwin",
            os_version: process.getSystemVersion(), arch: process.arch, package: process.platform === "win32" ? "windows-exe" : "mac-zip" })
      const target = yield* Schema.decodeUnknown(ReleaseTarget)({ os: metadata.os, arch: metadata.arch, package: metadata.package })
      const store = yield* makePreparedUpdateStore({ dataDirectory: dataDir, target, trustedPublishers }).pipe(Effect.provide(privateFiles))
      const options = { origin, metadata, sign: identity.sign, trustedPublishers,
        userAgent: `Magnitude/${app.getVersion()} ${process.arch} Electron/${process.versions.electron} ${metadata.os}/${process.getSystemVersion()}`,
        dataDirectory: dataDir, stateDirectory: stateDir }
      const platform = yield* Effect.gen(function* () {
        if (process.platform === "linux") return yield* makeLinuxUpdateSource(options)
        if (process.platform === "win32") {
          const publisher = updateConfiguration.value.windowsPublisher
          if (Option.isNone(publisher)) return yield* new ApplicationUpdateFailed({ message: "The Windows update publisher is missing." })
          return yield* makeWindowsUpdateSource({ ...options, applicationPath: process.execPath,
            cliPath: join(process.resourcesPath, "magnitude.exe"), addonPath }).pipe(
              Effect.provide([nativeWindowsInstallerVerifier(addonPath, publisher.value), privateFiles]))
        }
        return { source: yield* hostedUpdateSource(options, (archive, release) => store.prepare(archive, release).pipe(
          Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))), installer: undefined }
      }).pipe(Effect.provideService(PreparedUpdateStore, store), Effect.provide(NodeContext.layer))
      restartPreparedUpdate = platform.installer ? intent => installPreparedUpdate(intent).pipe(
        Effect.provideService(PreparedUpdateStore, store), Effect.provideService(PreparedUpdateInstaller, platform.installer!)) : undefined
      const saved = yield* store.read
      if (startupUpdateDeferred) {
        // A previously admitted native installer may still be completing its relaunch.
        if (Option.isSome(saved) && !isNewerVersion(saved.value.release.version, app.getVersion())) startupUpdateDeferred = false
        else return unavailableApplicationUpdate("An application update is being installed.")
      }
      if (process.platform === "darwin" && !earlyQuitRequested) {
        const operation = yield* macStartupUpdateOperation(dirname(dirname(process.resourcesPath)), app.getVersion()).pipe(
          Effect.provideService(PreparedUpdateStore, store), Effect.provide(NodeContext.layer))
        if (Option.isSome(operation)) {
          startupMacUpdate = true
          macUpdateOperation = operation.value
          reopenAfterUpdate = !background
          return unavailableApplicationUpdate("Completing the prepared application update.")
        }
      }
      let pending = yield* reconcilePreparedUpdate(app.getVersion()).pipe(Effect.provideService(PreparedUpdateStore, store))
      if (Option.isSome(pending) && pending.value.installation._tag === "Unattempted" && !earlyQuitRequested) {
        const attempted = yield* restartPreparedUpdate!({ continuation: { _tag: "Desktop", showWindow: !background }, allowAuthorizationPrompt: !background }).pipe(Effect.either)
        startupUpdateStarted = attempted._tag === "Right" && attempted.right === "Started"
        pending = yield* store.read
      }
      return yield* makeApplicationUpdate(pending).pipe(
        Effect.provideService(ApplicationUpdateSource, platform.source), Effect.provideService(PreparedUpdateStore, store), Effect.provideService(UpdatePreferences, preferences))
    }).pipe(
      Effect.catchTag("WindowsUpdateDirectoryFailed", error => Effect.succeed(unavailableApplicationUpdate(error.message))),
      Effect.catchAll(() => Effect.succeed(unavailableApplicationUpdate("Application update setup could not be read."))),
    )
  if (startupUpdateDeferred) return "Quit" as const
  if (startupMacUpdate) return "InstallMacUpdate" as const
  if (startupUpdateStarted) return "RestartUpdate" as const
  const updateSchedule = yield* makeUpdateSchedule(updates.check)
  const resumeUpdates = () => run(updateSchedule.resume)
  powerMonitor.on("resume", resumeUpdates)
  yield* Effect.addFinalizer(() => Effect.sync(() => powerMonitor.removeListener("resume", resumeUpdates)))

  const statusRow = yield* loadTrayStatusRow(app.isPackaged
    ? join(process.resourcesPath, "tray-status.node")
    : join(root, `desktop/dist/native/${process.platform}-${process.arch}/tray-status.node`))
  const nativeTray = Layer.succeed(NativeTrayFactory, { create: Effect.acquireRelease(Effect.try({ try: () => {
    // A monochrome template works in either macOS menu-bar appearance.
    const iconDirectory = app.isPackaged ? process.resourcesPath : join(root, "assets/brand")
    const icon = nativeImage.createFromPath(join(iconDirectory, process.platform === "win32" ? "tray-white.ico" : "trayTemplate@2x.png"))
    if (process.platform === "darwin") icon.setTemplateImage(true)
    const result = new Tray(icon)
    result.setToolTip("Magnitude")
    let syncTheme = () => {}
    if (process.platform === "win32") {
      const blackIcon = nativeImage.createFromPath(join(iconDirectory, "tray-black.ico"))
      syncTheme = () => result.setImage(nativeTheme.shouldUseDarkColorsForSystemIntegratedUI ? icon : blackIcon)
      syncTheme()
      nativeTheme.on("updated", syncTheme)
      result.on("click", () => run(show()))
      result.on("double-click", () => run(show()))
    }
    // With the live model row the tray pops its menu up itself: Electron keeps running
    // JavaScript during a menu it pops up, so the row updates while the menu is open.
    let shown: { readonly template: TrayMenu, readonly menu: Menu } | undefined
    if (Option.isSome(statusRow)) {
      const popUp = () => {
        if (!shown) return
        const index = shown.template.findIndex(item => item.id === MODEL_STATUS_ITEM)
        if (index >= 0) statusRow.value.expect(index)
        result.popUpContextMenu(shown.menu)
      }
      result.on("click", popUp)
      result.on("right-click", popUp)
    }
    const setMenu = (template: TrayMenu) => {
      const menu = Menu.buildFromTemplate([...template])
      if (Option.isSome(statusRow)) shown = { template, menu }
      else result.setContextMenu(menu)
    }
    return { tray: result, syncTheme, setMenu }
  }, catch: () => new NativeTrayFailed({ message: "Magnitude could not register its tray icon." }) }), value => Effect.sync(() => {
    nativeTheme.removeListener("updated", value.syncTheme)
    value.tray.destroy()
  })).pipe(
    Effect.map(({ setMenu }) => ({ setMenu: (menu: TrayMenu) => Effect.try({
      try: () => setMenu(menu),
      catch: () => new NativeTrayFailed({ message: "Magnitude could not update its tray menu." }),
    }) })),
  ) })
  let window: BrowserWindow
  const windowShown = () => window !== undefined && !window.isDestroyed() && window.isVisible()
  /** Errors raised while the window is shown appear in the app; native dialogs only when there is no window. */
  const showNotice = (notice: HostNotice) => windowShown()
    ? PubSub.publish(actions, { _tag: "ShowNotice", notice }).pipe(Effect.asVoid)
    : Effect.sync(() => dialog.showErrorBox(notice.title, notice.description))
  const pendingQuitDecision = yield* Ref.make(Option.none<Deferred.Deferred<QuitFailureDecision>>())
  const decideQuitFailure: Effect.Effect<QuitFailureDecision, unknown> = Effect.suspend(() => windowShown()
    ? Effect.gen(function* () {
      const decision = yield* Deferred.make<QuitFailureDecision>()
      yield* Ref.set(pendingQuitDecision, Option.some(decision))
      yield* PubSub.publish(actions, { _tag: "QuitFailed" })
      return yield* Deferred.await(decision)
    })
    : nativeQuitFailureDecision(options => dialog.showMessageBox(options)))
  let pendingPage: Page = "discover"
  let wantsWindow = !background
  const loadRenderer = () => Effect.tryPromise(() => process.env.ELECTRON_RENDERER_URL
    ? window.loadURL(process.env.ELECTRON_RENDERER_URL)
    : window.loadURL(`${DESKTOP_APP_ORIGIN}/index.html`)).pipe(
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
    // Raising the retained window must preserve the renderer's current page.
    if (page !== undefined) yield* PubSub.publish(actions, { _tag: "Navigate", page })
  })))
  const tray = Context.get(yield* Layer.build(TrayOwnerLive.pipe(Layer.provide(nativeTray))), TrayOwner)
  const refreshTray = Effect.gen(function* () {
    const current = yield* Ref.get(state)
    const presentation = yield* Ref.get(model)
    const update = yield* updates.state
    yield* tray.setMenu(buildTrayMenu({ service: current?._tag ?? "Unknown", model: presentation, updateReady: (update.transfer._tag === "Ready" || update.transfer._tag === "InstallationFailed") }, {
      open: page => run(show(page)),
      stopModel: () => run(PubSub.publish(actions, { _tag: "StopModel" })),
      quit: requestQuit,
      restartUpdate: () => run(updates.requireReady.pipe(Effect.zipRight(Queue.offer(quit, "RestartUpdate")), Effect.catchAll(() => Effect.void))),
    }))
    if (Option.isSome(statusRow) && Option.isSome(presentation.status)) statusRow.value.present(presentation.status.value)
  })
  yield* refreshTray
  yield* updates.changes.pipe(Stream.map(value => (value.transfer._tag === "Ready" || value.transfer._tag === "InstallationFailed")), Stream.changes,
    Stream.runForEach(() => refreshTray), Effect.forkScoped)
  if (process.platform === "linux") {
    const trayHost = Context.get(yield* Layer.build(linuxTrayHostLayer()), LinuxTrayHost)
    yield* trayHost.changes.pipe(Stream.runForEach(tray.observeHost), Effect.forkScoped)
  }
  const harnessEnvironment = yield* resolveHarnessEnvironment().pipe(Effect.provide(guardedCommandLayer(join(dirname(addonPath), "magnitude-command"))), Effect.forkScoped)
  const service = yield* makeApplicationService({ output: "DiagnosticTail", admission: "Supervised", runtime: applicationRuntime, profile,
    stateDirectory: stateDir, home: homedir(), environment: process.env }).pipe(Effect.provide(NodeSqliteDriverLayer))
  const snapshot = Effect.all({ service: service.state, tray: tray.state }).pipe(Effect.map(value => ({ version: 1 as const, pid: process.pid, endpoint, service: value.service, owner: { _tag: "Desktop" as const, tray: value.tray } })))
  const snapshots = Stream.zipLatest(service.changes, tray.changes).pipe(Stream.map(([service, tray]) => ({ version: 1 as const, pid: process.pid, endpoint, service, owner: { _tag: "Desktop" as const, tray } })))
  yield* service.changes.pipe(Stream.runForEach(current => Ref.set(state, current).pipe(Effect.zipRight(refreshTray))), Effect.forkScoped)
  const control: ApplicationControlOptions = { snapshot, update: action => Effect.gen(function* () {
    if (action === "check") yield* updateSchedule.check
    if (action === "download") yield* updates.download
    if (action === "discard") yield* updates.discard
    if (action === "install") yield* updates.requireReady
    return { state: yield* updates.state, afterReply: action === "install" ? Queue.offer(quit, "RestartUpdate").pipe(Effect.asVoid) : Effect.void }
  }).pipe(Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message }))), login: action => action === "read" ? loginStartup.read : loginStartup.set(action === "enable"), dispatch: intent => intent === "Quit" ? Queue.offer(quit, "Quit").pipe(Effect.asVoid) : intent === "ShowWindow" ? show() : intent === "Retry" ? service.retry : Effect.void }
  if (process.platform === "win32") {
    const name = yield* Schema.decodeUnknown(WindowsPipeName)(owner.socketPath)
    yield* serveWindowsApplicationControl(name, control).pipe(Effect.provide(nativeWindowsPrivatePipesLayer(addonPath)))
  } else yield* serveApplicationControl(owner.socketPath, control)
  const connectionError = (error: { readonly message: string }) => new HostError({ message: error.message })
  const handlers = InferenceHostRpcs.toLayer({
    Updates: () => updates.changes,
    SetAutoDownload: ({ enabled }) => preferenceWrites.withPermits(1)(updates.setAutoDownload(enabled)).pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError), Effect.as({})),
    CheckUpdate: () => updateSchedule.check.pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError), Effect.as({})),
    DownloadUpdate: () => updates.download.pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError), Effect.as({})),
    DiscardUpdate: () => updates.discard.pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError), Effect.as({})),
    RestartUpdate: () => updates.requireReady.pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError), Effect.zipRight(Effect.gen(function* () {
      if (!window || window.isDestroyed() || !window.isVisible() || window.isMinimized() || systemShutdownRequested) {
        return yield* new HostError({ message: "Open Magnitude before choosing Restart to update." })
      }
      yield* Queue.offer(quit, "RestartUpdate")
      return {}
    }))),
    Observe: () => snapshots,
    Actions: () => Stream.concat(Stream.succeed({ _tag: "Navigate" as const, page: pendingPage }), Stream.fromPubSub(actions)),
    PresentModel: value => Ref.set(model, value).pipe(Effect.zipRight(refreshTray), Effect.as({})),
    GetAppearance: () => appearance.read.pipe(Effect.tapError(() => Effect.sync(() => { nativeTheme.themeSource = "system" })),
      Effect.mapError(connectionError), Effect.tap(preference =>
      Effect.sync(() => { nativeTheme.themeSource = preference }))),
    SetAppearance: ({ preference }) => preferenceWrites.withPermits(1)(appearance.write(preference)).pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError),
      Effect.tap(() => Effect.sync(() => { nativeTheme.themeSource = preference })), Effect.as({})),
    Relaunch: () => Queue.offer(quit, "Relaunch").pipe(Effect.as({})),
    LoginStartup: () => Stream.repeatEffectWithSchedule(loginStartup.read.pipe(Effect.map(state => state._tag === "Unavailable" ? { ...state, message: isolatedProfile || !app.isPackaged ? "Launch at login isn’t available in this development or test build. Install Magnitude to enable it." : "Launch at login needs attention. Check Magnitude in your system startup settings." } : state), Effect.tapError(Effect.logError), Effect.catchAll(() => Effect.succeed({ _tag: "Unavailable" as const, message: "Couldn’t check launch at login. Check Magnitude in your system startup settings." }))), Schedule.spaced("2 seconds")).pipe(Stream.mapError(connectionError)),
    SetLoginStartup: ({ enabled }) => loginStartup.set(enabled).pipe(Effect.tapError(Effect.logError), Effect.mapError(connectionError), Effect.as({})),
    Retry: () => service.retry.pipe(Effect.as({})),
    Quit: () => Queue.offer(quit, "Quit").pipe(Effect.as({})),
    ResolveQuitFailure: ({ decision }) => Ref.getAndSet(pendingQuitDecision, Option.none()).pipe(
      Effect.flatMap(Option.match({ onNone: () => Effect.void, onSome: pending => Deferred.succeed(pending, decision) })), Effect.as({})),
  })
  yield* RpcServer.layer(InferenceHostRpcs).pipe(Layer.provide(handlers), Layer.provide(makeElectronRpcServerLayer(ipcMain)), Layer.build)
  window = yield* Effect.acquireRelease(Effect.sync(() => {
    const value = new BrowserWindow({ ...windowChrome(process.platform, nativeTheme.shouldUseDarkColors), width: 1120, height: 800, minWidth: 800, minHeight: 600, show: false, title: "Magnitude", icon: app.isPackaged ? join(process.resourcesPath, "application-icon.png") : join(root, "assets/brand/application-icon.png"), backgroundColor: nativeTheme.shouldUseDarkColors ? slate[925] : slate[50], webPreferences: { preload: join(here, "../preload/preload.mjs"), contextIsolation: true, nodeIntegration: false, sandbox: false, backgroundThrottling: false } })
    const syncWindowAppearance = () => {
      value.setBackgroundColor(nativeTheme.shouldUseDarkColors ? slate[925] : slate[50])
      if (process.platform === "win32") value.setTitleBarOverlay(windowControlColors(nativeTheme.shouldUseDarkColors))
    }
    nativeTheme.on("updated", syncWindowAppearance)
    value.once("closed", () => nativeTheme.removeListener("updated", syncWindowAppearance))
    value.on("close", event => { if (!exiting) { event.preventDefault(); value.hide() } })
    value.webContents.on("render-process-gone", () => run(Effect.gen(function* () {
      yield* Ref.set(model, unavailableModel("Model status unavailable"))
      const current = yield* Ref.get(state)
      if (exiting || current?._tag === "Stopping" || current?._tag === "Stopped") return
      const retry = yield* rendererRecovery.crashed
      if (!retry) yield* Ref.set(model, unavailableModel("Window unavailable · Open Magnitude to retry"))
      yield* refreshTray
      if (retry) yield* loadRenderer()
    })))
    value.webContents.on("did-fail-load", (_event, code, _description, _url, isMainFrame) => {
      if (!isMainFrame || code === -3 || exiting) return
      run(rendererRecovery.loadFailed.pipe(
        Effect.zipRight(Ref.set(model, unavailableModel("Window unavailable · Open Magnitude to retry"))),
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
        Effect.catchAll(() => showNotice({ title: "Could not open model source", description: "Open your browser and try the source link again." })),
      ))
      return { action: "deny" }
    })
    value.webContents.on("console-message", (_event, level, message) => console.log(`[renderer:${level}] ${message}`))
    value.webContents.on("preload-error", (_event, path, error) => console.error(path, error))
    return value
  }), value => Effect.sync(() => value.destroy()))
  const cliLink = process.platform === "darwin" && app.isPackaged && !isolatedProfile && app.isInApplicationsFolder()
    ? yield* Effect.gen(function* () {
      const environment = yield* Fiber.join(harnessEnvironment)
      return yield* makeMacCliRegistration({ home: homedir(), resourcesDirectory: process.resourcesPath, environment })
    }).pipe(Effect.provide(NodeContext.layer)) : undefined
  const installCli = cliLink?.install ?? Effect.void
  const cliResult = (operation: typeof installCli) => operation.pipe(
    Effect.tapError(Effect.logError),
    Effect.catchAll(() => showNotice({ title: "Couldn’t install the command-line tool", description: "Magnitude couldn’t register its terminal command. Check that your application is installed in a writable location." })),
  )
  Menu.setApplicationMenu(Menu.buildFromTemplate(buildApplicationMenu(process.platform, {
    open: page => run(show(page)), quit: requestQuit,
    ...(cliLink ? { commandLine: {
      install: () => run(cliResult(installCli)),
      remove: () => run(cliResult(cliLink.remove)),
    } } : {}),
  })))
  yield* loadRenderer()
  // Initial activation belongs to launch intent. Subsequent Dock activation is an explicit Open.
  const activate = () => run(show())
  app.on("activate", activate)
  yield* Effect.addFinalizer(() => Effect.sync(() => app.removeListener("activate", activate)))
  if (wantsWindow) yield* show()
  if (cliLink && wantsWindow && !startupUpdateDeferred) yield* installCli.pipe(
    Effect.catchAll(error => Effect.logWarning("Could not install the magnitude command", error.message)),
  ).pipe(Effect.forkScoped)
  for (;;) {
    const intent = yield* Queue.take(quit)
    reopenAfterUpdate = BrowserWindow.getAllWindows().some(window => window.isVisible())
    yield* updates.close
    const stopped = yield* service.shutdown.pipe(Effect.either)
    if (stopped._tag === "Right") {
      if (systemShutdownRequested || intent === "Quit") return "Quit" as const
      if (intent === "RestartUpdate" && process.platform === "darwin") return "InstallMacUpdate" as const
      if (intent === "Relaunch" || !restartPreparedUpdate) return "Relaunch" as const
      const installation = yield* restartPreparedUpdate({ continuation: { _tag: "Desktop", showWindow: reopenAfterUpdate }, allowAuthorizationPrompt: true }).pipe(Effect.either)
      if (installation._tag === "Left") yield* Effect.logError(installation.left.message)
      return installation._tag === "Right" && installation.right === "Started" ? "RestartUpdate" as const : "Relaunch" as const
    }
    if (systemShutdownRequested) yield* Effect.logError(stopped.left.message)
    else {
      const retry = yield* resolveQuitFailure(stopped.left.message, {
        decide: decideQuitFailure,
        forceQuit: () => { exiting = true; app.exit(1) },
      })
      if (retry) yield* Queue.offer(quit, "Quit")
    }
  }
})).pipe(Effect.provide(nativeHostLayer(addonPath)), Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))
Effect.runPromiseExit(program).then(Exit.match({
  onSuccess: intent => {
    exiting = true
    if (intent === "Relaunch") {
      // The renderer dev server dies with this process; its supervising script performs the relaunch.
      if (rendererServedByDevServer(process.env)) { app.exit(developmentRelaunchExitCode({ showWindow: reopenAfterUpdate })); return }
      // `args` replaces the argument list, so keep the original ones (the app directory in development).
      app.relaunch({ args: [...process.argv.slice(1).filter(argument => argument !== "--background"), ...(reopenAfterUpdate ? [] : ["--background"])] })
    }
    if (intent === "InstallMacUpdate") {
      const install = Effect.scoped(Effect.gen(function* () {
        const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: dataDir, override: Option.fromNullable(stateOverride) })
        yield* acquireApplicationMaintenance(stateDirectory)
        yield* acquireUpdateInstallationLease(stateDirectory)
        const architecture = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
        return yield* startMacForegroundInstallation({ resources: process.resourcesPath, stateDirectory, dataDirectory: dataDir,
          version: app.getVersion(), architecture, operation: macUpdateOperation, continuation: { _tag: "Desktop", showWindow: reopenAfterUpdate } })
      })).pipe(Effect.provide([NodeContext.layer, nativeHostLayer(addonPath)]))
      void Effect.runPromiseExit(install).then(result => {
        if (Exit.isFailure(result)) {
          console.error(Cause.pretty(result.cause))
          if (reopenAfterUpdate) dialog.showErrorBox("Magnitude could not install the update", "The update remains pending. Open Magnitude and try again.")
        }
        app.quit()
      })
      return
    }
    app.quit()
  },
  onFailure: cause => {
    const message = Cause.pretty(cause)
    console.error(message)
    if (!canPresentErrors) { exiting = true; app.exit(1); return }
    void app.whenReady().then(() => {
      if (!background && !systemShutdownRequested) dialog.showErrorBox("Magnitude couldn’t continue", "Quit Magnitude and open it again. If it still won’t start, reinstall the application. Your downloaded models and configuration are stored separately.")
      exiting = true
      app.exit(1)
    })
  },
}))
