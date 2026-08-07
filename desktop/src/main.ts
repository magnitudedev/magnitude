/**
 * Electron main entry — spec §5.1
 *
 * Responsibilities:
 * 1. Bundle path discovery — find the magnitude-acn binary
 * 2. ACN process management through one Effect-native service
 * 3. OS shell integration — BrowserWindow, preload, menu shortcuts
 *
 * The main process does NOT proxy ACN RPC traffic. It exposes desktop
 * platform actions and daemon boundaries through DesktopRpcs over Electron IPC;
 * the renderer SDK opens the ACN RPC connection directly to the endpoint
 * returned by that service.
 */
import { app, BrowserWindow, dialog, ipcMain, Menu, Notification, type MenuItemConstructorOptions } from "electron"
import * as nodePath from "node:path"
import * as nodeFs from "node:fs"
import { fileURLToPath } from "node:url"
import { spawn } from "node:child_process"
import { Array as Arr, Cause, Duration, Effect, Exit, Layer, Option, PubSub, Schedule, Scope, Stream } from "effect"
import { RpcServer } from "@effect/rpc"
import { FetchHttpClient } from "@effect/platform"
import { layer as nodeFileSystemLayer } from "@effect/platform-node-shared/NodeFileSystem"
import { layer as nodeCommandExecutorLayer } from "@effect/platform-node-shared/NodeCommandExecutor"
import { layer as nodePathLayer } from "@effect/platform-node-shared/NodePath"
import { inheritLoginShellEnv } from "./shell-env"
import { DesktopRpcError, DesktopRpcs, type MenuAction } from "./desktop-rpc"
import { makeElectronRpcServerLayer } from "./electron-rpc"
import { NodeSqliteDriverLayer } from "./sqlite-driver"

// SDK imports — these run in the main process (Node)
import {
  makeLocalAcnInstanceManager,
  ChildProcessSpawner,
  scopeAcnCandidate,
  AcnEnsuranceFailed,
  SDK_ACN_TARGET,
  type AcnInstanceManager as AcnInstanceManagerService,
} from "@magnitudedev/sdk"

// ESM doesn't have __dirname — polyfill it
const __dirname = nodePath.dirname(fileURLToPath(import.meta.url))

let mainWindow: BrowserWindow | null = null
let acnManagerPromise: Promise<AcnInstanceManagerService> | null = null
const acnEnsurerScope = Effect.runPromise(Scope.make())
const menuActions = Effect.runSync(PubSub.unbounded<MenuAction>())

/**
 * Node-compatible child process spawner for the local daemon launcher.
 * Uses child_process.spawn (NOT Bun.spawn) because Electron's main process is Node.
 */
const nodeSpawn: ChildProcessSpawner = {
  spawn: (command) =>
    Effect.uninterruptible(
      Effect.gen(function* () {
        const proc = yield* Effect.async<
          ReturnType<typeof spawn>,
          AcnEnsuranceFailed
        >((resume) => {
          const [executable, ...args] = command
          const proc = spawn(executable, args, {
            detached: true,
            stdio: ["pipe", "ignore", "ignore"],
            env: process.env,
          })
          const cleanup = () => {
            proc.off("spawn", onSpawn)
            proc.off("error", onError)
          }
          const onSpawn = () => {
            cleanup()
            resume(Effect.succeed(proc))
          }
          const onError = (cause: Error) => {
            cleanup()
            resume(Effect.fail(new AcnEnsuranceFailed({
                reason: `Failed to spawn Magnitude: ${cause.message}`,
            })))
          }
          proc.once("spawn", onSpawn)
          proc.once("error", onError)
          return Effect.sync(cleanup)
        })
        const pid = proc.pid
        if (pid === undefined) {
          return yield* new AcnEnsuranceFailed({
            reason: "Spawned ACN has no process ID",
          })
        }
        const exitedPromise = new Promise<number>((resolve) => {
          proc.once("close", (code) => resolve(code ?? 1))
          proc.once("error", () => resolve(1))
        })
        const exited = Effect.promise(() => exitedPromise)
        proc.unref()

        const signalTree = (name: NodeJS.Signals) =>
          Effect.try({
            try: () => {
              try {
                if (process.platform === "win32") {
                  if (!proc.kill(name)) throw new Error(`process ${pid} rejected ${name}`)
                } else {
                  process.kill(-pid, name)
                }
              } catch (cause) {
                if (cause instanceof Error && "code" in cause && cause.code === "ESRCH") return
                throw cause
              }
            },
            catch: (cause) =>
              new AcnEnsuranceFailed({
                reason: `Failed to send ${name} to ACN ${pid}: ${String(cause)}`,
              }),
          })
        const treeAbsent = Effect.sync(() => {
          if (process.platform === "win32") return proc.exitCode !== null
          try {
            process.kill(-pid, 0)
            return false
          } catch (cause) {
            if (cause instanceof Error && "code" in cause) {
              if (cause.code === "ESRCH") return true
              if (cause.code === "EPERM") return false
            }
            throw cause
          }
        })
        const waitForTreeAbsence = (duration: Duration.DurationInput) => treeAbsent.pipe(
          Effect.flatMap((absent) => absent ? Effect.void : Effect.fail("TreePresent" as const)),
          Effect.retry(Schedule.spaced(Duration.millis(20))),
          Effect.timeoutOption(duration),
          Effect.catchAll(() => Effect.succeed(Option.none())),
          Effect.map(Option.isSome),
        )
        const stopAndReap = Effect.gen(function* () {
          if (yield* treeAbsent) return
          yield* signalTree("SIGTERM")
          if (yield* waitForTreeAbsence(Duration.seconds(2))) return
          yield* signalTree("SIGKILL")
          if (!(yield* waitForTreeAbsence(Duration.seconds(2)))) {
            return yield* new AcnEnsuranceFailed({
              reason: `ACN candidate tree ${pid} did not exit after SIGKILL`,
            })
          }
        })
        const releaseParentChannel = Effect.async<void, AcnEnsuranceFailed>((resume) => {
          const stdin = proc.stdin
          if (stdin === null) {
            resume(
              Effect.fail(
                new AcnEnsuranceFailed({
                  reason: "ACN bootstrap pipe is unavailable",
                }),
              ),
            )
            return
          }
          const onError = (cause: Error) => {
            stdin.off("error", onError)
            resume(
              Effect.fail(
                new AcnEnsuranceFailed({
                  reason: `Failed to hand off ACN bootstrap: ${cause.message}`,
                }),
              ),
            )
          }
          stdin.once("error", onError)
          stdin.end(() => {
            stdin.off("error", onError)
            resume(Effect.void)
          })
          return Effect.sync(() => stdin.off("error", onError))
        })
        return yield* scopeAcnCandidate({
          pid,
          exited,
          releaseParentChannel,
          stopAndReap,
        })
      }),
    ),
}

/**
 * Full platform layer for the Electron main process (Node):
 * - FetchHttpClient for HTTP transport
 * - NodeFileSystem for filesystem access
 * - NodeCommandExecutor for command execution (binary version check)
 */
// NodeCommandExecutor depends on FileSystem, so we provide the FileSystem
// layer into it first, then merge everything together.
const nodePlatformLayer = Layer.mergeAll(
  FetchHttpClient.layer,
  nodeFileSystemLayer,
  nodeCommandExecutorLayer.pipe(Layer.provide(nodeFileSystemLayer)),
  nodePathLayer,
)

/** Storage map for the preload bridge (simple in-memory + file-backed) */
const storageDir = nodePath.join(app.getPath("userData"), "storage")
try {
  nodeFs.mkdirSync(storageDir, { recursive: true })
} catch {}

function storageFile(key: string): string {
  return nodePath.join(storageDir, `${key}.json`)
}

function storageGet(key: string): string | null {
  try {
    return nodeFs.readFileSync(storageFile(key), "utf8") ?? null
  } catch {
    return null
  }
}

function storageSet(key: string, value: string): void {
  try {
    nodeFs.writeFileSync(storageFile(key), value, "utf8")
  } catch {}
}

function storageRemove(key: string): void {
  try {
    nodeFs.unlinkSync(storageFile(key))
  } catch {}
}

/**
 * Find the magnitude-acn binary path.
 * In production, it's bundled in process.resourcesPath.
 * In development, let the SDK or source launch command resolve the binary.
 */
function findBinaryPath(): Option.Option<string> {
  // Check for bundled binary in resources (production)
  const resourcesPath = process.resourcesPath
  if (resourcesPath) {
    const bundledPath = nodePath.join(resourcesPath, "magnitude-acn")
    if (nodeFs.existsSync(bundledPath)) {
      return Option.some(bundledPath)
    }
    // Also check platform-specific subdirectory
    const platformName = `${process.platform}-${process.arch}`
    const platformPath = nodePath.join(resourcesPath, "bin", platformName, "magnitude-acn")
    if (nodeFs.existsSync(platformPath)) {
      return Option.some(platformPath)
    }
  }

  // Let the SDK resolve/download its cache. Do not pass the SDK cache as an
  // explicit binaryPath, or version repair turns into explicit-path failure.
  return Option.none()
}

function sendMenuAction(action: MenuAction): void {
  Effect.runFork(PubSub.publish(menuActions, action).pipe(Effect.asVoid))
}

function buildMenu(): Menu {
  const isMac = process.platform === "darwin"
  const template: MenuItemConstructorOptions[] = [
    ...(isMac
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" as const },
              { type: "separator" as const },
              { role: "services" as const },
              { type: "separator" as const },
              { role: "hide" as const },
              { role: "hideOthers" as const },
              { role: "unhide" as const },
              { type: "separator" as const },
              { role: "quit" as const },
            ],
          },
        ]
      : []),
    {
      label: "File",
      submenu: [
        {
          label: "New Session",
          accelerator: "CmdOrCtrl+N",
          click: () => sendMenuAction({ _tag: "new-session" }),
        },
        { type: "separator" as const },
        isMac ? { role: "close" as const } : { role: "quit" as const },
      ],
    },
    { role: "editMenu" as const },
    {
      label: "View",
      submenu: [
        {
          label: "Focus Sidebar Search",
          accelerator: "CmdOrCtrl+R",
          click: () => sendMenuAction({ _tag: "toggle-sidebar-search" }),
        },
        {
          label: "Toggle Transcript Mode",
          accelerator: "CmdOrCtrl+T",
          click: () => sendMenuAction({ _tag: "toggle-transcript-mode" }),
        },
        { type: "separator" as const },
        { role: "reload" as const },
        { role: "forceReload" as const },
        { role: "toggleDevTools" as const },
        { type: "separator" as const },
        { role: "resetZoom" as const },
        { role: "zoomIn" as const },
        { role: "zoomOut" as const },
        { type: "separator" as const },
        { role: "togglefullscreen" as const },
      ],
    },
    {
      label: "Window",
      submenu: [
        {
          label: "Settings",
          accelerator: "CmdOrCtrl+,",
          click: () => sendMenuAction({ _tag: "open-settings" }),
        },
        { role: "minimize" as const },
        ...(!isMac ? [{ type: "separator" as const }, { role: "close" as const }] : []),
      ],
    },
  ]
  return Menu.buildFromTemplate(template)
}

function createWindow(): void {
  const isMac = process.platform === "darwin"

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    show: false,
    ...(isMac
      ? {
          titleBarStyle: "hidden" as const,
          trafficLightPosition: { x: 16, y: 16 },
          vibrancy: "sidebar" as const,
          visualEffectState: "active" as const,
          transparent: true,
          backgroundColor: "#00000000",
        }
      : {}),
    webPreferences: {
      preload: nodePath.join(__dirname, "../preload/preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      backgroundThrottling: false,
    },
  })

  // CSP header
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    const scriptSrc = app.isPackaged ? "script-src 'self'" : "script-src 'self' 'unsafe-inline'"
    const connectSrc = app.isPackaged
      ? "connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*"
      : "connect-src 'self' http://127.0.0.1:* http://localhost:* ws://127.0.0.1:* ws://localhost:*"
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        "Content-Security-Policy": [
          `default-src 'self'; ${scriptSrc}; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; ${connectSrc}`,
        ],
      },
    })
  })

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show()
  })

  if (!app.isPackaged) {
    mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
      console.log(`[desktop:renderer:${level}] ${message} (${sourceId}:${line})`)
    })
    mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
      console.error("[desktop] Renderer failed to load:", errorCode, errorDescription, validatedURL)
    })
    mainWindow.webContents.on("preload-error", (_event, preloadPath, error) => {
      console.error("[desktop] Preload failed:", preloadPath, error)
    })
    mainWindow.webContents.on("render-process-gone", (_event, details) => {
      console.error("[desktop] Renderer process gone:", details)
    })
  }

  // Load the renderer
  if (process.env["ELECTRON_RENDERER_URL"]) {
    // Dev mode — electron-vite serves the renderer
    mainWindow.loadURL(process.env["ELECTRON_RENDERER_URL"])
  } else {
    // Production — load the built renderer
    mainWindow.loadFile(nodePath.join(__dirname, "../renderer/index.html"))
  }

  // On window close, the renderer sends __magnitude:interrupt-stream
  // to notify main that the stream should be cleaned up (§5.6)
  mainWindow.on("closed", () => {
    mainWindow = null
  })
}

function defaultLaunchCommand(): Option.Option<Arr.NonEmptyReadonlyArray<string>> {
  const isDev = !app.isPackaged
  const acnSourcePath = nodePath.resolve(__dirname, "..", "..", "..", "packages", "acn", "src", "binary.ts")
  return isDev
    ? Option.some(["bun", acnSourcePath, "serve"])
    : Option.none()
}

function localDaemonOptions() {
  const binaryPath = findBinaryPath()
  return {
    ...Option.match(binaryPath, {
      onNone: () => ({}),
      onSome: (path) => ({ binaryPath: path }),
    }),
    ...Option.match(defaultLaunchCommand(), {
      onNone: () => ({}),
      onSome: (command) => ({
        launchOverride: {
          target: SDK_ACN_TARGET,
          command,
        },
      }),
    }),
  }
}

async function getAcnManager(): Promise<AcnInstanceManagerService> {
  const scope = await acnEnsurerScope
  acnManagerPromise ??= makeLocalAcnInstanceManager(localDaemonOptions()).pipe(
    Effect.provideService(ChildProcessSpawner, nodeSpawn),
    Effect.provideService(Scope.Scope, scope),
    Effect.provide(Layer.merge(nodePlatformLayer, NodeSqliteDriverLayer)),
    Effect.runPromise,
  )
  return acnManagerPromise
}

const acnManager = Effect.promise(getAcnManager)

function messageFromUnknown(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function desktopRpcError(cause: unknown): DesktopRpcError {
  return new DesktopRpcError({ message: messageFromUnknown(cause) })
}

const promiseRpc = <A>(operation: () => Promise<A>): Effect.Effect<A, DesktopRpcError> =>
  Effect.tryPromise({
    try: operation,
    catch: desktopRpcError,
  })

const DesktopRpcHandlersLive = DesktopRpcs.toLayer({
  AcnEnsure: (request) => Stream.unwrap(
    acnManager.pipe(
      Effect.map((manager) => manager.ensure(request)),
    ),
  ),
  StorageGet: ({ key }) => Effect.sync(() => storageGet(key)),
  StorageSet: ({ key, value }) => Effect.sync(() => storageSet(key, value)).pipe(Effect.as({})),
  StorageRemove: ({ key }) => Effect.sync(() => storageRemove(key)).pipe(Effect.as({})),
  DialogOpenDirectory: () =>
    promiseRpc(async () => {
      const result = await dialog.showOpenDialog({ properties: ["openDirectory"] })
      if (result.canceled || result.filePaths.length === 0) return null
      return result.filePaths[0]!
    }),
  DialogOpenFile: ({ multiple }) =>
    promiseRpc(async () => {
      const result = await dialog.showOpenDialog({
        properties: multiple ? ["openFile", "multiSelections"] : ["openFile"],
      })
      if (result.canceled || result.filePaths.length === 0) return null
      return result.filePaths
    }),
  NotificationShow: ({ title, body }) =>
    Effect.sync(() => {
      try {
        new Notification({ title, body }).show()
      } catch {
        // ignore notification errors; the old bridge treated this as best effort.
      }
    }).pipe(Effect.as({})),
  Quit: () => Effect.sync(() => app.quit()).pipe(Effect.as({})),
  InterruptStream: () => Effect.succeed({}),
  StreamMenuActions: () => Stream.fromPubSub(menuActions),
})

function startDesktopRpcServer(): void {
  const DesktopRpcServerLive = RpcServer.layer(DesktopRpcs).pipe(
    Layer.provide(DesktopRpcHandlersLive),
    Layer.provide(makeElectronRpcServerLayer(ipcMain)),
  )

  Effect.runFork(
    Layer.launch(DesktopRpcServerLive).pipe(
      Effect.catchAllCause((cause) =>
        Effect.sync(() => {
          console.error("[desktop] Desktop RPC server failed:", Cause.pretty(cause))
        })
      ),
    ),
  )
}

app.whenReady().then(() => {
  // 1. Resolve the login shell environment before any lazy ACN launch.
  inheritLoginShellEnv()

  // 2. Start the desktop RPC server BEFORE creating any window, regardless of
  //    daemon status. This keeps storage/quit/menu handlers available even on
  //    the daemon-error screen (§5.6).
  startDesktopRpcServer()

  // 3. Set up application menu
  Menu.setApplicationMenu(buildMenu())

  // 4. Create the window without touching ACN process state. The shared client
  // lifecycle selects or starts an ACN on first RPC demand.
  createWindow()
})

// App lifecycle — clean up on quit
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit()
  }
})

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length !== 0) return
  createWindow()
})

app.on("will-quit", () => {
  void acnEnsurerScope.then((scope) =>
    Effect.runPromise(Scope.close(scope, Exit.void)),
  )
})

// ── Client lease release on quit (spec §5.6) ─────────────────────────
//
// The renderer owns its ClientLease and attempts scoped release before unload.
// The main process has no client lease of its own. Lease expiry remains the
// correctness path if the renderer cannot finish asynchronous unload cleanup.
