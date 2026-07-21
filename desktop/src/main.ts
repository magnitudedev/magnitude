/**
 * Electron main entry — spec §5.1
 *
 * Responsibilities:
 * 1. Bundle path discovery — find the magnitude-acn binary
 * 2. Daemon lifecycle — discover/spawn via the Effect-native SDK spawner
 * 3. OS shell integration — BrowserWindow, preload, menu shortcuts
 *
 * The main process does NOT proxy ACN RPC traffic. It exposes desktop
 * platform actions and daemon lifecycle through DesktopRpcs over Electron IPC;
 * the renderer SDK opens the ACN RPC connection directly to the endpoint
 * returned by that spawner.
 */
import { app, BrowserWindow, dialog, ipcMain, Menu, Notification, type MenuItemConstructorOptions } from "electron"
import * as nodePath from "node:path"
import * as nodeFs from "node:fs"
import { fileURLToPath } from "node:url"
import { spawn } from "node:child_process"
import { Cause, Effect, Layer, Option, PubSub, Stream } from "effect"
import { RpcServer } from "@effect/rpc"
import { FetchHttpClient } from "@effect/platform"
import { layer as nodeFileSystemLayer } from "@effect/platform-node-shared/NodeFileSystem"
import { layer as nodeCommandExecutorLayer } from "@effect/platform-node-shared/NodeCommandExecutor"
import { inheritLoginShellEnv } from "./shell-env"
import { DesktopRpcError, DesktopRpcs, type MenuAction } from "./desktop-rpc"
import { makeElectronRpcServerLayer } from "./electron-rpc"

// SDK imports — these run in the main process (Node)
import {
  makeLocalDaemonSpawner,
  type SpawnProcess,
  type DaemonSpawner,
  NoDaemon,
  DaemonSpawnFailed,
  BinaryNotFound,
  BinaryVersionMismatch,
  RegistrationFileInvalid,
  DownloadFailed,
  ChecksumMismatch,
  DaemonCrashed,
} from "@magnitudedev/sdk"

// ESM doesn't have __dirname — polyfill it
const __dirname = nodePath.dirname(fileURLToPath(import.meta.url))

let mainWindow: BrowserWindow | null = null
let daemonUrl: string = ""
let daemonSpawnerPromise: Promise<DaemonSpawner> | null = null
const menuActions = Effect.runSync(PubSub.unbounded<MenuAction>())

/**
 * Node-compatible spawn function for makeLocalDaemonSpawner.
 * Uses child_process.spawn (NOT Bun.spawn) because Electron's main process is Node.
 */
const nodeSpawn: SpawnProcess = (command) => {
  const proc = spawn(command[0]!, command.slice(1), {
    detached: true,
    stdio: ["ignore", "ignore", "ignore"],
    env: process.env,
  })
  proc.unref()
  return {
    pid: proc.pid,
    exited: new Promise<number | null>((resolve) => {
      proc.on("exit", (code) => resolve(code))
    }),
  }
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
 * In development, let the SDK or source spawn command resolve the binary.
 */
function findBinaryPath(): string | undefined {
  // Check for bundled binary in resources (production)
  const resourcesPath = process.resourcesPath
  if (resourcesPath) {
    const bundledPath = nodePath.join(resourcesPath, "magnitude-acn")
    if (nodeFs.existsSync(bundledPath)) {
      return bundledPath
    }
    // Also check platform-specific subdirectory
    const platformName = `${process.platform}-${process.arch}`
    const platformPath = nodePath.join(resourcesPath, "bin", platformName, "magnitude-acn")
    if (nodeFs.existsSync(platformPath)) {
      return platformPath
    }
  }

  // Let the SDK resolve/download its cache. Do not pass the SDK cache as an
  // explicit binaryPath, or version repair turns into explicit-path failure.
  return undefined
}

/**
 * Format a user-friendly error message for a DaemonError.
 */
function formatDaemonError(err: unknown): string {
  if (err instanceof NoDaemon) {
    return "Could not connect to the Magnitude daemon. Please try again."
  }
  if (err instanceof DaemonSpawnFailed) {
    return `Failed to start the Magnitude daemon: ${err.reason}`
  }
  if (err instanceof BinaryNotFound) {
    return `The Magnitude ACN binary was not found at: ${err.path}`
  }
  if (err instanceof BinaryVersionMismatch) {
    return `Binary version mismatch. Expected ${err.expected}, found ${err.actual} at ${err.path}.`
  }
  if (err instanceof RegistrationFileInvalid) {
    return `Registration file is invalid at ${err.path}: ${err.reason}`
  }
  if (err instanceof DownloadFailed) {
    return `Failed to download the ACN binary: ${err.reason} (HTTP ${err.status})`
  }
  if (err instanceof ChecksumMismatch) {
    return `Checksum mismatch for ${err.path}. Expected ${err.expected}, got ${err.actual}.`
  }
  if (err instanceof DaemonCrashed) {
    return `The Magnitude daemon crashed with exit code ${err.exitCode}.`
  }
  if (err instanceof Error) {
    return err.message
  }
  return String(err)
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

function defaultSpawnCommand(): string[] | undefined {
  const isDev = !app.isPackaged
  const acnSourcePath = nodePath.resolve(__dirname, "..", "..", "..", "packages", "acn", "src", "binary.ts")
  return isDev ? ["bun", acnSourcePath, "serve", "--register"] : undefined
}

async function getDaemonSpawner(): Promise<DaemonSpawner> {
  if (daemonSpawnerPromise) return daemonSpawnerPromise

  const binaryPath = findBinaryPath()
  daemonSpawnerPromise = makeLocalDaemonSpawner(nodeSpawn, {
    ...(binaryPath !== undefined ? { binaryPath } : {}),
  }).pipe(
    Effect.provide(nodePlatformLayer),
    Effect.runPromise,
  )
  return daemonSpawnerPromise
}

async function discoverDaemonUrl(): Promise<string | null> {
  const spawner = await getDaemonSpawner()
  const existing = await spawner.discover().pipe(Effect.runPromise)
  if (Option.isNone(existing)) {
    daemonUrl = ""
    return null
  }

  daemonUrl = existing.value
  return existing.value
}

async function spawnDaemon(command: readonly string[] | undefined): Promise<string> {
  const spawner = await getDaemonSpawner()
  const spawned = await spawner.spawn(command ? [...command] : defaultSpawnCommand()).pipe(Effect.runPromise)
  daemonUrl = spawned
  return spawned
}

async function ensureDaemonRunning(): Promise<string> {
  const existing = await discoverDaemonUrl()
  if (existing) {
    return existing
  }

  return spawnDaemon(undefined)
}

async function daemonIpc<T>(operation: () => Promise<T>): Promise<T> {
  try {
    return await operation()
  } catch (err) {
    throw new Error(formatDaemonError(err))
  }
}

function messageFromUnknown(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function desktopRpcError(cause: unknown): DesktopRpcError {
  return new DesktopRpcError({ message: messageFromUnknown(cause) })
}

const daemonRpc = <A>(operation: () => Promise<A>): Effect.Effect<A, DesktopRpcError> =>
  Effect.tryPromise({
    try: () => daemonIpc(operation),
    catch: desktopRpcError,
  })

const promiseRpc = <A>(operation: () => Promise<A>): Effect.Effect<A, DesktopRpcError> =>
  Effect.tryPromise({
    try: operation,
    catch: desktopRpcError,
  })

const DesktopRpcHandlersLive = DesktopRpcs.toLayer({
  DaemonEnsure: () => daemonRpc(ensureDaemonRunning),
  DaemonDiscover: () => daemonRpc(discoverDaemonUrl),
  DaemonSpawn: ({ command }) => daemonRpc(() => spawnDaemon(command ?? undefined)),
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
  // 1. Resolve the login shell environment before any lazy ACN spawn.
  inheritLoginShellEnv()

  // 2. Start the desktop RPC server BEFORE creating any window, regardless of
  //    daemon status. This keeps storage/quit/menu handlers available even on
  //    the daemon-error screen (§5.6).
  startDesktopRpcServer()

  // 3. Set up application menu
  Menu.setApplicationMenu(buildMenu())

  // 4. Create the window without touching daemon lifecycle. The renderer's
  // shared SDK coordinator discovers or spawns on first RPC demand.
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

// ── Client lease release on quit (spec §5.6) ─────────────────────────
//
// The protocol does not expose a ReleaseLease or Disconnect RPC. The daemon
// tracks client connections via heartbeat and automatically reaps stale
// leases when the heartbeat stops. Since the Electron main process does not
// hold an open RPC connection (the renderer connects independently), there
// is no lease to release from main. The renderer's stream fiber is
// interrupted via `interruptStream()` in the `beforeunload` handler.
//
// No client disposal needed — the renderer manages its own connection.
// The daemon process is detached and persists beyond the app lifecycle.
