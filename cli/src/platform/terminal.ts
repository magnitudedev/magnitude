/**
 * Terminal Platform implementation — CLI-specific.
 *
 * Uses Bun APIs for process spawning, clipboard (OSC 52), and terminal size.
 * Stubs for unsupported capabilities (storage, notifications, dialogs).
 */
import { Effect, Exit, Layer, Option, Scope } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  AcnProcessManager,
  BunDetachedChildProcessSpawner,
  ChildProcessSpawner,
  makeLocalAcnDaemonAdministrator,
  makeAcnJitRuntime,
  makeLocalAcnProcessManager,
} from "@magnitudedev/sdk"
import type {
  Platform,
  Storage,
  Clipboard,
  Notification,
  Dialogs,
  TerminalCapabilities,
} from "@magnitudedev/client-common"
import { makeCliEffectLoggingLayer } from "./effect-logger"

const noopStorage: Storage = {
  async getItem() { return null },
  async setItem() {},
  async removeItem() {},
}

const osc52Clipboard: Clipboard = {
  async readText(): Promise<string> {
    // OSC 52 read is not reliably supported across terminals
    return ""
  },
  async writeText(text: string): Promise<void> {
    // OSC 52 clipboard write — works in most modern terminals
    const encoded = Buffer.from(text).toString("base64")
    process.stdout.write(`\x1b]52;c;${encoded}\x07`)
  },
}

const noopNotifications: Notification = {
  show() {},
}

const noopDialogs: Dialogs = {
  async openDirectory() { return null },
  async openFile() { return null },
}

const terminalCapabilities: TerminalCapabilities = {
  get width() { return process.stdout.columns ?? 80 },
  get height() { return process.stdout.rows ?? 24 },
  os: process.platform,
  onResize(cb: () => void): () => void {
    process.stdout.on("resize", cb)
    return () => { process.stdout.off("resize", cb) }
  },
  async getPalette() {
    // Palette detection is handled by the renderer in index.tsx
    return null
  },
  setTerminalTitle(title: string): void {
    process.stdout.write(`\x1b]2;${title}\x07`)
  },
}

export interface TerminalPlatformOptions {
  readonly launchCommand: Option.Option<ReadonlyArray<string>>
  readonly debug: boolean
  readonly effectLoggingLayer: Option.Option<Layer.Layer<never, never, never>>
}

const makeTerminalAcnProcessManager = (
  debug: boolean,
  scope: Scope.CloseableScope,
) => makeLocalAcnProcessManager({ ...(debug ? { debug: true } : {}) }).pipe(
  Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
  Effect.provideService(Scope.Scope, scope),
  Effect.provide([BunContext.layer, FetchHttpClient.layer]),
)

export async function stopTerminalAcn(): Promise<void> {
  const administrator = await Effect.runPromise(
    makeLocalAcnDaemonAdministrator().pipe(
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
    ),
  )
  await Effect.runPromise(administrator.stopCurrent)
}

export async function createTerminalPlatform(options: TerminalPlatformOptions): Promise<Platform> {
  const effectLoggingLayer = Option.getOrElse(
    options.effectLoggingLayer,
    () => makeCliEffectLoggingLayer({ debug: options.debug }),
  )
  const managerScope = await Effect.runPromise(Scope.make())
  const processManager = await Effect.runPromise(
    makeTerminalAcnProcessManager(options.debug, managerScope),
  )
  const acn = await Effect.runPromise(
    makeAcnJitRuntime({ launchCommand: options.launchCommand }).pipe(
      Effect.provideService(AcnProcessManager, processManager),
      Effect.provideService(Scope.Scope, managerScope),
      Effect.provide(FetchHttpClient.layer),
    ),
  )
  const transport = Layer.mergeAll(FetchHttpClient.layer, effectLoggingLayer)
  const protocolLayer = acn.protocolLayer.pipe(Layer.provide(transport))
  const shutdown = () => Effect.runPromise(
    acn.close.pipe(Effect.ensuring(Scope.close(managerScope, Exit.void))),
  )

  return {
    id: "terminal",
    protocolLayer,
    acnStartup: acn.startup,
    shutdown,
    clipboard: osc52Clipboard,
    storage: noopStorage,
    notifications: noopNotifications,
    dialogs: noopDialogs,
    async openLink(url: string): Promise<void> {
      const opener = process.platform === "darwin" ? "open" : "xdg-open"
      Bun.spawn([opener, url])
    },
    async openPath(path: string): Promise<void> {
      const opener = process.platform === "darwin" ? "open" : "xdg-open"
      Bun.spawn([opener, path])
    },
    showItemInFolder(path: string): void {
      if (process.platform === "darwin") {
        Bun.spawn(["open", "-R", path])
      }
    },
    fetch: globalThis.fetch.bind(globalThis),
    async getDefaultServer(): Promise<string | null> {
      return null
    },
    async setDefaultServer(): Promise<void> {},
    quit(): void {
      process.kill(process.pid, "SIGTERM")
    },
    terminal: terminalCapabilities,
  }
}
