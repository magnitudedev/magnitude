/**
 * Terminal Platform implementation — CLI-specific.
 *
 * Uses Bun APIs for process spawning, clipboard (OSC 52), and terminal size.
 * Stubs for unsupported capabilities (storage, notifications, dialogs).
 */
import { Array as Arr, Effect, Layer, Option, Runtime, Scope } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  AcnInstanceManager,
  BunDetachedChildProcessSpawner,
  ChildProcessSpawner,
  makeAcnJitRuntime,
  makeLocalAcnInstanceManager,
  SDK_ACN_TARGET,
} from "@magnitudedev/sdk"
import { BunSqliteDriverLayer } from "@magnitudedev/sdk/bun"
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
    // Palette detection is owned by the interactive terminal runtime.
    return null
  },
  setTerminalTitle(title: string): void {
    process.stdout.write(`\x1b]2;${title}\x07`)
  },
}

export interface TerminalPlatformOptions {
  readonly launchCommand: Option.Option<Arr.NonEmptyReadonlyArray<string>>
  readonly debug: boolean
  readonly effectLoggingLayer: Option.Option<Layer.Layer<never, never, never>>
}

export interface TerminalPlatformRuntime {
  readonly platform: Platform
  readonly close: Effect.Effect<void>
}

const makeTerminalAcnInstanceManager = (
  debug: boolean,
  launchCommand: Option.Option<Arr.NonEmptyReadonlyArray<string>>,
) => makeLocalAcnInstanceManager({
  ...(debug ? { debug: true } : {}),
  ...Option.match(launchCommand, {
    onNone: () => ({}),
    onSome: (command) => ({
      launchOverride: {
        target: SDK_ACN_TARGET,
        command,
      },
    }),
  }),
}).pipe(
  Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
  Effect.provide([BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer]),
)

export const stopTerminalAcn = Effect.scoped(
  makeLocalAcnInstanceManager().pipe(
    Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
    Effect.provide([BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer]),
    Effect.flatMap((manager) => manager.stop),
  ),
)

export const makeTerminalPlatform = (
  options: TerminalPlatformOptions,
): Effect.Effect<TerminalPlatformRuntime, never, Scope.Scope> => Effect.gen(function* () {
  const effectLoggingLayer = Option.getOrElse(
    options.effectLoggingLayer,
    () => makeCliEffectLoggingLayer({ debug: options.debug }),
  )
  const manager = yield* makeTerminalAcnInstanceManager(
    options.debug,
    options.launchCommand,
  )
  const acn = yield* makeAcnJitRuntime().pipe(
    Effect.provideService(AcnInstanceManager, manager),
    Effect.provide(FetchHttpClient.layer),
  )
  const close = yield* Effect.cached(acn.close)
  yield* Effect.addFinalizer(() => close.pipe(Effect.ignore))
  const runtime = yield* Effect.runtime<never>()
  const runPromise = Runtime.runPromise(runtime)
  const transport = Layer.mergeAll(FetchHttpClient.layer, effectLoggingLayer)
  const protocolLayer = acn.protocolLayer.pipe(Layer.provide(transport))

  const platform: Platform = {
    id: "terminal",
    protocolLayer,
    acnStartup: acn.startup,
    shutdown: () => runPromise(close),
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

  return { platform, close }
})
