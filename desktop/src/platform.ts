/**
 * Desktop Platform implementation — spec §5.3
 *
 * Wraps the `__magnitudeDesktop` DesktopApi exposed by the preload bridge.
 * ACN process management remains one contract across the Electron boundary.
 */
import { Effect, Exit, Layer, Option, Schema, Scope, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  DaemonSpawnFailed,
  DaemonError,
  AcnProcessManager,
  makeAcnJitRuntime,
  type AcnLaunchEvent,
  type AcnProcessManager as AcnProcessManagerService,
} from "@magnitudedev/sdk"
import type { Platform, Storage, Clipboard, Notification, Dialogs } from "@magnitudedev/client-common"
import type { DesktopApi, MenuAction } from "./desktop-rpc"

const DEFAULT_SERVER_KEY = "default-server"

const desktopStorage: Storage = {
  async getItem(key: string): Promise<string | null> {
    return api.storage.getItem(key)
  },
  async setItem(key: string, value: string): Promise<void> {
    await api.storage.setItem(key, value)
  },
  async removeItem(key: string): Promise<void> {
    await api.storage.removeItem(key)
  },
}

const desktopClipboard: Clipboard = {
  async readText(): Promise<string> {
    return api.clipboard.readText()
  },
  async writeText(text: string): Promise<void> {
    await api.clipboard.writeText(text)
  },
}

const desktopNotifications: Notification = {
  show(title: string, body: string): void {
    api.notifications.show(title, body)
  },
}

const desktopDialogs: Dialogs = {
  async openDirectory(): Promise<string | null> {
    return api.dialogs.openDirectory()
  },
  async openFile(options?: { multiple?: boolean }): Promise<string[] | null> {
    return api.dialogs.openFile(options)
  },
}

// Late-bound reference to the desktop API
let api: DesktopApi

const errorMessage = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause)

const daemonError = (cause: unknown) => Schema.is(DaemonError)(cause)
  ? cause
  : new DaemonSpawnFailed({ reason: errorMessage(cause) })

function createDesktopAcnProcessManager(desktopApi: DesktopApi): AcnProcessManagerService {
  return AcnProcessManager.of({
    observeCurrent:
      Effect.tryPromise({
        try: async () => {
          const instance = await desktopApi.acnProcessManager.current()
          return Option.fromNullable(instance)
        },
        catch: daemonError,
      }),
    launch: (request) =>
      Stream.asyncPush<AcnLaunchEvent, DaemonError>((emit) =>
        Effect.tryPromise({
          try: () => desktopApi.acnProcessManager.launch(request, (event) => emit.single(event)),
          catch: daemonError,
        }).pipe(
          Effect.match({
            onFailure: emit.fail,
            onSuccess: () => emit.end(),
          }),
          Effect.forkScoped,
        ),
      ),
    terminate: (instance) => Effect.tryPromise({
      try: () => desktopApi.acnProcessManager.terminate(instance),
      catch: daemonError,
    }),
  })
}

export async function createDesktopPlatform(desktopApi: DesktopApi): Promise<Platform> {
  api = desktopApi
  const processManager = createDesktopAcnProcessManager(desktopApi)
  const acnScope = await Effect.runPromise(Scope.make())
  const acn = await Effect.runPromise(
    makeAcnJitRuntime().pipe(
      Effect.provideService(AcnProcessManager, processManager),
      Effect.provideService(Scope.Scope, acnScope),
      Effect.provide(FetchHttpClient.layer),
    ),
  )
  const protocolLayer = acn.protocolLayer.pipe(Layer.provide(FetchHttpClient.layer))
  const shutdown = () => Effect.runPromise(
    acn.close.pipe(Effect.ensuring(Scope.close(acnScope, Exit.void))),
  )
  return {
    id: "desktop",
    protocolLayer,
    acnStartup: acn.startup,
    shutdown,
    clipboard: desktopClipboard,
    storage: desktopStorage,
    notifications: desktopNotifications,
    dialogs: desktopDialogs,
    async openLink(url: string): Promise<void> {
      await api.openExternal(url)
    },
    async openPath(path: string): Promise<void> {
      await api.openPath(path)
    },
    showItemInFolder(path: string): void {
      api.showItemInFolder?.(path)
    },
    fetch: globalThis.fetch.bind(globalThis),
    async getDefaultServer(): Promise<string | null> {
      return api.storage.getItem(DEFAULT_SERVER_KEY)
    },
    async setDefaultServer(url: string): Promise<void> {
      await api.storage.setItem(DEFAULT_SERVER_KEY, url)
    },
    onMenuAction(cb: (action: MenuAction) => void): () => void {
      return api.onMenuAction(cb)
    },
    quit(): void {
      void shutdown().finally(() => {
        api.quit()
      })
    },
  }
}
