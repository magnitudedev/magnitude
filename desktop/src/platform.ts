/**
 * Desktop Platform implementation — spec §5.3
 *
 * Wraps the `__magnitudeDesktop` DesktopApi exposed by the preload bridge.
 * ACN ensurance remains one contract across the Electron boundary.
 */
import { Effect, Exit, Layer, Schema, Scope, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  AcnEnsuranceFailed,
  AcnAdministrationFailed,
  AcnEnsuranceError,
  AcnInstanceManager,
  makeAcnJitRuntime,
  type AcnEnsureEvent,
  type AcnInstanceManager as AcnInstanceManagerService,
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

const ensuranceError = (cause: unknown) => Schema.is(AcnEnsuranceError)(cause)
  ? cause
  : new AcnEnsuranceFailed({ reason: errorMessage(cause) })

function createDesktopAcnManager(desktopApi: DesktopApi): AcnInstanceManagerService {
  return AcnInstanceManager.of({
    ensure: (request) => Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceError>((emit) =>
      Effect.acquireRelease(
        Effect.sync(() => desktopApi.acnEnsurer.ensure(
          request,
          (event) => emit.single(event),
          (error) => emit.fail(ensuranceError(error)),
          () => emit.end(),
        )),
        (unsubscribe) => Effect.sync(unsubscribe),
      ).pipe(Effect.asVoid),
    ),
    stop: Effect.fail(new AcnAdministrationFailed({
      reason: "Desktop renderer cannot administer the ACN",
    })),
  })
}

export async function createDesktopPlatform(desktopApi: DesktopApi): Promise<Platform> {
  api = desktopApi
  const manager = createDesktopAcnManager(desktopApi)
  const acnScope = await Effect.runPromise(Scope.make())
  const acn = await Effect.runPromise(
    makeAcnJitRuntime().pipe(
      Effect.provideService(AcnInstanceManager, manager),
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
