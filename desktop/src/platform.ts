/**
 * Desktop Platform implementation — spec §5.3
 *
 * Wraps the `__magnitudeDesktop` DesktopApi exposed by the preload bridge.
 * `DaemonSpawner` delegates daemon lifecycle to Electron main over IPC.
 */
import { Effect, Layer, Option } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { RpcClient } from "@effect/rpc"
import { DaemonSpawnFailed, DaemonSpawnerTag, recoveringProtocolLayer, type DaemonSpawner } from "@magnitudedev/sdk"
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

function createDesktopDaemonSpawner(desktopApi: DesktopApi): DaemonSpawner {
  return {
    discover: () =>
      Effect.tryPromise({
        try: async () => {
          const url = await desktopApi.daemon.discover()
          return url === null ? Option.none<string>() : Option.some(url)
        },
        catch: (cause) => new DaemonSpawnFailed({ reason: errorMessage(cause) }),
      }),
    spawn: (command) =>
      Effect.tryPromise({
        try: () => desktopApi.daemon.spawn(command),
        catch: (cause) => new DaemonSpawnFailed({ reason: errorMessage(cause) }),
      }),
  }
}

export function createDesktopPlatform(desktopApi: DesktopApi): Platform {
  api = desktopApi
  const daemonSpawnerLayer = Layer.succeed(DaemonSpawnerTag, createDesktopDaemonSpawner(desktopApi))
  const protocolLayer = recoveringProtocolLayer().pipe(
    Layer.provide(Layer.mergeAll(FetchHttpClient.layer, daemonSpawnerLayer)),
  )
  return {
    id: "desktop",
    protocolLayer,
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
      api.quit()
    },
  }
}
