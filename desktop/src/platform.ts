/**
 * Desktop Platform implementation — spec §5.3
 *
 * Wraps the `__magnitudeDesktop` DesktopApi exposed by the preload bridge.
 * ACN ensurance remains one contract across the Electron boundary.
 */
import { Effect, Exit, Schema, Scope, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  AcnEnsuranceFailed,
  AcnAdministrationFailed,
  AcnEnsuranceError,
  AcnInstanceManager,
  makeAcnConnection,
  type AcnConnection,
  type AcnEnsureEvent,
  type AcnInstanceManager as AcnInstanceManagerService,
} from "@magnitudedev/sdk"
import type {
  Platform,
  Storage,
  Clipboard,
  Notification,
  Dialogs,
  EmbeddedBrowserCapability,
  BrowserWorkspaceState,
} from "@magnitudedev/client-common"
import type { DesktopApi, MenuAction } from "./desktop-rpc"
import {
  decodeDesktopAcnEnsureEvent,
  decodeDesktopBrowserWorkspaceState,
} from "./desktop-rpc"

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

const ensuranceError = (cause: unknown) =>
  Schema.is(AcnEnsuranceError)(cause)
    ? cause
    : new AcnEnsuranceFailed({ reason: errorMessage(cause) })

function createDesktopAcnManager(
  desktopApi: DesktopApi,
): AcnInstanceManagerService {
  return AcnInstanceManager.of({
    ensure: (request) =>
      Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceError>((emit) =>
        Effect.acquireRelease(
          Effect.sync(() =>
            desktopApi.acnEnsurer.ensure(
              request,
              (event) => {
                try {
                  emit.single(decodeDesktopAcnEnsureEvent(event))
                } catch (cause) {
                  emit.fail(ensuranceError(cause))
                }
              },
              (error) => emit.fail(ensuranceError(error)),
              () => emit.end(),
            ),
          ),
          (unsubscribe) => Effect.sync(unsubscribe),
        ).pipe(Effect.asVoid),
      ),
    stop: Effect.fail(
      new AcnAdministrationFailed({
        reason: "Desktop renderer cannot administer the ACN",
      }),
    ),
  })
}

export async function createDesktopAcnConnection(
  desktopApi: DesktopApi,
): Promise<AcnConnection> {
  const manager = createDesktopAcnManager(desktopApi)
  const acnScope = await Effect.runPromise(Scope.make())
  const connection = await Effect.runPromise(
    makeAcnConnection().pipe(
      Effect.provideService(AcnInstanceManager, manager),
      Effect.provideService(Scope.Scope, acnScope),
      Effect.provide(FetchHttpClient.layer),
    ),
  )
  const close = await Effect.runPromise(Effect.cached(
    connection.close.pipe(Effect.ensuring(Scope.close(acnScope, Exit.void))),
  ))
  return { ...connection, close }
}

export async function createDesktopPlatform(
  desktopApi: DesktopApi,
  connection: AcnConnection,
): Promise<Platform> {
  api = desktopApi
  let browserSnapshot: BrowserWorkspaceState | null = null
  const browserListeners = new Set<() => void>()
  let resolveInitialBrowser: ((state: BrowserWorkspaceState) => void) | null = null
  let rejectInitialBrowser: ((cause: unknown) => void) | null = null
  const initialBrowser = new Promise<BrowserWorkspaceState>((resolve, reject) => {
    resolveInitialBrowser = resolve
    rejectInitialBrowser = reject
  })
  const unsubscribeBrowser = desktopApi.browser.observe(
    (encoded) => {
      const state = decodeDesktopBrowserWorkspaceState(encoded)
      if (browserSnapshot !== null && state.revision <= browserSnapshot.revision) return
      browserSnapshot = state
      resolveInitialBrowser?.(state)
      resolveInitialBrowser = null
      rejectInitialBrowser = null
      for (const listener of browserListeners) listener()
    },
    (cause) => {
      rejectInitialBrowser?.(cause)
      resolveInitialBrowser = null
      rejectInitialBrowser = null
      console.error("[desktop] Embedded browser state stream failed:", cause)
    },
    () => {
      rejectInitialBrowser?.(new Error("Embedded browser state stream ended."))
      resolveInitialBrowser = null
      rejectInitialBrowser = null
    },
  )
  browserSnapshot = await initialBrowser
  let pendingViewport: Parameters<EmbeddedBrowserCapability["setViewport"]>[0] | undefined
  let viewportFlush: Promise<void> | null = null
  const setBrowserViewport: EmbeddedBrowserCapability["setViewport"] = (bounds) => {
    pendingViewport = bounds
    if (viewportFlush !== null) return viewportFlush
    viewportFlush = (async () => {
      while (pendingViewport !== undefined) {
        const next = pendingViewport
        pendingViewport = undefined
        await desktopApi.browser.setViewport(next)
      }
    })().finally(() => {
      viewportFlush = null
    })
    return viewportFlush
  }
  const embeddedBrowser: EmbeddedBrowserCapability = {
    getSnapshot: () => {
      if (browserSnapshot === null) throw new Error("Embedded browser state is unavailable.")
      return browserSnapshot
    },
    subscribe: (listener) => {
      browserListeners.add(listener)
      return () => browserListeners.delete(listener)
    },
    createTab: (url) => desktopApi.browser.createTab(url),
    activateTab: (tabId) => desktopApi.browser.activateTab(tabId),
    closeTab: (tabId) => desktopApi.browser.closeTab(tabId),
    navigate: (input) => desktopApi.browser.navigate(input),
    goBack: () => desktopApi.browser.goBack(),
    goForward: () => desktopApi.browser.goForward(),
    reload: () => desktopApi.browser.reload(),
    stop: () => desktopApi.browser.stop(),
    continueInsecureNavigation: () => desktopApi.browser.continueInsecureNavigation(),
    cancelInsecureNavigation: () => desktopApi.browser.cancelInsecureNavigation(),
    setViewport: setBrowserViewport,
    openExternal: () => desktopApi.browser.openExternal(),
    respondToPermission: (requestId, allow) =>
      desktopApi.browser.respondToPermission(requestId, allow),
    cancelDownload: (downloadId) => desktopApi.browser.cancelDownload(downloadId),
    revealDownload: (downloadId) => desktopApi.browser.revealDownload(downloadId),
  }
  return {
    id: "desktop",
    clipboard: desktopClipboard,
    storage: desktopStorage,
    notifications: desktopNotifications,
    dialogs: desktopDialogs,
    embeddedBrowser,
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
      unsubscribeBrowser()
      browserListeners.clear()
      void Effect.runPromise(connection.close).finally(() => {
        api.quit()
      })
    },
  }
}
