import {
  app,
  BrowserWindow,
  WebContentsView,
  session,
  shell,
  type DownloadItem,
  type WebContents,
} from "electron"
import { randomUUID } from "node:crypto"
import {
  BrowserDownloadIdSchema,
  BrowserPermissionRequestIdSchema,
  BrowserTabIdSchema,
  type BrowserDownloadId,
  type BrowserDownloadState,
  type BrowserPermissionRequest,
  type BrowserPermissionRequestId,
  type BrowserTabId,
  type BrowserTabPhase,
  type BrowserTabState,
  type BrowserViewportRect,
  type BrowserWorkspaceState,
} from "@magnitudedev/client-common/platform/embedded-browser"
import {
  Context,
  Data,
  Effect,
  Layer,
  Option,
  Runtime,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  browserNavigationFailureMessage,
  isAllowedBrowserNavigation,
  resolveBrowserNavigation,
} from "./embedded-browser-navigation"
import { embeddedBrowserShortcut } from "./embedded-browser-shortcuts"

const BROWSER_PARTITION = "persist:magnitude-browser"
const HIDDEN_DISCARD_DELAY_MS = 5 * 60 * 1_000
const PERMISSION_REQUEST_TIMEOUT_MS = 30_000
const PROMPTABLE_PERMISSIONS = new Set(["media", "geolocation", "notifications", "clipboard-read"])
const AUTOMATIC_PERMISSIONS = new Set(["clipboard-sanitized-write", "fullscreen"])

export class EmbeddedBrowserUnavailable extends Data.TaggedError(
  "EmbeddedBrowserUnavailable",
)<{ readonly reason: string }> {}

interface InternalTab {
  readonly id: BrowserTabId
  view: WebContentsView | null
  title: string
  url: string
  pendingUrl: string | null
  faviconUrl: string | null
  phase: BrowserTabPhase
  error: string | null
  insecureUrl: string | null
  allowedInsecureOrigin: string | null
  discardTimer: ReturnType<typeof setTimeout> | null
}

interface InternalPermissionRequest {
  readonly state: BrowserPermissionRequest
  readonly key: string
  readonly callbacks: Array<(allowed: boolean) => void>
  readonly timer: ReturnType<typeof setTimeout>
}

interface InternalDownload {
  readonly id: BrowserDownloadId
  readonly tabId: BrowserTabId
  readonly item: DownloadItem
  status: BrowserDownloadState["status"]
}

export interface EmbeddedBrowserService {
  readonly changes: Stream.Stream<BrowserWorkspaceState>
  readonly createTab: (url: string | null) => Effect.Effect<BrowserTabId, EmbeddedBrowserUnavailable>
  readonly activateTab: (tabId: BrowserTabId) => Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly closeTab: (tabId: BrowserTabId) => Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly navigate: (input: string) => Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly goBack: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly goForward: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly reload: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly stop: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly continueInsecureNavigation: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly cancelInsecureNavigation: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly setViewport: (bounds: BrowserViewportRect | null) => Effect.Effect<void>
  readonly openExternal: Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly respondToPermission: (
    requestId: BrowserPermissionRequestId,
    allow: boolean,
  ) => Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly cancelDownload: (
    downloadId: BrowserDownloadId,
  ) => Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly revealDownload: (
    downloadId: BrowserDownloadId,
  ) => Effect.Effect<void, EmbeddedBrowserUnavailable>
  readonly shutdown: Effect.Effect<void>
}

export class EmbeddedBrowser extends Context.Tag("desktop/EmbeddedBrowser")<
  EmbeddedBrowser,
  EmbeddedBrowserService
>() {}

const makeTab = (): InternalTab => ({
  id: BrowserTabIdSchema.make(randomUUID()),
  view: null,
  title: "New tab",
  url: "",
  pendingUrl: null,
  faviconUrl: null,
  phase: "blank",
  error: null,
  insecureUrl: null,
  allowedInsecureOrigin: null,
  discardTimer: null,
})

const permissionOrigin = (
  details: { readonly requestingUrl?: string; readonly securityOrigin?: string },
  contents: WebContents,
): string => {
  const requestingUrl = details.requestingUrl ?? details.securityOrigin ?? contents.getURL()
  try {
    return new URL(requestingUrl).origin
  } catch {
    return "Unknown origin"
  }
}

const normalizedOrigin = (value: string): string => {
  try {
    return new URL(value).origin
  } catch {
    return value
  }
}

const permissionKey = (tabId: BrowserTabId, origin: string, permission: string): string =>
  `${tabId}\0${origin}\0${permission}`

const makeEmbeddedBrowser = (window: BrowserWindow) =>
  Effect.gen(function* () {
    const runtime = yield* Effect.runtime<never>()
    const tabs = new Map<BrowserTabId, InternalTab>()
    const tabOrder: BrowserTabId[] = []
    const permissions = new Map<BrowserPermissionRequestId, InternalPermissionRequest>()
    const permissionOrder: BrowserPermissionRequestId[] = []
    const pendingPermissionIds = new Map<string, BrowserPermissionRequestId>()
    const permissionGrants = new Set<string>()
    const downloads = new Map<BrowserDownloadId, InternalDownload>()
    let revision = 0
    let focusLocationRevision = 0
    let activeTabId: BrowserTabId | null = null
    let viewport: BrowserViewportRect | null = null
    let disposed = false

    const tabState = (tab: InternalTab): BrowserTabState => {
      const contents = tab.view?.webContents
      const history = contents !== undefined && !contents.isDestroyed()
        ? contents.navigationHistory
        : null
      return {
        id: tab.id,
        title: tab.title,
        url: tab.url,
        pendingUrl: tab.pendingUrl,
        faviconUrl: tab.faviconUrl,
        phase: tab.phase,
        canGoBack: history?.canGoBack() ?? false,
        canGoForward: history?.canGoForward() ?? false,
        error: tab.error,
        insecureUrl: tab.insecureUrl,
      }
    }

    const downloadState = (download: InternalDownload): BrowserDownloadState => ({
      id: download.id,
      tabId: download.tabId,
      fileName: download.item.getFilename(),
      savePath: download.item.getSavePath() || null,
      receivedBytes: download.item.getReceivedBytes(),
      totalBytes: download.item.getTotalBytes(),
      status: download.status,
    })

    const snapshot = (): BrowserWorkspaceState => ({
      revision,
      focusLocationRevision,
      activeTabId: Option.fromNullable(activeTabId),
      tabs: tabOrder.flatMap((id) => {
        const tab = tabs.get(id)
        return tab === undefined ? [] : [tabState(tab)]
      }),
      permissionRequest: permissionOrder.length === 0
        ? null
        : permissions.get(permissionOrder[0]!)?.state ?? null,
      downloads: [...downloads.values()].map(downloadState),
    })
    const state = yield* SubscriptionRef.make(snapshot())

    const publish = (): void => {
      if (disposed) return
      revision += 1
      Runtime.runSync(runtime)(SubscriptionRef.set(state, snapshot()))
    }

    const activeTab = (): InternalTab => {
      if (activeTabId === null) {
        throw new EmbeddedBrowserUnavailable({ reason: "There is no active browser tab." })
      }
      const tab = tabs.get(activeTabId)
      if (tab === undefined) {
        throw new EmbeddedBrowserUnavailable({ reason: "The active browser tab is unavailable." })
      }
      return tab
    }

    const hideView = (tab: InternalTab): void => {
      tab.view?.setVisible(false)
    }

    const clearDiscardTimer = (tab: InternalTab): void => {
      if (tab.discardTimer === null) return
      clearTimeout(tab.discardTimer)
      tab.discardTimer = null
    }

    const closeView = (tab: InternalTab, discarded: boolean): void => {
      clearDiscardTimer(tab)
      const view = tab.view
      tab.view = null
      if (view !== null) {
        view.setVisible(false)
        window.contentView.removeChildView(view)
        if (!view.webContents.isDestroyed()) view.webContents.close({ waitForBeforeUnload: false })
      }
      if (discarded && tab.url.length > 0 && tab.phase !== "crashed") {
        tab.phase = "discarded"
        tab.pendingUrl = null
      }
    }

    const scheduleDiscard = (tab: InternalTab): void => {
      if (tab.view === null || tab.discardTimer !== null) return
      tab.discardTimer = setTimeout(() => {
        tab.discardTimer = null
        if (disposed || tab.id === activeTabId && viewport !== null) return
        closeView(tab, true)
        publish()
      }, HIDDEN_DISCARD_DELAY_MS)
    }

    const syncVisibility = (): void => {
      const blockedByPrompt = permissionOrder.length > 0
      const currentViewport = viewport
      for (const tab of tabs.values()) {
        const visible = tab.id === activeTabId
          && currentViewport !== null
          && !blockedByPrompt
          && (tab.phase === "loading" || tab.phase === "ready")
        if (visible) {
          clearDiscardTimer(tab)
          if (tab.view !== null) {
            tab.view.setBounds(currentViewport)
            if (!tab.view.getVisible()) window.contentView.addChildView(tab.view)
            tab.view.setVisible(true)
          }
        } else {
          hideView(tab)
          scheduleDiscard(tab)
        }
      }
    }

    const settlePermissionRequest = (
      requestId: BrowserPermissionRequestId,
      allowed: boolean,
    ): boolean => {
      const request = permissions.get(requestId)
      if (request === undefined) return false
      permissions.delete(requestId)
      if (pendingPermissionIds.get(request.key) === requestId) {
        pendingPermissionIds.delete(request.key)
      }
      const index = permissionOrder.indexOf(requestId)
      if (index >= 0) permissionOrder.splice(index, 1)
      clearTimeout(request.timer)
      if (allowed) permissionGrants.add(request.key)
      for (const callback of request.callbacks) callback(allowed)
      return true
    }

    const clearTabPermissions = (tabId: BrowserTabId): void => {
      for (const key of [...permissionGrants]) {
        if (key.startsWith(`${tabId}\0`)) permissionGrants.delete(key)
      }
      for (const requestId of [...permissionOrder]) {
        const request = permissions.get(requestId)
        if (request?.state.tabId !== tabId) continue
        settlePermissionRequest(requestId, false)
      }
    }

    const failTab = (tab: InternalTab, message: string): void => {
      tab.phase = "failed"
      tab.pendingUrl = null
      tab.error = message
      tab.insecureUrl = null
      hideView(tab)
      publish()
    }

    const checkPageNavigation = (tab: InternalTab, url: string): boolean => {
      const target = resolveBrowserNavigation(url)
      if (target._tag === "invalid") {
        failTab(tab, target.reason)
        return false
      }
      if (
        target._tag === "insecure"
        && tab.allowedInsecureOrigin !== new URL(target.url).origin
      ) {
        tab.phase = "insecure"
        tab.pendingUrl = null
        tab.insecureUrl = target.url
        tab.error = null
        hideView(tab)
        publish()
        return false
      }
      return true
    }

    const configureView = (tab: InternalTab, view: WebContentsView): void => {
      const contents = view.webContents
      contents.setWindowOpenHandler(({ url }) => {
        if (isAllowedBrowserNavigation(url)) {
          Runtime.runFork(runtime)(service.createTab(url))
        }
        return { action: "deny" }
      })
      contents.on("will-navigate", (event) => {
        if (!checkPageNavigation(tab, event.url)) event.preventDefault()
      })
      contents.on("will-redirect", (event) => {
        if (!checkPageNavigation(tab, event.url)) event.preventDefault()
      })
      contents.on("did-start-navigation", (event) => {
        if (!event.isMainFrame || event.isSameDocument) return
        tab.phase = "loading"
        tab.pendingUrl = event.url
        tab.error = null
        tab.insecureUrl = null
        clearTabPermissions(tab.id)
        publish()
      })
      const commitDocumentNavigation = (url: string): void => {
        tab.url = url
        tab.pendingUrl = null
        tab.error = null
        tab.insecureUrl = null
        tab.allowedInsecureOrigin = null
        tab.title = contents.getTitle().trim() || new URL(url).hostname || url
        syncVisibility()
        publish()
      }
      contents.on("did-navigate", (_event, url) => {
        // A prior load may finish after the user has entered an invalid or
        // insecure destination. Only a navigation that is still represented
        // by the tab's loading phase may commit over an interstitial.
        if (tab.phase === "loading") commitDocumentNavigation(url)
      })
      contents.on("did-navigate-in-page", (_event, url, isMainFrame) => {
        if (!isMainFrame || (tab.phase !== "loading" && tab.phase !== "ready")) return
        // The old document may emit a delayed history/hash update while a new
        // document is still pending. It must not replace that newer request.
        if (tab.phase === "loading" && tab.pendingUrl !== null) return
        tab.url = url
        tab.title = contents.getTitle().trim() || new URL(url).hostname || url
        publish()
      })
      contents.on("did-fail-load", (_event, errorCode, _errorDescription, _url, isMainFrame) => {
        if (!isMainFrame || errorCode === -3) return
        failTab(tab, browserNavigationFailureMessage(errorCode))
      })
      contents.on("did-stop-loading", () => {
        if (tab.phase !== "loading") return
        const currentUrl = contents.getURL()
        if (isAllowedBrowserNavigation(currentUrl)) tab.url = currentUrl
        tab.pendingUrl = null
        tab.phase = tab.url.length > 0 ? "ready" : "blank"
        tab.title = contents.getTitle().trim() || tab.title
        syncVisibility()
        publish()
      })
      contents.on("page-title-updated", (_event, title) => {
        tab.title = title.trim() || tab.title
        publish()
      })
      contents.on("page-favicon-updated", (_event, favicons) => {
        tab.faviconUrl = favicons.find((url) => url.startsWith("https:") || url.startsWith("data:")) ?? null
        publish()
      })
      contents.on("render-process-gone", (_event, details) => {
        tab.error = details.reason === "oom" ? "This page ran out of memory." : "This page crashed."
        tab.phase = "crashed"
        tab.pendingUrl = null
        closeView(tab, false)
        publish()
      })
      contents.on("before-input-event", (event, input) => {
        const shortcut = embeddedBrowserShortcut(input, process.platform)
        if (shortcut === null) return
        event.preventDefault()
        if ((shortcut === "next-tab" || shortcut === "previous-tab") && tabOrder.length > 0) {
          if (activeTabId === null) return
          const currentIndex = tabOrder.indexOf(activeTabId)
          const direction = shortcut === "previous-tab" ? -1 : 1
          const nextIndex = (currentIndex + direction + tabOrder.length) % tabOrder.length
          Runtime.runFork(runtime)(service.activateTab(tabOrder[nextIndex]!))
        } else if (shortcut === "focus-location") {
          focusLocationRevision += 1
          publish()
        } else if (shortcut === "new-tab") {
          Runtime.runFork(runtime)(service.createTab(null))
        } else if (shortcut === "close-tab") {
          Runtime.runFork(runtime)(service.closeTab(tab.id))
        } else if (shortcut === "reload") {
          Runtime.runFork(runtime)(service.reload)
        } else if (shortcut === "go-back") {
          Runtime.runFork(runtime)(service.goBack)
        } else if (shortcut === "go-forward") {
          Runtime.runFork(runtime)(service.goForward)
        } else if (shortcut === "stop") {
          if (contents.isLoadingMainFrame()) contents.stop()
          focusLocationRevision += 1
          publish()
        }
      })
    }

    const ensureView = (tab: InternalTab): WebContentsView => {
      if (tab.view !== null && !tab.view.webContents.isDestroyed()) return tab.view
      const view = new WebContentsView({
        webPreferences: {
          partition: BROWSER_PARTITION,
          nodeIntegration: false,
          nodeIntegrationInSubFrames: false,
          contextIsolation: true,
          sandbox: true,
          webSecurity: true,
          allowRunningInsecureContent: false,
          experimentalFeatures: false,
          backgroundThrottling: true,
          devTools: !appIsPackaged(),
        },
      })
      view.setVisible(false)
      if (viewport !== null) view.setBounds(viewport)
      window.contentView.addChildView(view)
      tab.view = view
      configureView(tab, view)
      return view
    }

    const loadUrl = (tab: InternalTab, url: string): void => {
      const view = ensureView(tab)
      tab.phase = "loading"
      tab.pendingUrl = url
      tab.error = null
      tab.insecureUrl = null
      syncVisibility()
      publish()
      void view.webContents.loadURL(url).catch((_cause: unknown) => {
        if (
          tab.view !== view
          || view.webContents.isDestroyed()
          || tab.phase !== "loading"
          || tab.pendingUrl !== url
        ) return
        failTab(tab, "This page could not be loaded.")
      })
    }

    const createTabInternal = (url: string | null): InternalTab => {
      const previous = activeTabId === null ? undefined : tabs.get(activeTabId)
      if (previous !== undefined) hideView(previous)
      const tab = makeTab()
      tabs.set(tab.id, tab)
      tabOrder.push(tab.id)
      activeTabId = tab.id
      if (url === null) focusLocationRevision += 1
      syncVisibility()
      publish()
      if (url !== null) {
        const target = resolveBrowserNavigation(url)
        if (target._tag === "navigate") loadUrl(tab, target.url)
        else if (target._tag === "insecure") {
          tab.phase = "insecure"
          tab.insecureUrl = target.url
          publish()
        } else failTab(tab, target.reason)
      }
      return tab
    }

    const disposeTab = (tab: InternalTab): void => {
      clearTabPermissions(tab.id)
      closeView(tab, false)
    }

    const browserSession = session.fromPartition(BROWSER_PARTITION, { cache: true })
    const findTabByContents = (contents: WebContents | null): InternalTab | undefined => {
      if (contents === null) return undefined
      return [...tabs.values()].find((tab) => tab.view?.webContents.id === contents.id)
    }
    browserSession.setPermissionCheckHandler((contents, permission, origin, details) => {
      if (AUTOMATIC_PERMISSIONS.has(permission)) return true
      const tab = findTabByContents(contents)
      const requestingOrigin = normalizedOrigin(origin)
      const grantedForTab = (candidate: InternalTab): boolean =>
        permissionGrants.has(permissionKey(candidate.id, requestingOrigin, permission))
      if (tab !== undefined) return grantedForTab(tab)

      // Electron supplies null WebContents for notifications and cross-origin
      // subframes. Narrow cross-origin checks to the tab whose top-level origin
      // embeds the requester; otherwise the requesting origin is the strongest
      // identity Electron exposes for the check.
      const embeddingOrigin = details.embeddingOrigin === undefined
        ? null
        : normalizedOrigin(details.embeddingOrigin)
      return [...tabs.values()].some((candidate) =>
        grantedForTab(candidate)
        && (embeddingOrigin === null || normalizedOrigin(candidate.url) === embeddingOrigin),
      )
    })
    browserSession.setPermissionRequestHandler((contents, permission, callback, details) => {
      if (AUTOMATIC_PERMISSIONS.has(permission)) {
        callback(true)
        return
      }
      const tab = findTabByContents(contents)
      if (tab === undefined || !PROMPTABLE_PERMISSIONS.has(permission)) {
        callback(false)
        return
      }
      const origin = permissionOrigin(details, contents)
      const key = permissionKey(tab.id, origin, permission)
      if (permissionGrants.has(key)) {
        callback(true)
        return
      }
      const pendingId = pendingPermissionIds.get(key)
      const pending = pendingId === undefined ? undefined : permissions.get(pendingId)
      if (pending !== undefined) {
        pending.callbacks.push(callback)
        return
      }
      if (pendingId !== undefined) pendingPermissionIds.delete(key)
      const id = BrowserPermissionRequestIdSchema.make(randomUUID())
      const timer = setTimeout(() => {
        if (!settlePermissionRequest(id, false)) return
        syncVisibility()
        publish()
      }, PERMISSION_REQUEST_TIMEOUT_MS)
      permissions.set(id, {
        state: { id, tabId: tab.id, origin, permission },
        key,
        callbacks: [callback],
        timer,
      })
      pendingPermissionIds.set(key, id)
      permissionOrder.push(id)
      syncVisibility()
      publish()
    })

    const onWillDownload = (_event: Electron.Event, item: DownloadItem, contents: WebContents): void => {
      const tab = findTabByContents(contents)
      if (tab === undefined) {
        item.cancel()
        return
      }
      const id = BrowserDownloadIdSchema.make(randomUUID())
      const download: InternalDownload = { id, tabId: tab.id, item, status: "progressing" }
      downloads.set(id, download)
      item.setSaveDialogOptions({ title: "Save download", defaultPath: item.getFilename() })
      item.on("updated", (_downloadEvent, status) => {
        download.status = status === "interrupted" ? "failed" : "progressing"
        publish()
      })
      item.once("done", (_downloadEvent, status) => {
        download.status = status === "interrupted" ? "failed" : status
        const completed = [...downloads.values()].filter((entry) => entry.status !== "progressing")
        for (const stale of completed.slice(0, -20)) downloads.delete(stale.id)
        publish()
      })
      publish()
    }
    browserSession.on("will-download", onWillDownload)

    const unavailable = (reason: string) => new EmbeddedBrowserUnavailable({ reason })
    const effect = <A>(operation: () => A): Effect.Effect<A, EmbeddedBrowserUnavailable> =>
      Effect.try({ try: operation, catch: (cause) => unavailable(cause instanceof Error ? cause.message : String(cause)) })

    const shutdown = Effect.sync(() => {
      if (disposed) return
      disposed = true
      browserSession.setPermissionCheckHandler(null)
      browserSession.setPermissionRequestHandler(null)
      browserSession.removeListener("will-download", onWillDownload)
      for (const download of downloads.values()) {
        if (download.status === "progressing") download.item.cancel()
      }
      downloads.clear()
      for (const requestId of [...permissionOrder]) settlePermissionRequest(requestId, false)
      pendingPermissionIds.clear()
      for (const tab of tabs.values()) disposeTab(tab)
      tabs.clear()
      tabOrder.splice(0)
    })

    const service: EmbeddedBrowserService = {
      changes: state.changes,
      createTab: (url) => effect(() => createTabInternal(url).id),
      activateTab: (tabId) => effect(() => {
        const next = tabs.get(tabId)
        if (next === undefined) throw unavailable("This browser tab no longer exists.")
        if (tabId === activeTabId) return
        if (activeTabId !== null) hideView(activeTab())
        activeTabId = tabId
        clearDiscardTimer(next)
        if (next.phase === "blank") focusLocationRevision += 1
        if (next.phase === "discarded" && next.url.length > 0) loadUrl(next, next.url)
        else {
          syncVisibility()
          publish()
        }
      }),
      closeTab: (tabId) => effect(() => {
        const index = tabOrder.indexOf(tabId)
        const tab = tabs.get(tabId)
        if (index < 0 || tab === undefined) throw unavailable("This browser tab no longer exists.")
        const wasActive = tabId === activeTabId
        disposeTab(tab)
        tabs.delete(tabId)
        tabOrder.splice(index, 1)
        if (tabOrder.length === 0) {
          activeTabId = null
        } else if (wasActive) {
          activeTabId = tabOrder[Math.min(index, tabOrder.length - 1)]!
          if (tabs.get(activeTabId)?.phase === "blank") focusLocationRevision += 1
        }
        syncVisibility()
        publish()
      }),
      navigate: (input) => effect(() => {
        const tab = activeTab()
        tab.view?.webContents.stop()
        const target = resolveBrowserNavigation(input)
        if (target._tag === "invalid") {
          failTab(tab, target.reason)
        } else if (target._tag === "insecure") {
          tab.phase = "insecure"
          tab.pendingUrl = null
          tab.insecureUrl = target.url
          tab.error = null
          hideView(tab)
          publish()
        } else loadUrl(tab, target.url)
      }),
      goBack: effect(() => {
        const contents = activeTab().view?.webContents
        if (contents !== undefined && contents.navigationHistory.canGoBack()) contents.navigationHistory.goBack()
      }),
      goForward: effect(() => {
        const contents = activeTab().view?.webContents
        if (contents !== undefined && contents.navigationHistory.canGoForward()) contents.navigationHistory.goForward()
      }),
      reload: effect(() => {
        const tab = activeTab()
        if ((tab.phase === "crashed" || tab.phase === "discarded" || tab.phase === "failed") && tab.url.length > 0) {
          closeView(tab, false)
          loadUrl(tab, tab.url)
        } else tab.view?.webContents.reload()
      }),
      stop: effect(() => { activeTab().view?.webContents.stop() }),
      continueInsecureNavigation: effect(() => {
        const tab = activeTab()
        if (tab.insecureUrl === null) throw unavailable("There is no insecure navigation to continue.")
        tab.allowedInsecureOrigin = new URL(tab.insecureUrl).origin
        loadUrl(tab, tab.insecureUrl)
      }),
      cancelInsecureNavigation: effect(() => {
        const tab = activeTab()
        if (tab.phase !== "insecure") return
        tab.insecureUrl = null
        tab.pendingUrl = null
        tab.error = null
        tab.phase = tab.url.length > 0 ? "ready" : "blank"
        syncVisibility()
        publish()
      }),
      setViewport: (bounds) => Effect.sync(() => {
        viewport = bounds === null ? null : {
          x: Math.max(0, Math.round(bounds.x)),
          y: Math.max(0, Math.round(bounds.y)),
          width: Math.max(0, Math.round(bounds.width)),
          height: Math.max(0, Math.round(bounds.height)),
        }
        syncVisibility()
      }),
      openExternal: effect(() => {
        const url = activeTab().url
        if (!isAllowedBrowserNavigation(url)) throw unavailable("There is no safe page to open externally.")
        void shell.openExternal(url)
      }),
      respondToPermission: (requestId, allow) => effect(() => {
        if (!settlePermissionRequest(requestId, allow)) {
          throw unavailable("This permission request is no longer active.")
        }
        syncVisibility()
        publish()
      }),
      cancelDownload: (downloadId) => effect(() => {
        const download = downloads.get(downloadId)
        if (download === undefined || download.status !== "progressing") {
          throw unavailable("This download is no longer active.")
        }
        download.item.cancel()
      }),
      revealDownload: (downloadId) => effect(() => {
        const download = downloads.get(downloadId)
        const path = download?.item.getSavePath() ?? ""
        if (download?.status !== "completed" || path.length === 0) {
          throw unavailable("This download is not available on disk.")
        }
        shell.showItemInFolder(path)
      }),
      shutdown,
    }

    yield* Effect.addFinalizer(() => shutdown)

    return service
  })

const appIsPackaged = (): boolean => app.isPackaged

export const makeEmbeddedBrowserLive = (
  window: BrowserWindow,
): Layer.Layer<EmbeddedBrowser> => Layer.scoped(EmbeddedBrowser, makeEmbeddedBrowser(window))
