/**
 * Browser Platform implementation — spec §5.3
 *
 * Uses browser APIs: localStorage, navigator.clipboard, window.open, fetch.
 */
import { Effect, Exit, Layer, Scope } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { MagnitudeClient, MagnitudeServiceStarter } from "@magnitudedev/sdk"
import { makeFirstPartyConnection, makeRemoteServiceStarter, type FirstPartyConnection } from "@magnitudedev/client-common"
import type {
  Platform,
  Storage,
  Clipboard,
  Notification,
  Dialogs,
} from "@magnitudedev/client-common"

interface FileSystemFileHandle {
  readonly name: string
}

interface WindowWithFSAccess extends Window {
  showOpenFilePicker?(opts: {
    multiple?: boolean
  }): Promise<FileSystemFileHandle[]>
}

const STORAGE_KEY_PREFIX = "magnitude:"
const DEFAULT_SERVER_KEY = `${STORAGE_KEY_PREFIX}default-server`

const browserStorage: Storage = {
  async getItem(key: string): Promise<string | null> {
    try {
      return localStorage.getItem(`${STORAGE_KEY_PREFIX}${key}`)
    } catch {
      return null
    }
  },
  async setItem(key: string, value: string): Promise<void> {
    try {
      localStorage.setItem(`${STORAGE_KEY_PREFIX}${key}`, value)
    } catch {
      // ignore quota errors
    }
  },
  async removeItem(key: string): Promise<void> {
    try {
      localStorage.removeItem(`${STORAGE_KEY_PREFIX}${key}`)
    } catch {
      // ignore
    }
  },
}

const browserClipboard: Clipboard = {
  async readText(): Promise<string> {
    try {
      return await navigator.clipboard.readText()
    } catch {
      return ""
    }
  },
  async writeText(text: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // fallback: execCommand
      const textarea = document.createElement("textarea")
      textarea.value = text
      textarea.style.position = "fixed"
      textarea.style.opacity = "0"
      document.body.appendChild(textarea)
      textarea.select()
      try {
        document.execCommand("copy")
      } finally {
        document.body.removeChild(textarea)
      }
    }
  },
}

const browserNotifications: Notification = {
  show(title: string, body: string): void {
    if (
      typeof Notification !== "undefined" &&
      Notification.permission === "granted"
    ) {
      try {
        new Notification(title, { body })
      } catch {
        // ignore
      }
    }
  },
}

const browserDialogs: Dialogs = {
  async openDirectory(): Promise<string | null> {
    // Browsers cannot obtain an absolute agent-host path from a directory
    // handle. Project forms accept an explicit agent-host path instead.
    return null
  },
  async openFile(options?: { multiple?: boolean }): Promise<string[] | null> {
    const picker = (window as WindowWithFSAccess).showOpenFilePicker
    if (!picker) return null
    try {
      const handles = await picker.call(window, {
        multiple: options?.multiple ?? false,
      })
      return handles.map((h) => h.name)
    } catch {
      return null
    }
  },
}

export async function createBrowserAcnConnection(
  proxyUrl: string = window.location.origin
): Promise<FirstPartyConnection> {
  const starter = await Effect.runPromise(
    makeRemoteServiceStarter(proxyUrl).pipe(
      Effect.provide(FetchHttpClient.layer)
    )
  )
  const acnScope = await Effect.runPromise(Scope.make())
  const connection = await Effect.runPromise(
    makeFirstPartyConnection(MagnitudeClient.layer({ origin: `${proxyUrl}/acn` }).pipe(
      Layer.provide([FetchHttpClient.layer, Layer.succeed(MagnitudeServiceStarter, starter)]),
    )).pipe(
      Effect.provideService(Scope.Scope, acnScope),
      Effect.provide(FetchHttpClient.layer)
    )
  )
  const close = await Effect.runPromise(Effect.cached(
    connection.close.pipe(Effect.ensuring(Scope.close(acnScope, Exit.void))),
  ))
  const scopedConnection: FirstPartyConnection = { ...connection, close }
  window.addEventListener("pagehide", (event) => {
    if (event.persisted) return
    Effect.runFork(scopedConnection.close)
  })
  return scopedConnection
}

export function createBrowserPlatform(): Platform {
  return {
    id: "web",
    clipboard: browserClipboard,
    storage: browserStorage,
    notifications: browserNotifications,
    dialogs: browserDialogs,
    async openLink(url: string): Promise<void> {
      window.open(url, "_blank", "noopener,noreferrer")
    },
    async openPath(_path: string): Promise<void> {
      // No-op in browser — cannot open local paths
    },
    showItemInFolder(_path: string): void {
      // No-op in browser
    },
    fetch: globalThis.fetch.bind(globalThis),
    async getDefaultServer(): Promise<string | null> {
      return browserStorage.getItem(DEFAULT_SERVER_KEY)
    },
    async setDefaultServer(url: string): Promise<void> {
      await browserStorage.setItem(DEFAULT_SERVER_KEY, url)
    },
  }
}
