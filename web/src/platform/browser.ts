/**
 * Browser Platform implementation — spec §5.3
 *
 * Uses browser APIs: localStorage, navigator.clipboard, window.open, fetch.
 */
import { Effect, Layer } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { DaemonSpawnerTag, makeRemoteDaemonSpawner } from "@magnitudedev/sdk"
import type { Platform, Storage, Clipboard, Notification, Dialogs } from "@magnitudedev/client-common"

// Experimental File System Access API — only available in Chromium browsers.
// This is a client-host capability, not agent-host filesystem access.
interface FileSystemDirectoryHandle { readonly name: string }
interface FileSystemFileHandle { readonly name: string }

interface WindowWithFSAccess extends Window {
  showDirectoryPicker?(): Promise<FileSystemDirectoryHandle>
  showOpenFilePicker?(opts: { multiple?: boolean }): Promise<FileSystemFileHandle[]>
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
    if (typeof Notification !== "undefined" && Notification.permission === "granted") {
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
    const picker = (window as WindowWithFSAccess).showDirectoryPicker
    if (!picker) return null
    try {
      const handle = await picker.call(window)
      return handle.name
    } catch {
      return null
    }
  },
  async openFile(options?: { multiple?: boolean }): Promise<string[] | null> {
    const picker = (window as WindowWithFSAccess).showOpenFilePicker
    if (!picker) return null
    try {
      const handles = await picker.call(window, { multiple: options?.multiple ?? false })
      return handles.map((h) => h.name)
    } catch {
      return null
    }
  },
}

export function createBrowserPlatform(proxyUrl: string = ""): Platform {
  return {
    id: "web",
    daemonSpawnerLayer: Layer.effect(
      DaemonSpawnerTag,
      makeRemoteDaemonSpawner(proxyUrl).pipe(Effect.provide(FetchHttpClient.layer)),
    ),
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
