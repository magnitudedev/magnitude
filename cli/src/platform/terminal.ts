/**
 * Terminal and OS adapter used by the interactive CLI.
 *
 * Uses Bun APIs for process spawning, clipboard (OSC 52), and terminal size.
 * Stubs for unsupported capabilities (storage, notifications, dialogs).
 */
import type {
  Platform,
  Storage,
  Clipboard,
  Notification,
  Dialogs,
  TerminalCapabilities,
} from "@magnitudedev/client-common"

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

export const makeTerminalAdapter = (): Platform => ({
  id: "terminal",
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
})
