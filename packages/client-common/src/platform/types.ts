/**
 * Platform abstraction — spec §5.3
 *
 * Abstracts OS-specific operations that differ between desktop (Electron),
 * browser, and terminal (CLI). Each app provides its own implementation;
 * components consume the Platform via `usePlatform` without knowing which
 * environment they are in.
 */

import type { Layer } from "effect"
import { RpcClient } from "@effect/rpc"
import type { MenuAction } from "../types/menu-action"

export type { MenuAction }

export interface Storage {
  getItem(key: string): Promise<string | null>
  setItem(key: string, value: string): Promise<void>
  removeItem(key: string): Promise<void>
}

export interface Clipboard {
  readText(): Promise<string>
  writeText(text: string): Promise<void>
}

export interface Notification {
  show(title: string, body: string): void
}

export interface Dialogs {
  openDirectory(): Promise<string | null>
  openFile(options?: { multiple?: boolean }): Promise<string[] | null>
}

/**
 * Terminal-specific capabilities (CLI only).
 * Web/desktop platforms leave `terminal` undefined.
 */
export interface TerminalCapabilities {
  readonly width: number
  readonly height: number
  /** process.platform — used for WindowsWarningScreen gate */
  readonly os?: string
  readonly onResize: (cb: () => void) => () => void
  readonly getPalette?: () => Promise<{ defaultBackground?: string } | null>
  readonly setTerminalTitle?: (title: string) => void
}

export interface Platform {
  readonly id: "web" | "desktop" | "terminal"
  readonly protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>
  readonly clipboard: Clipboard
  readonly storage: Storage
  readonly notifications: Notification
  readonly dialogs: Dialogs
  readonly openLink: (url: string) => Promise<void>
  readonly openPath: (path: string) => Promise<void>
  readonly showItemInFolder: (path: string) => void
  readonly fetch: typeof fetch
  readonly getDefaultServer: () => Promise<string | null>
  readonly setDefaultServer: (url: string) => Promise<void>
  readonly onMenuAction?: (cb: (action: MenuAction) => void) => () => void
  readonly quit?: () => void
  /** Terminal capabilities — present only on the CLI platform */
  readonly terminal?: TerminalCapabilities
}
