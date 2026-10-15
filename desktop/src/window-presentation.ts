import { Effect } from "effect"

export interface PresentableWindow {
  isMinimized(): boolean
  restore(): void
  show(): void
  isVisible(): boolean
  focus(): void
}

/**
 * Restores, shows and focuses a window. Windows replaces a process's first normal ShowWindow with the
 * show command its launcher passed, so a hidden start (Node's `windowsHide`, which Magnitude 0.2.8
 * used to relaunch after an update) leaves the window hidden. That substitution happens once, so a
 * window still hidden after the first show is shown again.
 */
export const presentWindow = (platform: NodeJS.Platform, window: PresentableWindow) => Effect.sync(() => {
  if (window.isMinimized()) window.restore()
  window.show()
  if (platform === "win32" && !window.isVisible()) window.show()
  window.focus()
})
