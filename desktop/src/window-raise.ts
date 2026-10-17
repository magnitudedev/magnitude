import type { App, BrowserWindow } from "electron"

export type RaisableWindow = Pick<BrowserWindow, "isMinimized" | "restore" | "show" | "focus" | "setAlwaysOnTop">

/**
 * Brings the window forward for a link, a second launch, or the tray. Windows refuses focus to a process
 * that isn't in the foreground and only flashes the taskbar button, so the window is briefly made topmost
 * while it takes focus. macOS activates the app even while another one is active.
 */
export const raiseWindow = (window: RaisableWindow, application: Pick<App, "focus">, platform: NodeJS.Platform): void => {
  if (window.isMinimized()) window.restore()
  if (platform === "win32") {
    window.setAlwaysOnTop(true)
    window.show()
    window.focus()
    window.setAlwaysOnTop(false)
    return
  }
  window.show()
  if (platform === "darwin") application.focus({ steal: true })
  window.focus()
}
