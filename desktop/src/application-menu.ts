import type { MenuItemConstructorOptions } from "electron"
import type { Page } from "./desktop-rpc"

export const buildApplicationMenu = (platform: NodeJS.Platform, actions: {
  readonly open: (page: Page) => void
  readonly quit: () => void
}): MenuItemConstructorOptions[] => {
  const quit = { label: "Quit Magnitude", accelerator: "CmdOrCtrl+Q", click: actions.quit }
  return [
    platform === "darwin"
      ? { label: "Magnitude", submenu: [
        { label: "About Magnitude", role: "about" }, { type: "separator" },
        { label: "Settings…", accelerator: "CmdOrCtrl+,", click: () => actions.open("settings") },
        { type: "separator" }, quit,
      ] }
      : { label: "File", submenu: [quit] },
    { label: "Edit", submenu: [{ role: "undo" }, { role: "redo" }, { type: "separator" }, { role: "cut" }, { role: "copy" }, { role: "paste" }, { role: "selectAll" }] },
    { label: "View", submenu: [
      { label: "Discover Models", click: () => actions.open("discover") },
      { label: "Catalog", click: () => actions.open("catalog") },
      { label: "My Models", click: () => actions.open("models") },
      { label: "Connections", click: () => actions.open("connections") },
      { label: "Usage", click: () => actions.open("usage") },
      { label: "Status", click: () => actions.open("status") },
      { label: "Settings", click: () => actions.open("settings") },
    ] },
    { label: "Window", submenu: [{ role: "minimize" }, { role: "close" }] },
  ]
}
