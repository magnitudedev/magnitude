import type { BrowserWindowConstructorOptions } from "electron"
import { slate } from "@magnitudedev/client-common"

export const windowControlColors = (dark: boolean) => ({
  color: dark ? slate[925] : slate[50],
  symbolColor: dark ? slate[200] : slate[900],
})

export const windowChrome = (platform: NodeJS.Platform, dark: boolean): BrowserWindowConstructorOptions =>
  platform === "darwin"
    ? { titleBarStyle: "hidden", trafficLightPosition: { x: 16, y: 14 } }
    : platform === "win32"
      ? { titleBarStyle: "hidden", titleBarOverlay: { ...windowControlColors(dark), height: 32 }, autoHideMenuBar: true }
      : {}
