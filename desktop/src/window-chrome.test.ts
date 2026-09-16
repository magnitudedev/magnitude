import { expect, it } from "vitest"
import { windowChrome, windowControlColors } from "./window-chrome"

it("retains native Mac traffic lights without a separate title bar", () => {
  expect(windowChrome("darwin", true)).toEqual({ titleBarStyle: "hidden", trafficLightPosition: { x: 16, y: 14 } })
})
it("reserves a Windows caption overlay that follows the theme", () => {
  for (const dark of [false, true]) expect(windowChrome("win32", dark)).toEqual({
    titleBarStyle: "hidden", titleBarOverlay: { ...windowControlColors(dark), height: 32 }, autoHideMenuBar: true,
  })
  expect(windowControlColors(true)).not.toEqual(windowControlColors(false))
})
it("preserves Linux window-manager decorations", () => {
  expect(windowChrome("linux", false)).toEqual({})
  expect(windowChrome("linux", true)).toEqual({})
})
