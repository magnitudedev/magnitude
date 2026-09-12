import { describe, expect, it, vi } from "vitest"
import { buildApplicationMenu } from "./application-menu"

describe("application menu", () => {
  it.each(["darwin", "linux", "win32"] as const)("keeps owner Quit distinct from window close on %s", platform => {
    const actions = { open: vi.fn(), quit: vi.fn() }
    const menu = buildApplicationMenu(platform, actions)
    const items = menu.flatMap(item => Array.isArray(item.submenu) ? item.submenu : [])
    const quit = items.filter(item => item.label === "Quit Magnitude")
    expect(quit).toHaveLength(1)
    expect(quit[0]?.accelerator).toBe("CmdOrCtrl+Q")
    expect(quit[0]?.click).toBe(actions.quit)
    if (!quit[0]?.click) throw new Error("Quit action missing")
    Reflect.apply(quit[0].click, undefined, [])
    expect(actions.quit).toHaveBeenCalledOnce()
    expect(actions.open).not.toHaveBeenCalled()
    const close = items.find(item => item.role === "close")
    expect(close).toBeDefined()
    expect(close?.click).toBeUndefined()
    expect(menu[0]?.label).toBe(platform === "darwin" ? "Magnitude" : "File")
  })
  it.each(["darwin", "linux", "win32"] as const)("routes all View destinations on %s without quitting", platform => {
    const actions = { open: vi.fn(), quit: vi.fn() }
    const view = buildApplicationMenu(platform, actions).find(item => item.label === "View")
    if (!Array.isArray(view?.submenu)) throw new Error("View menu missing")
    for (const item of view.submenu) {
      if (!item.click) throw new Error("Navigation action missing")
      Reflect.apply(item.click, undefined, [])
    }
    expect(actions.open.mock.calls).toEqual([["discover"], ["catalog"], ["models"], ["connections"], ["usage"], ["status"], ["settings"]])
    expect(actions.quit).not.toHaveBeenCalled()
  })
})
