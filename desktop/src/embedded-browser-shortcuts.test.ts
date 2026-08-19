import { describe, expect, it } from "vitest"
import { embeddedBrowserShortcut } from "./embedded-browser-shortcuts"

const input = (key: string, overrides: Partial<Parameters<typeof embeddedBrowserShortcut>[0]> = {}) => ({
  type: "keyDown",
  key,
  control: false,
  meta: false,
  shift: false,
  ...overrides,
})

describe("embeddedBrowserShortcut", () => {
  it("uses Command for browser commands on macOS", () => {
    expect(embeddedBrowserShortcut(input("t", { meta: true }), "darwin")).toBe("new-tab")
    expect(embeddedBrowserShortcut(input("l", { control: true }), "darwin")).toBeNull()
  })

  it("uses Control for browser commands on Linux and Windows", () => {
    for (const platform of ["linux", "win32"] as const) {
      expect(embeddedBrowserShortcut(input("W", { control: true }), platform)).toBe("close-tab")
      expect(embeddedBrowserShortcut(input("r", { control: true }), platform)).toBe("reload")
    }
  })

  it("supports history, location, and tab traversal commands", () => {
    expect(embeddedBrowserShortcut(input("[", { meta: true }), "darwin")).toBe("go-back")
    expect(embeddedBrowserShortcut(input("]", { meta: true }), "darwin")).toBe("go-forward")
    expect(embeddedBrowserShortcut(input("l", { meta: true }), "darwin")).toBe("focus-location")
    expect(embeddedBrowserShortcut(input("Tab", { control: true }), "darwin")).toBe("next-tab")
    expect(embeddedBrowserShortcut(input("Tab", { control: true, shift: true }), "linux")).toBe("previous-tab")
  })

  it("maps Escape to stop without a modifier", () => {
    expect(embeddedBrowserShortcut(input("Escape"), "darwin")).toBe("stop")
  })

  it("ignores key-up and unrelated input", () => {
    expect(embeddedBrowserShortcut(input("t", { meta: true, type: "keyUp" }), "darwin")).toBeNull()
    expect(embeddedBrowserShortcut(input("x", { meta: true }), "darwin")).toBeNull()
  })
})
