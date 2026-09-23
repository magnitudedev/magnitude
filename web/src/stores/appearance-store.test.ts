import { afterAll, describe, expect, it, vi } from "vitest"

describe("appearance preference", () => {
  it("defaults to the operating system, follows live changes, and persists overrides", async () => {
    const attributes = new Map<string, string>()
    const style = new Map<string, string>()
    let dark = true
    const changeListeners: Array<() => void> = []
    const storage = new Map<string, string>()
    vi.stubGlobal("document", {
      documentElement: {
        dataset: new Proxy(
          {},
          {
            set: (_target, key, value) => {
              attributes.set(String(key), String(value))
              return true
            },
          }
        ),
        style: {
          colorScheme: "",
          setProperty: (key: string, value: string) => style.set(key, value),
        },
      },
    })
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    })
    vi.stubGlobal("matchMedia", () => ({
      get matches() {
        return dark
      },
      addEventListener: (_event: string, listener: () => void) => {
        changeListeners.push(listener)
      },
    }))

    const appearance = await import("./appearance-store")
    const browser = await import("./browser-appearance")
    browser.initializeBrowserAppearance()
    expect(appearance.getAppearancePreference()).toBe("system")
    expect(appearance.getResolvedAppearance()).toBe("dark")
    expect(attributes.get("theme")).toBe("dark")

    dark = false
    changeListeners[0]?.()
    expect(appearance.getResolvedAppearance()).toBe("light")
    expect(style.get("--magnitude-slate-900")).toBeDefined()
    expect(style.get("--magnitude-blue-700")).toBeDefined()

    browser.setBrowserAppearancePreference("dark")
    expect(storage.get("magnitude.appearance")).toBe("dark")
    dark = false
    changeListeners[0]?.()
    expect(appearance.getResolvedAppearance()).toBe("dark")

    browser.setBrowserAppearancePreference("system")
    expect(storage.has("magnitude.appearance")).toBe(false)
    expect(appearance.getResolvedAppearance()).toBe("light")

    // Applying host-owned state must not read or write browser persistence.
    appearance.setAppearancePreference("dark")
    expect(storage.has("magnitude.appearance")).toBe(false)
    expect(appearance.getResolvedAppearance()).toBe("dark")
  })
})

afterAll(() => vi.unstubAllGlobals())
