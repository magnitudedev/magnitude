import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { renderToStaticMarkup } from "react-dom/server"
import { afterAll, describe, expect, it, vi } from "vitest"

const target = new EventTarget()
vi.stubGlobal("window", target)
vi.stubGlobal("document", Object.assign(new EventTarget(), { visibilityState: "visible" }))

const { NavigationDrawer } = await import("./app")

afterAll(() => {
  vi.unstubAllGlobals()
})

describe("NavigationDrawer", () => {
  it("renders a full-width drawer with Models, Connections, Chat, Docs and Settings", () => {
    const html = renderToStaticMarkup(<NavigationDrawer page="models" onClose={() => {}} />)
    expect(html).toContain('aria-label="Navigation"')
    expect(html).toContain("w-64")
    expect(html).toContain("max-w-[85vw]")
    expect(html).not.toContain("-translate-x-full")
    expect(html).toContain("Magnitude")
    expect(html).toContain('aria-label="Close navigation"')
    expect(html).toContain(">Models<")
    expect(html).toContain(">Connections<")
    expect(html).toContain(">Chat<")
    expect(html).toContain(">Docs<")
    expect(html).toContain('aria-label="Settings"')
  })

  it("defines slide-in-left keyframes once in tailwind.css", () => {
    const css = readFileSync(fileURLToPath(new URL("./styles/tailwind.css", import.meta.url)), "utf8")
    const matches = css.match(/@keyframes\s+slide-in-left\b/g) ?? []
    expect(matches).toHaveLength(1)
  })
})
