import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { HostNoticeStack } from "./host-notices"

const zLayers = (text: string) => [...text.matchAll(/\bz-(?:(\d+)|\[(\d+)\])/g)].map(match => Number(match[1] ?? match[2]))

describe("host notices", () => {
  const notice = { title: "Magnitude can’t open this link", description: "The link has an unknown parameter: x" }
  it("layer above every dialog layer, so an invalid magnitude:// link notice isn't hidden under the Add dialog", () => {
    const html = renderToStaticMarkup(<HostNoticeStack notices={[notice]} onDismiss={() => {}} />)
    expect(html).toContain("Magnitude can’t open this link")
    expect(html).toContain("Dismiss")
    const stack = html.match(/<div data-host-notices=""[^>]*>/)?.[0] ?? ""
    expect(stack).toContain("fixed")
    const notices = Math.max(...zLayers(stack))
    for (const ui of ["dialog.tsx", "alert-dialog.tsx"]) {
      const layers = zLayers(readFileSync(fileURLToPath(new URL(`./ui/${ui}`, import.meta.url)), "utf8"))
      expect(layers.length, ui).toBeGreaterThan(0)
      expect(notices, ui).toBeGreaterThan(Math.max(...layers))
    }
  })
  it("renders nothing without notices", () => {
    expect(renderToStaticMarkup(<HostNoticeStack notices={[]} onDismiss={() => {}} />)).toBe("")
  })
})
