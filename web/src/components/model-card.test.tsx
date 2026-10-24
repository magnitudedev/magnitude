import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it, vi } from "vitest"
import type { CatalogLocalModel } from "@magnitudedev/sdk"
import { Option } from "effect"
import { ModelCard } from "./model-card"
import { catalogModel, doesNotFit } from "../../test/models/fixtures"

vi.mock("@magnitudedev/client-common", async importOriginal => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useLocalModelCommandStatus: () => ({ pending: false, pendingOperations: [], failures: [] }),
  useLocalModelStopStatus: () => ({ pending: false }),
  useLocalModelMutations: () => ({}),
}))

const model = { ...catalogModel(), acquisitionState: { _tag: "NotInstalled" as const } }
const render = (value: CatalogLocalModel = model, page: "catalog" | "models" = "catalog") => renderToStaticMarkup(<ModelCard model={value} models={[value]} page={page} hardware={Option.none()} />)

describe("unified catalog rows", () => {
  it("shows four inline metrics and a plain quant for an undownloaded model", () => {
    const html = render()
    expect(html.match(/<dt /g)).toHaveLength(4)
    expect(html).not.toContain("Not downloaded")
    expect(html).toContain("Memory")
    expect(html).toContain("Drafter")
    expect(html).toContain("Estimated speed")
    expect(html).toContain("Intelligence")
    expect(html).not.toContain("font-mono")
  })
  it("retains operational status in My Models", () => {
    const html = render(model, "models")
    expect(html).toContain("Not downloaded")
    expect(html.match(/<dt /g)).toHaveLength(5)
  })
  it("states non-fit clearly and disables acquisition without an extra panel", () => {
    if (model.servingState._tag !== "Assessed") throw new Error("Expected an assessed fixture")
    const html = render({ ...model, servingState: { ...model.servingState, assessment: doesNotFit } })
    expect(html).toContain("Doesn’t fit this machine")
    expect(html).toContain("more memory than this computer")
    expect(html).toMatch(/<button[^>]*disabled=""/)
    expect(html).not.toContain('role="alert"')
  })
  it("uses the same dash as unavailable speed when the drafter does not fit", () => {
    const html = render({ ...model, drafterWithheld: true })
    expect(html).not.toContain("Off · doesn’t fit")
    expect(html).toMatch(/aria-label="Drafter unavailable">—<\/span>/)
    const dash = html.match(/<span[^>]*aria-label="Drafter unavailable"[^>]*>/)?.[0]
    expect(dash).toBeDefined()
    expect(dash).not.toContain("title=")
  })

})
