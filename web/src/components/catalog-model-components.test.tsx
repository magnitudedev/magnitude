import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { CatalogComponent, CatalogLocalModel } from "@magnitudedev/sdk"
import { CatalogComponentDetails, CatalogDrafterOffer } from "./catalog-model-components"
import { catalogModel } from "../../test/models/fixtures"

const projector = { component: "Projector", downloadBytes: 1_034_945_421 } as const
const drafter = { component: "Drafter", downloadBytes: 59_200_000 } as const
const lacking = (...missingComponents: CatalogLocalModel["missingComponents"]): CatalogLocalModel => ({ ...catalogModel(), missingComponents })

/** The first element whose text is `label`, clicked through React's own props. */
const click = (element: React.ReactElement | null, label: string): boolean => {
  if (element === null || typeof element !== "object") return false
  const props = element.props as { children?: unknown; onClick?: () => void }
  const text = [props.children].flat(Infinity).filter((child) => typeof child === "string").join("")
  if (text === label && props.onClick) { props.onClick(); return true }
  return [props.children].flat(Infinity).some((child) => typeof child === "object" && child !== null && click(child as React.ReactElement, label))
}

describe("catalog model components", () => {
  it("offers a missing projector in Details as image input, downloading only the projector", () => {
    const requested: CatalogComponent[] = []
    const details = CatalogComponentDetails({ model: lacking(projector), disabled: false, onDownload: (component) => requested.push(component) })
    const html = renderToStaticMarkup(details)
    expect(html).toContain(">Image input<")
    expect(html).toContain(">Download (1.03 GB)<")
    expect(click(details, "Download (1.03 GB)")).toBe(true)
    expect(requested).toEqual(["Projector"])
  })

  it("offers a missing catalog drafter on the card and in Details", () => {
    const requested: CatalogComponent[] = []
    const model = lacking(projector, drafter)
    expect(renderToStaticMarkup(<CatalogDrafterOffer model={model} disabled={false} onDownload={() => {}} />)).toContain("Faster with a drafter · Add (59.2 MB)")
    const details = CatalogComponentDetails({ model, disabled: false, onDownload: (component) => requested.push(component) })
    expect(renderToStaticMarkup(details)).toContain(">Drafter<")
    expect(click(details, "Download (59.2 MB)")).toBe(true)
    expect(requested).toEqual(["Drafter"])
  })

  it("always shows the Drafter row, naming the drafter a fully downloaded model serves with", () => {
    const base = catalogModel()
    const serving = base.servingState
    if (serving._tag !== "Assessed") throw new Error("fixture is assessed")
    const drafted: CatalogLocalModel = { ...base, servingState: { ...serving, speculativeMethod: Option.some({ _tag: "Mtp" }) } }
    const html = renderToStaticMarkup(CatalogComponentDetails({ model: drafted, disabled: false, onDownload: () => {} }))
    expect(html).toContain(">Drafter<")
    expect(html).toContain(">Magnitude MTP for Qwen3.6 35B-A3B<")
    expect(html).not.toContain("Download")
  })

  it("says None for a catalog model with no drafter, and Not assessed before its assessment", () => {
    const none = renderToStaticMarkup(CatalogComponentDetails({ model: catalogModel(), disabled: false, onDownload: () => {} }))
    expect(none).toContain(">Drafter<")
    expect(none).toContain(">None<")
    const assessing = renderToStaticMarkup(CatalogComponentDetails({ model: { ...catalogModel(), servingState: { _tag: "Assessing", profile: { contextLength: 32_768 } } }, disabled: false, onDownload: () => {} }))
    expect(assessing).toContain(">Not assessed<")
  })

  it("offers no component download while the model downloads, optimizes, or is removed", () => {
    expect(renderToStaticMarkup(<CatalogDrafterOffer model={catalogModel()} disabled={false} onDownload={() => {}} />)).toBe("")
    const installed = catalogModel().acquisitionState
    const updating = { ...lacking(projector, drafter), acquisitionState: { ...installed, _tag: "Optimizing", progress: { stage: "preparing", completed: 0, total: 0 } } } as unknown as CatalogLocalModel
    const html = renderToStaticMarkup(CatalogComponentDetails({ model: updating, disabled: false, onDownload: () => {} }))
    expect(html).toContain(">Drafter<")
    expect(html).not.toContain("Download")
    expect(html).not.toContain("Image input")
  })

  it("says in Details when a catalog model runs without its drafter because it fits only without it", () => {
    const html = renderToStaticMarkup(CatalogComponentDetails({ model: { ...catalogModel(), drafterWithheld: true }, disabled: false, onDownload: () => {} }))
    expect(html).toContain(">Drafter<")
    expect(html).toContain("Off - this model fits this computer only without it")
  })
})
