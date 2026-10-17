import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
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
  const text = [props.children].flat().filter((child) => typeof child === "string").join("")
  if (text === label && props.onClick) { props.onClick(); return true }
  return [props.children].flat().some((child) => typeof child === "object" && child !== null && click(child as React.ReactElement, label))
}

describe("catalog model components", () => {
  it("offers a missing projector in Details as image input, downloading only the projector", () => {
    const requested: CatalogComponent[] = []
    const details = CatalogComponentDetails({ model: lacking(projector), disabled: false, onDownload: (component) => requested.push(component) })
    const html = renderToStaticMarkup(details!)
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
    expect(renderToStaticMarkup(details!)).toContain(">Drafter<")
    expect(click(details, "Download (59.2 MB)")).toBe(true)
    expect(requested).toEqual(["Drafter"])
  })

  it("shows nothing for a model with every component, or while it downloads", () => {
    expect(CatalogComponentDetails({ model: catalogModel(), disabled: false, onDownload: () => {} })).toBeNull()
    expect(renderToStaticMarkup(<CatalogDrafterOffer model={catalogModel()} disabled={false} onDownload={() => {}} />)).toBe("")
    const installed = catalogModel().acquisitionState
    const updating = { ...lacking(projector), acquisitionState: { ...installed, _tag: "Optimizing", progress: { stage: "preparing", completed: 0, total: 0 } } } as unknown as CatalogLocalModel
    expect(CatalogComponentDetails({ model: updating, disabled: false, onDownload: () => {} })).toBeNull()
  })
})
