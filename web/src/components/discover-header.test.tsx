import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { DiscoverHeader } from "./discover-header"
import { selectModels } from "./models-page"
import { modelLabs } from "./model-logo"
import { added, catalogModel, customModel, doesNotFit } from "../../test/models/fixtures"

const fit = catalogModel()
if (fit.servingState._tag !== "Assessed") throw new Error("Expected an assessed fixture")
const large = { ...fit, servingState: { ...fit.servingState, assessment: doesNotFit } }
const custom = customModel(added, { presentation: { ...customModel(added).presentation, displayName: "Gemma 4", variantOf: Option.none() } })

describe("Discover filter counts", () => {
  it.each([
    { filter: "all", search: "", lab: null, count: 3 },
    { filter: "fits", search: "", lab: null, count: 2 },
    { filter: "fits", search: "", lab: modelLabs.find(lab => lab.name === "Qwen")!, count: 1 },
    { filter: "fits", search: "gemma", lab: null, count: 1 },
    { filter: "fits", search: "no such model", lab: null, count: 0 },
  ])("uses the same result count in the tab and search for $filter / $search", ({ filter, search, lab, count }) => {
    const { visible } = selectModels({ library: [fit, large, custom], hidden: [], filter, lab, search, sort: "recommended" })
    expect(visible).toHaveLength(count)
    const html = renderToStaticMarkup(<DiscoverHeader view="all" count={visible.length} search={search} onSearch={() => {}} onView={() => {}} hardware={null} action={<button>Add from Hugging Face</button>} />)
    expect(html).toContain(`placeholder="Search ${count} models…"`)
    expect(html).toMatch(new RegExp(`All models<span[^>]*>${count}</span>`))
    expect(html.match(/role="tab"/g)).toHaveLength(2)
    expect(html).not.toContain('role="status"')
    expect(html).not.toContain("matching")
    expect(html).not.toContain("Catalog")
  })
})

it("does not display a zero count before the catalog is observed", () => {
  const html = renderToStaticMarkup(<DiscoverHeader view="recommended" count={null} search="" onSearch={() => {}} onView={() => {}} hardware={null} action={null} />)
  expect(html).toContain('placeholder="Search models…"')
  expect(html).not.toContain("Search 0 models")
  expect(html).toMatch(/All models<span[^>]*>…<\/span>/)
})

it("places the found-models notice below hardware and above both Discover views", () => {
  const html = renderToStaticMarkup(<DiscoverHeader view="recommended" count={1} search="" onSearch={() => {}} onView={() => {}} hardware={<p>Your machine</p>} notice={<p>Found models</p>} action={null} />)
  expect(html.indexOf("Your machine")).toBeLessThan(html.indexOf("Found models"))
  expect(html.indexOf("Found models")).toBeLessThan(html.indexOf('role="tablist"'))
})
