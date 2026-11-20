import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { offeredFromRecommendation } from "./data"
import { RecommendedForYourMachine, selectedModel, useRecommendedSelection } from "./recommended-for-your-machine"
import type { Recommendation, RecommendationsView } from "./recommendations"

const GB = 1e9
const ladder = [4, 6].map(bits => ({ ...makeSetupModel(false), modelId: CatalogFormModelIdSchema.make(`flash:gguf:q${bits}`), storageBytes: bits * 20 * GB }) as LocalModel)
const recommendation = {
  base: "flash", name: "Qwen3.8 Flash Next", developer: { lab: "alibaba", name: "Alibaba", country: "CN" },
  chosen: Option.some("flash:gguf:q4"), rungs: ladder.map(model => model.modelId),
} as unknown as RecommendedModel
const recommendations: readonly Recommendation[] = [
  { model: offeredFromRecommendation(recommendation, new Map(ladder.map(model => [model.modelId, model])), "Device", null)!, note: "Smartest that runs well here" },
]

function Shown({ measuring, placeholder = null }: { measuring: RecommendationsView["measuring"]; placeholder?: React.ReactNode }) {
  const selection = useRecommendedSelection(recommendations)
  return <RecommendedForYourMachine recommendations={recommendations} placeholder={placeholder} measuring={measuring}
    usOnly={false} setUsOnly={() => {}} owned={new Set()} selection={selection} />
}

/** Whether any button or link sits inside a button or an element acting as one. */
const nestsControls = (html: string) => /role="button"/.test(html) || /<button\b(?:(?!<\/button>).)*<(?:button|a)\b/s.test(html)

const STATUS = /<div role="status".*?<\/div>/

describe("Recommended for your machine", () => {
  it("shows measuring progress below the heading row without crowding it, leaving everything else as it is", () => {
    const measuring = renderToStaticMarkup(<Shown measuring={{ settled: 12, total: 36 }} />)
    const settled = renderToStaticMarkup(<Shown measuring={null} />)
    expect(measuring).toContain("Measuring speeds on this computer · 12 of 36")
    expect(measuring).toContain("width:33.33")
    expect(measuring.indexOf("Only models from US developers")).toBeLessThan(measuring.indexOf("Measuring speeds"))
    expect(measuring.replace(STATUS, "")).toBe(settled)
  })

  it("integrates measuring progress into the loading placeholder when recommendations are still settling", () => {
    const html = renderToStaticMarkup(<Shown measuring={{ settled: 8, total: 32 }} placeholder={<div>skeleton</div>} />)
    expect(html).toContain("Measuring speeds on this computer")
    expect(html).toContain("8 of 32")
    expect(html).toContain("width:25%")
    expect(html).toContain("opacity-45")
  })

  it("opens a card from its own button, with the repository link beside it rather than inside it", () => {
    const html = renderToStaticMarkup(<Shown measuring={null} />)
    expect(html).toContain('aria-expanded="true"')
    expect(nestsControls(html)).toBe(false)
  })
})

describe("the chosen recommendation", () => {
  const listed = (...keys: string[]) => keys.map(key => ({ model: { key }, note: undefined }) as unknown as Recommendation)

  it("stays chosen while it is offered, whatever is added above it or filtered out around it", () => {
    expect(selectedModel("lfm3", listed("qwen27", "lfm25", "lfm3"))).toBe("lfm3")
    expect(selectedModel("lfm3", listed("lfm25", "lfm3", "flash"))).toBe("lfm3")
  })

  it("opens the first until one is chosen, none once closed, and the first again once the chosen one isn't offered", () => {
    expect(selectedModel(undefined, listed("a", "b"))).toBe("a")
    expect(selectedModel(null, listed("a", "b"))).toBeNull()
    expect(selectedModel("gone", listed("a", "b"))).toBe("a")
  })

  it("opens the first while a filter leaves the chosen model out, and the chosen one again once it is back", () => {
    expect(selectedModel("qwen4", listed("lfm25", "lfm3"))).toBe("lfm25")
    expect(selectedModel("qwen4", listed("qwen4", "lfm25", "lfm3"))).toBe("qwen4")
  })
})
