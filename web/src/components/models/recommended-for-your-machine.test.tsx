import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { offeredFromRecommendation } from "./data"
import { RecommendedForYourMachine, useRecommendedSelection } from "./recommended-for-your-machine"
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

function Shown({ measuring }: { measuring: RecommendationsView["measuring"] }) {
  const selection = useRecommendedSelection(recommendations)
  return <RecommendedForYourMachine recommendations={recommendations} placeholder={null} measuring={measuring}
    usOnly={false} setUsOnly={() => {}} owned={new Set()} selection={selection} />
}

const STATUS = /<span role="status"[^>]*>.*?<\/span><\/span>/

describe("Recommended for your machine", () => {
  it("says quants are still being measured in its title's row, leaving the cards exactly as they are without it", () => {
    const measuring = renderToStaticMarkup(<Shown measuring={{ settled: 29, total: 36 }} />)
    const settled = renderToStaticMarkup(<Shown measuring={null} />)
    expect(measuring).toContain("Measuring 7 more quants")
    expect(measuring).toContain('title="Still measuring how 7 more quants run on this computer')
    expect(measuring.indexOf("Measuring 7 more quants")).toBeLessThan(measuring.indexOf("Only models from US developers"))
    expect(measuring.indexOf("Measuring 7 more quants")).toBeLessThan(measuring.indexOf("Qwen3.8 Flash Next"))
    expect(measuring.replace(STATUS, "")).toBe(settled)
  })
})
