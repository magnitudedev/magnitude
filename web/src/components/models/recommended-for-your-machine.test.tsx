import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { hardwareBenchmarkView, type HardwareBenchmarkView } from "./benchmark"
import { offeredFromRecommendation } from "./data"
import { PreparationProgress, preparationLabels } from "./onboarding"
import { RecommendedForYourMachine, selectedModel, useRecommendedSelection } from "./recommended-for-your-machine"
import type { Recommendation } from "./recommendations"

const GB = 1e9
const ladder = [4, 6].map(bits => ({ ...makeSetupModel(false), modelId: CatalogFormModelIdSchema.make(`flash:gguf:q${bits}`), storageBytes: bits * 20 * GB }) as LocalModel)
const recommendation = {
  base: "flash", name: "Qwen3.8 Flash Next", developer: { lab: "alibaba", name: "Alibaba", country: "CN" },
  chosen: Option.some("flash:gguf:q4"), rungs: ladder.map(model => model.modelId),
} as unknown as RecommendedModel
const recommendations: readonly Recommendation[] = [
  { model: offeredFromRecommendation(recommendation, new Map(ladder.map(model => [model.modelId, model])), "Device", null)!, note: "Smartest that runs well here" },
]

function Shown({ benchmark, placeholder = null }: { benchmark: HardwareBenchmarkView | null; placeholder?: React.ReactNode }) {
  const selection = useRecommendedSelection(recommendations)
  return <RecommendedForYourMachine recommendations={recommendations} placeholder={placeholder} benchmark={benchmark}
    usOnly={false} setUsOnly={() => {}} owned={new Set()} selection={selection} />
}

/** Whether any button or link sits inside a button or an element acting as one. */
const nestsControls = (html: string) => /role="button"/.test(html) || /<button\b(?:(?!<\/button>).)*<(?:button|a)\b/s.test(html)

const STATUS = /<div role="status".*?<\/div>/

describe("Recommended for your machine", () => {
  const activeBenchmark = hardwareBenchmarkView(
    { _tag: "Measuring", progress: Option.some({ completed: 38, total: 113, phase: "TokenGeneration" }) },
    false,
  )!

  it("shows benchmark progress below the heading row without crowding it, leaving everything else as it is", () => {
    const measuring = renderToStaticMarkup(<Shown benchmark={activeBenchmark} />)
    const settled = renderToStaticMarkup(<Shown benchmark={null} />)
    expect(measuring).toContain("Benchmarking your hardware · Testing token generation speed · 34%")
    expect(measuring.indexOf("Only models from US developers")).toBeLessThan(measuring.indexOf("Benchmarking your hardware"))
    expect(measuring.replace(STATUS, "")).toBe(settled)
  })

  it("integrates standardized benchmark progress into the loading placeholder when recommendations are still settling", () => {
    const benchmark = hardwareBenchmarkView(
      { _tag: "Measuring", progress: Option.some({ completed: 16, total: 100, phase: "MemoryAndStorage" }) },
      false,
    )!
    const html = renderToStaticMarkup(<Shown benchmark={benchmark} placeholder={<div>skeleton</div>} />)
    expect(html).toContain("Benchmarking your hardware")
    expect(html).toContain("Testing memory and SSD speed")
    expect(html).toContain("16%")
    expect(html).toContain("width:16%")
    expect(html).toContain("opacity-45")
  })

  it("chooses the quant already on this computer over the recommended one", () => {
    const chosen = (owned: ReadonlySet<string>) => {
      let quant: string | undefined
      function Probe() { quant = useRecommendedSelection(recommendations, owned).pickedQuant?.key; return null }
      renderToStaticMarkup(<Probe />)
      return quant
    }
    expect(chosen(new Set())).toBe("flash:gguf:q4")
    expect(chosen(new Set(["flash:gguf:q6"]))).toBe("flash:gguf:q6")
  })

  it("maps every benchmark phase and edge state consistently", () => {
    expect(hardwareBenchmarkView({ _tag: "Measuring", progress: Option.none() }, false)?.subtitle).toBe("Testing baseline compute")
    expect(hardwareBenchmarkView({ _tag: "Measuring", progress: Option.some({ completed: 10, total: 100, phase: "Baseline" }) }, false)?.subtitle).toBe("Testing baseline compute")
    expect(hardwareBenchmarkView({ _tag: "Measuring", progress: Option.some({ completed: 15, total: 100, phase: "MemoryAndStorage" }) }, false)?.subtitle).toBe("Testing memory and SSD speed")
    expect(hardwareBenchmarkView({ _tag: "Measuring", progress: Option.some({ completed: 50, total: 100, phase: "TokenGeneration" }) }, false)?.subtitle).toBe("Testing token generation speed")
    expect(hardwareBenchmarkView({ _tag: "Measuring", progress: Option.some({ completed: 80, total: 100, phase: "PromptProcessing" }) }, false)?.subtitle).toBe("Testing prompt processing speed")
    expect(hardwareBenchmarkView({ _tag: "Measuring", progress: Option.some({ completed: 95, total: 100, phase: "LongContextAttention" }) }, false)?.subtitle).toBe("Testing long-context attention")
    expect(hardwareBenchmarkView({ _tag: "Measured" }, false)).toMatchObject({ subtitle: "Ranking models for your machine", percent: 100, fraction: 1 })
    expect(hardwareBenchmarkView({ _tag: "Unmeasured" }, false, true)).toMatchObject({ subtitle: "Waiting for your GPU to be free", percent: null, fraction: null })
    expect(hardwareBenchmarkView({ _tag: "Unmeasured" }, false, false)).toMatchObject({ subtitle: "Waiting for your machine to be ready", percent: null, fraction: null })
    expect(hardwareBenchmarkView({ _tag: "Failed", retryAt: Date.now() + 60_000 }, false)?.failed).toBe(true)
    expect(hardwareBenchmarkView({ _tag: "Measured" }, true)).toBeNull()
  })

  it("opens a card from its own button, with the repository link beside it rather than inside it", () => {
    const html = renderToStaticMarkup(<Shown benchmark={null} />)
    expect(html).toContain('aria-expanded="true"')
    expect(nestsControls(html)).toBe(false)
  })

  it("renders the same standardized benchmark copy in onboarding step 4", () => {
    const benchmark = hardwareBenchmarkView(
      { _tag: "Measuring", progress: Option.some({ completed: 16, total: 100, phase: "MemoryAndStorage" }) },
      false,
    )!
    const html = renderToStaticMarkup(<PreparationProgress steps={[
      { label: preparationLabels[0], detail: null, fraction: null, state: "done" },
      { label: preparationLabels[1], detail: null, fraction: null, state: "done" },
      { label: preparationLabels[2], detail: "0 models found", fraction: null, state: "done" },
      { label: preparationLabels[3], subtitle: benchmark.subtitle, detail: `${benchmark.percent}%`, fraction: benchmark.fraction, state: "active" },
    ]} />)
    expect(html).toContain("Benchmarking your hardware")
    expect(html).toContain("Testing memory and SSD speed")
    expect(html).toContain("16%")
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
