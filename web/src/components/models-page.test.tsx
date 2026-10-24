import { renderToStaticMarkup } from "react-dom/server"
import type { ReactElement } from "react"
import { describe, expect, it, vi } from "vitest"
import { Option } from "effect"
import type { ModelId } from "@magnitudedev/sdk"
import { DiscoverSections, HiddenModelUnhide, ModelsListing, afterModelAdded, otherVersionsRequest, selectModels } from "./models-page"
import { ModelsHeader } from "./models-header"
import { modelLabs } from "./model-logo"
import { added, catalogModel, customModel, doesNotFit, foundIn } from "../../test/models/fixtures"

const commands = vi.hoisted(() => ({ pending: new Set<string>(), unhide: vi.fn() }))

vi.mock("@magnitudedev/client-common", async importOriginal => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useLocalModelCommandStatus: (modelId: ModelId) => ({ pending: commands.pending.has(modelId), pendingOperations: [], failures: [] }),
  useLocalModelMutations: () => ({ unhide: commands.unhide }),
}))

const qwen = catalogModel()
const gemma = customModel(foundIn({ _tag: "LmStudio" }), { presentation: { ...customModel(added).presentation, displayName: "Gemma 4 31B it", variantOf: Option.none() } })
const tooLarge = customModel(added, { servingState: { ...customModel(added).servingState, assessment: doesNotFit } as ReturnType<typeof customModel>["servingState"] })
const qwenLab = modelLabs.find(lab => lab.name === "Qwen")!
const header = (count: number) => renderToStaticMarkup(<ModelsHeader title="Catalog" action={null} selectedCount={count} matchingCount={0} search="" onSearch={() => {}} controls={null} />)

describe("catalog search placeholder", () => {
  const select = (overrides: Partial<Parameters<typeof selectModels>[0]>) =>
    selectModels({ library: [qwen, gemma, tooLarge], hidden: [], filter: "all", lab: null, search: "", sort: "name", ...overrides })
  it("counts what the filter and lab leave, not the typed search", () => {
    const typed = select({ search: "gemma" })
    expect(typed.selected).toHaveLength(3)
    expect(typed.visible.map(model => model.modelId)).toEqual([gemma.modelId])
    expect(header(typed.selected.length)).toContain('placeholder="Search 3 models…"')
    expect(select({ filter: "fits" }).selected).toHaveLength(2)
    expect(header(select({ filter: "fits", search: "nothing matches" }).selected.length)).toContain('placeholder="Search 2 models…"')
    expect(select({ lab: qwenLab }).selected.map(model => model.modelId)).toEqual([qwen.modelId, tooLarge.modelId])
    expect(header(select({ filter: "fits", lab: qwenLab, search: "gemma" }).selected.length)).toContain('placeholder="Search 1 model…"')
  })
  it("lists hidden models under the Hidden filter", () => {
    const hidden = select({ filter: "hidden", hidden: [gemma] })
    expect(hidden.selected.map(model => model.modelId)).toEqual([gemma.modelId])
  })
})

describe("hidden filter", () => {
  const card = (model: Parameters<typeof otherVersionsRequest>[1]) => <article key={model.modelId}>card {model.modelId}</article>
  it("routes to the hidden list, with Unhide disabled while that model's command is pending", () => {
    commands.pending.clear()
    commands.pending.add(gemma.modelId)
    const other = customModel(foundIn({ _tag: "Ollama" }), { modelId: customModel(added).modelId })
    const html = renderToStaticMarkup(<ModelsListing filter="hidden" models={[gemma, other, qwen]} card={card} empty={<p>empty</p>} footer={<p>footer</p>} />)
    expect(html).toContain("Hidden from Magnitude")
    expect(html).not.toContain("card ")
    expect(html).not.toContain("footer")
    expect(html.match(/<button[^>]*>.*?Unhide<\/button>/g)).toHaveLength(2)
    expect(html.match(/<button[^>]*disabled=""[^>]*>.*?Unhide<\/button>/g)).toHaveLength(1)
    expect(html.indexOf("disabled=\"\"")).toBeLessThan(html.indexOf("Ollama"))
  })
  it("unhides that model", () => {
    commands.pending.clear()
    commands.unhide.mockClear()
    const button = HiddenModelUnhide({ model: gemma }) as ReactElement<{ pending: boolean; onUnhide: () => void }>
    expect(button.props.pending).toBe(false)
    button.props.onUnhide()
    expect(commands.unhide).toHaveBeenCalledWith(gemma.modelId)
  })
  it("shows cards, the empty state, and the footer for every other filter", () => {
    const html = renderToStaticMarkup(<ModelsListing filter="all" models={[qwen]} card={card} empty={<p>empty</p>} footer={<p>footer</p>} />)
    expect(html).toContain(`card ${qwen.modelId}`)
    expect(html).not.toContain("empty")
    expect(html).toContain("footer")
    expect(renderToStaticMarkup(<ModelsListing filter="downloaded" models={[]} card={card} empty={<p>empty</p>} />)).toContain("empty")
  })
})

describe("find other versions", () => {
  it("is offered only on Catalog page catalog cards and searches for the model's name", () => {
    expect(otherVersionsRequest("catalog", qwen)).toEqual(Option.some({ _tag: "Search", query: "Qwen3.6 35B-A3B" }))
    expect(otherVersionsRequest("models", qwen)).toEqual(Option.none())
    expect(otherVersionsRequest("discover", qwen)).toEqual(Option.some({ _tag: "Search", query: "Qwen3.6 35B-A3B" }))
    expect(otherVersionsRequest("catalog", gemma)).toEqual(Option.none())
  })
})

describe("adding a model", () => {
  it("closes the dialog, then shows My Models", () => {
    const calls: string[] = []
    afterModelAdded(() => calls.push("close"), page => calls.push(page))()
    expect(calls).toEqual(["close", "models"])
  })
})

describe("discover", () => {
  const discover = (state: Parameters<typeof DiscoverSections>[0]["state"]) => renderToStaticMarkup(<DiscoverSections
    preference={<p>preference</p>} recommendations={<p>recommendations</p>} state={state} onAdd={() => {}} onExplore={() => {}} />)
  it("keeps the preference above the recommendations without a duplicate Add action or model-details view", () => {
    const html = discover("ranked")
    expect(html.indexOf("preference")).toBeLessThan(html.indexOf("recommendations"))
    expect(html).not.toContain("Add from Hugging Face")
    expect(html).not.toContain("Details")
    expect(html).not.toContain("No fitting recommendations")
  })
  it("offers All models and Add when nothing fits, and does not mistake missing hardware for no fit", () => {
    expect(discover("pending")).not.toContain("No fitting recommendations")
    expect(discover("unavailable")).toContain("hardware recommendations are unavailable")
    expect(discover("unavailable")).not.toContain("No fitting recommendations")
    expect(discover("failed")).toContain("Couldn’t assess model recommendations")
    expect(discover("failed")).not.toContain("No fitting recommendations")
    const none = discover("none")
    expect(none).toContain("No fitting recommendations right now.")
    expect(none).toContain("Browse all models")
    expect(none).toContain("Add from Hugging Face")
  })
})
