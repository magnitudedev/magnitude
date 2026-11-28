import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { ExploreView, type RunnableFamily } from "./explore-view"
import type { OfferedModel } from "./expanding-model"
import type { QuantView } from "./view"

const render = (
  query: string,
  results: Parameters<typeof ExploreView>[0]["results"],
  families: readonly RunnableFamily[] = [],
) => renderToStaticMarkup(
  <ExploreView query={query} setQuery={() => {}} results={results} families={families} owned={new Set()}
    download={() => null} useOpened={model => model} />,
)

const makePick = (key: string, name: string, note?: string) => {
  const q4: QuantView = {
    key: `${key}:gguf:q4`,
    bits: 4,
    publisher: "unsloth",
    name: "UD-Q4_K_XL",
    repository: `unsloth/${name.replace(/\s+/g, "-")}-GGUF`,
    sizeBytes: 16e9,
    downloadBytes: 16e9,
    drafterBytes: 1e9,
    drafterTarget: null,
    fits: true,
    measuring: false,
    speed: 48,
    phrase: "Fits in memory",
    placement: [{ kind: "Unified", bytes: 16e9 }],
    servedContext: 131_072,
    quality: null,
    missingDiskBytes: 0,
  }
  const q6: QuantView = { ...q4, key: `${key}:gguf:q6`, bits: 6, name: "UD-Q6_K_XL", sizeBytes: 22e9, downloadBytes: 22e9 }
  const model: OfferedModel = {
    key,
    catalog: true,
    name,
    lab: undefined,
    country: "US",
    fineTuneOf: null,
    drafter: { method: "DFlash 2", sizeBytes: 1e9 },
    contextLimit: 131_072,
    quants: [q4, q6],
    recommended: q4.key,
    checking: false,
  }
  return { model, note }
}

describe("Explore models", () => {
  it("falls back to the Hugging Face idle card when there are no picks", () => {
    const html = render("", { _tag: "Idle" })
    expect(html).toContain("Search Hugging Face, or paste a link")
    expect(html).toContain("Results from Hugging Face show here")
    expect(html).toMatch(/<img[^>]*aria-hidden="true"/)
    expect(html).not.toContain("Recommended for your machine")
  })

  it("shows Magnitude Pick cards with a quiet Hugging Face search prompt underneath in the idle state", () => {
    const pick = makePick("gemma-4-12b", "Gemma 4 12B", "Smartest that runs well here")
    const html = render("", { _tag: "Idle", picks: [pick] })
    expect(html).toContain("Gemma 4 12B")
    expect(html).toContain("Magnitude Pick")
    expect(html).toContain("Smartest that runs well here")
    expect(html).toContain("Or search for anything else — Hugging Face results show here")
    expect(html.indexOf("Gemma 4 12B")).toBeLessThan(html.indexOf("Or search for anything else"))
    expect(html).not.toContain("Recommended for your machine")
    expect(html).not.toContain("Only models from US developers")
  })

  it("pins matching Magnitude Picks above Searching Hugging Face", () => {
    const pick = makePick("qwen3.8-27b", "Qwen3.8 27B")
    const html = render("qwen", { _tag: "Searching", picks: [pick] })
    expect(html).toContain("Qwen3.8 27B")
    expect(html).toContain("Magnitude Pick")
    expect(html).toContain("Searching Hugging Face…")
    expect(html.indexOf("Qwen3.8 27B")).toBeLessThan(html.indexOf("Searching Hugging Face…"))
  })

  it("pins matching Magnitude Picks first above Hugging Face results", () => {
    const pick = makePick("gemma-4-12b", "Gemma 4 12B")
    const hfModel: OfferedModel = {
      ...pick.model,
      key: "bartowski/gemma-4-12b-GGUF",
      catalog: false,
      name: "Gemma 4 12B GGUF",
      downloads: 42_000,
      uploader: "bartowski",
      quantCount: 12,
    }
    const html = render("gemma", {
      _tag: "Results",
      picks: [pick],
      rows: [{ _tag: "Offered", model: hfModel, prefetched: false }],
      more: null,
    })
    expect(html).toContain("Magnitude Pick")
    expect(html).toContain("Gemma 4 12B GGUF")
    expect(html.indexOf("Magnitude Pick")).toBeLessThan(html.indexOf("Gemma 4 12B GGUF"))
  })

  it("says no other models on Hugging Face matched when a Magnitude Pick matched but HF returned nothing", () => {
    const pick = makePick("gemma-4-12b", "Gemma 4 12B")
    const html = render("gemma 4 12b", { _tag: "Results", picks: [pick], rows: [], more: null })
    expect(html).toContain("Gemma 4 12B")
    expect(html).toContain("No other models on Hugging Face match “gemma 4 12b”.")
  })

  it("shows Checking models while the first batch of checks is settling", () => {
    const html = render("qwen", { _tag: "Results", rows: [], more: { _tag: "Checking" } })
    expect(html).toContain("Checking models…")
    expect(html).not.toContain("No models on Hugging Face")
  })

  it("offers a manual Check more results button when the first pages had no runnable models", () => {
    const html = render("phi", { _tag: "Results", rows: [], more: { _tag: "Manual", onMore: () => {} } })
    expect(html).toContain("No models in the most downloaded Hugging Face results")
    expect(html).toContain("Check more results")
  })
})
