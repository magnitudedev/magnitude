import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { ExploreView } from "./explore-view"

const render = (query: string, results: Parameters<typeof ExploreView>[0]["results"]) => renderToStaticMarkup(
  <ExploreView query={query} setQuery={() => {}} results={results} families={[]} owned={new Set()}
    download={() => null} useOpened={model => model} onSeen={() => {}} />
)

describe("Explore models", () => {
  it("is a search: before one, it says Hugging Face results show here, under the Hugging Face logo", () => {
    const html = render("", { _tag: "Idle" })
    expect(html).toContain("Search Hugging Face, or paste a link")
    expect(html).toContain("Results from Hugging Face show here")
    expect(html).toMatch(/<img[^>]*aria-hidden="true"/)
    expect(html).not.toContain("Best for your machine")
  })

  it("replaces the empty state with the results once searching", () => {
    const html = render("qwen", { _tag: "Searching" })
    expect(html).toContain("Searching Hugging Face…")
    expect(html).not.toContain("Results from Hugging Face show here")
  })
})
