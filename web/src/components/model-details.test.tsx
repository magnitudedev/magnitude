import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { HttpsUrlSchema, type CatalogLocalModel } from "@magnitudedev/sdk"
import { CatalogSources, ModelDetails } from "./model-details"
import { catalogModel } from "../../test/models/fixtures"

const catalog = (): CatalogLocalModel => {
  const model = catalogModel()
  return { ...model, presentation: { ...model.presentation, sourceUrls: [HttpsUrlSchema.make("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF")] } }
}

describe("model details", () => {
  it("lists a recommendation's sources, then License and Context window", () => {
    const html = renderToStaticMarkup(<ModelDetails model={catalog()} source={<CatalogSources model={catalog()} />} />)
    expect(html).toContain("Hugging Face · unsloth")
    expect(html.indexOf("Hugging Face · unsloth")).toBeLessThan(html.indexOf(">License<"))
    expect(html.indexOf(">License<")).toBeLessThan(html.indexOf(">Context window<"))
  })
})
