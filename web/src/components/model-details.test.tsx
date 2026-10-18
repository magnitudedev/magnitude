import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { HttpsUrlSchema, type CatalogLocalModel, type CustomLocalModel } from "@magnitudedev/sdk"
import { CatalogSources, ModelDetails } from "./model-details"
import { CatalogComponentDetails } from "./catalog-model-components"
import { CustomModelDrafterRow, CustomModelSource } from "./custom-model-details"
import { catalogModel, customModel, foundIn } from "../../test/models/fixtures"

const catalog = (): CatalogLocalModel => {
  const model = catalogModel()
  return {
    ...model,
    presentation: { ...model.presentation, sourceUrls: [HttpsUrlSchema.make("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF")] },
    missingComponents: [{ component: "Drafter", downloadBytes: 59_200_000 }],
  }
}

const catalogDetails = (model: CatalogLocalModel) => renderToStaticMarkup(<ModelDetails model={model} radar open
  drafter={<CatalogComponentDetails model={model} disabled={false} onDownload={() => {}} />}
  source={<CatalogSources model={model} />} />)

const customDetails = (model: CustomLocalModel) => renderToStaticMarkup(<ModelDetails model={model} radar open
  drafter={<CustomModelDrafterRow model={model} disabled={false} onChangeDrafter={() => {}} onTurnOffDrafter={() => {}} onAddDrafter={() => {}} />}
  source={<CustomModelSource model={model} variantOfName={Option.none()} />} />)

/** Each label's position in the markup, so a test can assert the sections' order. */
const positions = (html: string, labels: readonly string[]) => labels.map(label => {
  const index = html.indexOf(label)
  expect(index, label).toBeGreaterThanOrEqual(0)
  return index
})
const ascending = (values: readonly number[]) => values.every((value, index) => index === 0 || values[index - 1]! < value)

const sections = [">Drafter<", "Source", ">License<", ">Context window<", "capability profile"]

describe("model details", () => {
  it("lays out a catalog model as Drafter, Sources, License and Context window, then the radar", () => {
    const html = catalogDetails(catalog())
    expect(ascending(positions(html, sections))).toBe(true)
    expect(html).toContain("Hugging Face · unsloth")
    expect(html).toContain(">1%<")
  })

  it("keeps the Drafter row first for a fully downloaded catalog model", () => {
    const html = catalogDetails({ ...catalogModel(), presentation: catalog().presentation })
    expect(ascending(positions(html, sections))).toBe(true)
    expect(html).toContain(">None<")
  })

  it("lays out a custom model the same way, with its source and the same radar", () => {
    const html = customDetails(customModel(foundIn({ _tag: "LmStudio" })))
    expect(ascending(positions(html, sections))).toBe(true)
    expect(html).toContain("LM Studio · /Users/me/.ollama/models/blobs/sha256-1")
    expect(html).toContain(">Speed (est.)")
    expect(html).toContain(">Fidelity<")
    expect(html).toContain(">Medium (Q4)<")
  })

  it("never rates a custom model's intelligence", () => {
    const html = customDetails(customModel(foundIn({ _tag: "Ollama" })))
    expect(html).toContain(">Intelligence<")
    expect(html).toContain(">Not rated<")
    expect(html).not.toMatch(/>\d+%</)
  })

  it("shows speed only on the radar, never as a separate estimate", () => {
    for (const html of [catalogDetails(catalog()), customDetails(customModel(foundIn({ _tag: "Ollama" })))]) {
      expect(html).not.toContain("Estimated speed on your machine")
      expect(html.match(/Speed \(est\.\)/g)).toHaveLength(1)
    }
  })
})
