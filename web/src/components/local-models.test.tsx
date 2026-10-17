import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { HiddenModels, SourceTag, UnhideButton, UnsupportedModels, unsupportedSummary } from "./local-model-lists"
import { ModelRemoveAction, keepsExternalCopy, removalConfirmation, removalKind } from "./model-remove-action"
import { FoundModelsBanner, HuggingFaceLink, NoRecommendations, foundModelsSummary } from "./discover-custom"
import { ModelsHeader, matchStatus, searchPlaceholder } from "./models-header"
import { labForName, modelLab, variantLab } from "./model-logo"
import { added, catalogId, catalogModel, customModel, doesNotFit, foundIn, imported, unsupportedLocal } from "../../test/models/fixtures"

describe("models header", () => {
  it("puts the add action beside the title and the count in the search placeholder", () => {
    const html = renderToStaticMarkup(<ModelsHeader title="Catalog" action={<button type="button">Add from Hugging Face</button>} selectedCount={39} matchingCount={4} search="qwen" onSearch={() => {}} controls={<span>selects</span>} />)
    expect(html).toContain("Catalog</h1><button")
    expect(html).toContain('placeholder="Search 39 models…"')
    expect(html.indexOf("Search models")).toBeLessThan(html.indexOf("selects"))
    expect(html).toMatch(/class="sr-only" role="status">4 models</)
    expect(html).not.toMatch(/>\s*39 models\s*</)
  })
  it("words one model and many", () => {
    expect(searchPlaceholder(1)).toBe("Search 1 model…")
    expect(matchStatus(0)).toBe("0 models")
  })
})

describe("model labs", () => {
  it("follows a custom model's catalog lineage, then its name", () => {
    expect(modelLab(catalogModel())?.name).toBe("Qwen")
    expect(modelLab(customModel(added))?.name).toBe("Qwen")
    expect(modelLab(customModel(added, { presentation: { ...customModel(added).presentation, variantOf: Option.none(), displayName: "Gemma 4 31B it abliterated" } }))?.name).toBe("Gemma")
    expect(variantLab(Option.some(catalogId), "Anything")?.name).toBe("Qwen")
    expect(labForName("Llama 3.1 8B Instruct")?.name).toBe("Meta")
    expect(labForName("gpt-oss 20B")).toBeUndefined()
  })
})

describe("removing by origin", () => {
  const render = (model: Parameters<typeof removalKind>[0]) => renderToStaticMarkup(<ModelRemoveAction model={model} running={false} disabled={false} onRemove={() => {}} onHide={() => {}} />)
  it("deletes catalog and added files, forgets imported files, and hides found models", () => {
    expect(removalKind(catalogModel())).toBe("delete")
    expect(removalKind(customModel(added))).toBe("delete")
    expect(removalKind(customModel(imported))).toBe("forget")
    expect(removalKind(customModel(foundIn({ _tag: "LmStudio" })))).toBe("hide")
    expect(render(customModel(added))).toContain('title="Remove download"')
    expect(render(customModel(imported))).toContain('aria-label="Remove Qwen3.6 35B A3B Uncensored (Q4_K_P) from Magnitude"')
    expect(render(customModel(foundIn({ _tag: "Ollama" })))).toContain('aria-label="Hide Qwen3.6 35B A3B Uncensored (Q4_K_P) from Magnitude"')
    expect(render(customModel(added))).not.toContain("Hide")
    expect(render(customModel(imported))).not.toContain("Hide")
  })
  it("never offers deleting a catalog model whose files are only in another app's location", () => {
    const external = catalogModel()
    const found = { ...external, acquisitionState: { _tag: "Installed" as const, installation: { _tag: "Resolved" as const, installedBytes: 24_000_000_000, primaryPath: "/Users/me/.cache/huggingface/hub/qwen.gguf", ownership: "External" as const }, residencyState: { _tag: "Unloaded" as const } } }
    expect(removalKind(found)).toBe("hide")
    expect(render(found)).toContain('aria-label="Hide Qwen3.6 35B-A3B (Q4) from Magnitude"')
    expect(render(found)).not.toContain("Remove download")
    expect(renderToStaticMarkup(<ModelRemoveAction model={found} running={false} disabled={false} onRemove={() => {}} />)).toBe("")
    expect(removalKind({ ...found, acquisitionState: { ...found.acquisitionState, installation: { ...found.acquisitionState.installation, ownership: "Mixed" as const } } })).toBe("delete")
  })
  it("removes only the parts Magnitude stores for a found catalog model, and still offers Hide", () => {
    const external = catalogModel()
    const found = { ...external, storedComponents: ["Projector", "Drafter"] as const, acquisitionState: { _tag: "Installed" as const, installation: { _tag: "Resolved" as const, installedBytes: 1, primaryPath: "/Users/me/.cache/huggingface/hub/qwen.gguf", ownership: "External" as const }, residencyState: { _tag: "Unloaded" as const } } }
    expect(removalKind(found)).toBe("parts")
    const html = render(found)
    expect(html).toContain('aria-label="Hide Qwen3.6 35B-A3B (Q4) from Magnitude"')
    expect(html).toContain('aria-label="Remove the files Magnitude downloaded for Qwen3.6 35B-A3B (Q4)"')
    expect(removalConfirmation(found, false)).toEqual({
      title: "Remove the image input and drafter Magnitude downloaded for Qwen3.6 35B-A3B (Q4)?",
      description: "The model file is in another app's location, which Magnitude never deletes, so the model stays in My Models and runs from there. Hide it to stop listing it.",
      confirmLabel: "Remove",
    })
  })
  it("says removing an added model with the same bytes elsewhere deletes only Magnitude's copy and keeps it listed", () => {
    const addedModel = customModel(added)
    const alsoFound = customModel(added, { acquisitionState: { _tag: "Installed", installation: { _tag: "Resolved", installedBytes: 23_000_000_000, primaryPath: "/models/qwen.gguf", ownership: "Mixed" }, residencyState: { _tag: "Unloaded" } } })
    expect(keepsExternalCopy(alsoFound)).toBe(true)
    expect(keepsExternalCopy(addedModel)).toBe(false)
    expect(keepsExternalCopy(customModel(foundIn({ _tag: "LmStudio" })))).toBe(false)
    expect(removalConfirmation(alsoFound, false)).toEqual({
      title: "Remove Magnitude’s copy of Qwen3.6 35B A3B Uncensored (Q4_K_P)?",
      description: "Only the files Magnitude downloaded are deleted. The same model is also in another location on this computer, so it stays in My Models from there.",
      confirmLabel: "Remove",
    })
    expect(removalConfirmation(alsoFound, true).title).toBe("Stop Qwen3.6 35B A3B Uncensored (Q4_K_P) and remove Magnitude’s copy?")
    expect(removalConfirmation(addedModel, false)).toEqual({
      title: "Remove the downloaded files for Qwen3.6 35B A3B Uncensored (Q4_K_P)?",
      description: "You can download it again later.",
      confirmLabel: "Remove",
    })
    expect(removalConfirmation(addedModel, false).description).not.toContain("another location")
  })
  it("tells an imported file how to add it again, without doubled punctuation", () => {
    const confirmation = removalConfirmation(customModel(imported), false)
    expect(confirmation.description).toBe("The file stays where it is. You can add it again with Add GGUF file…")
    expect(confirmation.description).not.toMatch(/…\.$/)
    expect(confirmation.confirmLabel).toBe("Remove from Magnitude")
    expect(removalConfirmation(customModel(imported), true).confirmLabel).toBe("Stop and remove")
  })
  it("offers no hide action without a hide handler", () => {
    expect(renderToStaticMarkup(<ModelRemoveAction model={customModel(foundIn({ _tag: "Ollama" }))} running={false} disabled={false} onRemove={() => {}} />)).toBe("")
  })
})

describe("models that can't run", () => {
  const models = [
    unsupportedLocal("DeepSeek V4 Flash (UD-Q4_K_XL)", foundIn({ _tag: "LmStudio" }), { _tag: "Architecture", architecture: "deepseek4" }),
    unsupportedLocal("Qwen3.6 35B A3B (IQ1_M)", foundIn({ _tag: "HuggingFaceCache" }), { _tag: "WeightTypes", types: [{ ggmlType: "IQ1_M", tensorCount: 300 }] }),
    unsupportedLocal("gpt-oss 20B", foundIn({ _tag: "Ollama" }), { _tag: "Unknown", kind: "Sliding", detail: "PRIVATE" }),
  ]
  it("collapses into one line that expands to dimmed rows with a source tag and plain reason", () => {
    const html = renderToStaticMarkup(<UnsupportedModels models={models} />)
    expect(html).toContain("<details")
    expect(html).not.toContain("<details open")
    expect(html).toContain(unsupportedSummary(3))
    expect(html).toContain("3 other models on this computer can’t run in Magnitude")
    expect(html).toContain("No kernels for the deepseek4 architecture")
    expect(html).toContain("Uses IQ1_M weights")
    expect(html).toContain(">LM Studio<")
    expect(html).toContain(">Hugging Face cache<")
    expect(html).toContain("text-slate-500")
    expect(html).not.toMatch(/<button/)
  })
  it("shows an unknown reason's detail in the row itself", () => {
    const html = renderToStaticMarkup(<UnsupportedModels models={models} />)
    expect(html).toMatch(/>gpt-oss 20B<.*>Ollama<.*<span class="min-w-0 flex-1">Magnitude can&#x27;t run this model \(PRIVATE\)<\/span>/)
  })
  it("says many unrecognized tensors once", () => {
    const gemma = unsupportedLocal("Gemma 4 E2B (Q4_K_M)", foundIn({ _tag: "LmStudio" }), { _tag: "UnrecognizedTensor", name: "blk.15.attn_k" })
    const tensors = Array.from({ length: 60 }, (_, index) => ({ _tag: "UnrecognizedTensor" as const, name: `blk.${15 + index}.attn_k` }))
    const html = renderToStaticMarkup(<UnsupportedModels models={[{ ...gemma, reasons: [tensors[0]!, ...tensors.slice(1)] }]} />)
    expect(html).toContain("Has parts Magnitude doesn&#x27;t recognize (60 tensors)")
    expect(html).not.toContain("blk.15.attn_k")
  })
  it("is absent when everything runs", () => {
    expect(renderToStaticMarkup(<UnsupportedModels models={[]} />)).toBe("")
  })
})

describe("hidden filter", () => {
  it("lists hidden models with Unhide, and says when there are none", () => {
    const html = renderToStaticMarkup(<HiddenModels models={[customModel(foundIn({ _tag: "Ollama" }))]} unhide={model => <UnhideButton pending={false} onUnhide={() => model} />} />)
    expect(html).toContain("Hidden from Magnitude")
    expect(html).toMatch(/<button[^>]*>.*Unhide<\/button>/)
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>.*Unhide/)
    expect(html).toContain(">Ollama<")
    expect(renderToStaticMarkup(<HiddenModels models={[]} unhide={() => null} />)).toContain("No hidden models")
  })
  it("disables Unhide while its command is pending", () => {
    expect(renderToStaticMarkup(<UnhideButton pending onUnhide={() => {}} />)).toMatch(/<button[^>]*disabled=""[^>]*>.*Unhide<\/button>/)
  })
  it("tags where files came from", () => {
    expect(renderToStaticMarkup(<SourceTag label="Hugging Face" />)).toContain(">Hugging Face<")
  })
})

describe("discover", () => {
  const found = [
    customModel(foundIn({ _tag: "Ollama" })),
    customModel(foundIn({ _tag: "LmStudio" }), { modelId: customModel(imported).modelId }),
    customModel(foundIn({ _tag: "HuggingFaceCache" })),
  ]
  it("summarizes found models that fit, where they were found, and their labs", () => {
    expect(foundModelsSummary(found)).toEqual(Option.some(expect.objectContaining({ count: 3, places: "Ollama, LM Studio, and your Hugging Face cache" })))
    const html = renderToStaticMarkup(<FoundModelsBanner models={found} onView={() => {}} />)
    expect(html).toContain("You already have 3 models that run here.")
    expect(html).toContain("Found in Ollama, LM Studio, and your Hugging Face cache.")
    expect(html).toContain("View in My Models")
    expect(html).toContain("Qwen logo")
  })
  it("counts only found models that fit", () => {
    const tooLarge = customModel(foundIn({ _tag: "Ollama" }), { servingState: { ...found[0]!.servingState, _tag: "Assessed", assessment: doesNotFit } as typeof found[0]["servingState"] })
    expect(renderToStaticMarkup(<FoundModelsBanner models={[customModel(added), customModel(imported), tooLarge]} onView={() => {}} />)).toBe("")
    expect(renderToStaticMarkup(<FoundModelsBanner models={[customModel(foundIn({ _tag: "Folder", path: "/Volumes/External/models" }), { presentation: { ...found[0]!.presentation, sourceLabel: "models" } })]} onView={() => {}} />))
      .toContain("You already have 1 model that runs here.</span> <span class=\"text-slate-600 dark:text-slate-300\">Found in your models folder.")
  })
  it("adds one Hugging Face line under the recommendations, and offers it beside Explore Catalog when nothing fits", () => {
    const link = renderToStaticMarkup(<HuggingFaceLink onAdd={() => {}} />)
    expect(link).toContain("Looking for a specific version, finetune, or uncensored model?")
    expect(link).toContain("Add from Hugging Face")
    expect(link).not.toContain("border-dashed")
    const empty = renderToStaticMarkup(<NoRecommendations onExplore={() => {}} onAdd={() => {}} />)
    expect(empty).toContain("No fitting recommendations right now.")
    expect(empty).toContain("Explore Catalog")
    expect(empty).toContain("Add from Hugging Face")
  })
})
