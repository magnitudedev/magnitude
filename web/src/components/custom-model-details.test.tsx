import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { AttachedDrafterAvailability } from "@magnitudedev/sdk"
import { customModelSource, drafterUpdateAvailable } from "./custom-model-details"
import { added, attachedDrafter, catalogModel, customModel, foundIn, imported, repository } from "../../test/models/fixtures"

const attached = (availability: AttachedDrafterAvailability, updateAvailable = false) => Option.some(attachedDrafter(availability, { updateAvailable }))
const downloading: AttachedDrafterAvailability = { _tag: "Downloading", progress: { stage: "downloading", completedBytes: 450, totalBytes: 900, bytesPerSecond: Option.none() } }

describe("custom model details", () => {
  it("names the source by origin: a pinned repository, an imported path, or where it was found", () => {
    expect(customModelSource(customModel(added))).toEqual({ label: `${repository} · pinned to 4f1c2e9`, href: Option.some(`https://huggingface.co/${repository}`) })
    expect(customModelSource(customModel(imported)).label).toBe("/Users/me/models/qwen.gguf")
    expect(customModelSource(customModel(foundIn({ _tag: "Ollama" }))).label).toBe("Ollama · /Users/me/.ollama/models/blobs/sha256-1")
  })
  it("shows Update for any custom model whose catalog drafter has a newer revision, once nothing is downloading it", () => {
    expect(drafterUpdateAvailable(customModel(foundIn({ _tag: "Ollama" }), { drafter: attached({ _tag: "Present" }, true) }))).toBe(true)
    expect(drafterUpdateAvailable(customModel(added, { drafter: attached({ _tag: "DownloadFailed", failure: { _tag: "NetworkUnavailable" } }, true) }))).toBe(true)
    expect(drafterUpdateAvailable(customModel(added, { drafter: attached(downloading, true) }))).toBe(false)
    expect(drafterUpdateAvailable(customModel(added, { drafter: attached({ _tag: "Present" }) }))).toBe(false)
    expect(drafterUpdateAvailable(customModel(added))).toBe(false)
    expect(drafterUpdateAvailable(catalogModel())).toBe(false)
  })
})
