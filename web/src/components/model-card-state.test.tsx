import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { ModelReleaseDateSchema, type CatalogLocalModel, type CustomLocalModel } from "@magnitudedev/sdk"
import { addedModelAwaitingFiles, modelCanDownload, modelCardDrafter, modelCardStatus, modelSubtitle, modelTransferEstimate, modelUpdateDescription } from "./model-card-state"
import { added, attachedDrafter, catalogModel, customModel, doesNotFit, drafterCandidate, foundIn, inFolder } from "../../test/models/fixtures"

const moe = (): CatalogLocalModel => {
  const model = catalogModel()
  return { ...model, catalogData: { ...model.catalogData, releaseDate: ModelReleaseDateSchema.make("2020-01-01"), parameterization: { architecture: "mixtureOfExperts", totalParameters: 35_000_000_000, activeParameters: 3_000_000_000 } } }
}

describe("model card subtitle", () => {
  it("names the maker and release age on Catalog, and the maker and size on My Models", () => {
    expect(modelSubtitle(moe(), [moe()], "catalog")).toMatch(/^Qwen · \d+ days ago$/)
    expect(modelSubtitle(moe(), [moe()], "models")).toBe("Qwen · 35B total · 3B active")
  })
  it("names an added model's uploader and the catalog model it is a variant of, never its source", () => {
    expect(modelSubtitle(customModel(added), [catalogModel()], "models")).toBe("HauhauCS · Variant of Qwen3.6 35B-A3B")
    expect(modelSubtitle(customModel(foundIn({ _tag: "LmStudio" })), [catalogModel()], "models")).toBe("Qwen · Variant of Qwen3.6 35B-A3B")
  })
})

describe("model card drafter", () => {
  it("offers a catalog model's missing drafter and shows the one it serves with", () => {
    expect(modelCardDrafter({ ...catalogModel(), missingComponents: [{ component: "Drafter", downloadBytes: 59_200_000 }] })).toEqual({ _tag: "Add" })
    expect(modelCardDrafter(catalogModel())).toEqual({ _tag: "None", changeable: false })
    expect(modelCardDrafter({ ...catalogModel(), drafterWithheld: true })._tag).toBe("Withheld")
  })
  it("lets a custom model change its drafter in place, with Update when a newer revision exists", () => {
    const attached = modelCardDrafter(customModel(added, { drafter: Option.some(attachedDrafter({ _tag: "Present" }, { updateAvailable: true })) }))
    expect(attached).toMatchObject({ _tag: "Attached", title: "Magnitude DFlash for Qwen3.6 35B-A3B", changeable: true, update: true })
    expect(modelCardDrafter(customModel(inFolder))).toEqual({ _tag: "None", changeable: true })
  })
  it("offers the recommended drafter only while none is attached or turned off", () => {
    expect(modelCardDrafter(customModel(added, { recommendedDrafter: Option.some(drafterCandidate()) }))).toEqual({ _tag: "Add" })
    expect(modelCardDrafter(customModel(added, { drafterChoice: { _tag: "None" }, recommendedDrafter: Option.some(drafterCandidate()) }))).toEqual({ _tag: "None", changeable: true })
  })
  it("shows an automatic drafter that isn't worked out yet as pending, never as none", () => {
    expect(modelCardDrafter(customModel(inFolder, { drafterPending: true }))).toEqual({ _tag: "Pending" })
    expect(modelCardDrafter(customModel(added, { drafterPending: true, recommendedDrafter: Option.some(drafterCandidate()) }))).toEqual({ _tag: "Pending" })
  })
  it("shows a drafter-only download's progress in the cell", () => {
    const downloading = attachedDrafter({ _tag: "Downloading", progress: { stage: "downloading", completedBytes: 450, totalBytes: 900, bytesPerSecond: Option.none() } })
    expect(modelCardDrafter(customModel(added, { drafter: Option.some(downloading) }))).toEqual({ _tag: "Adding", percent: 50 })
  })
})

describe("model card status", () => {
  it("says where the model is, from not downloaded to running", () => {
    expect(modelCardStatus({ ...catalogModel(), acquisitionState: { _tag: "NotInstalled" } }, Option.none())).toEqual({ label: "Not downloaded", tone: "absent" })
    expect(modelCardStatus(catalogModel(), Option.none())).toEqual({ label: "Downloaded", tone: "stored" })
    expect(modelCardStatus(customModel(foundIn({ _tag: "LmStudio" })), Option.none())).toEqual({ label: "On this computer", tone: "stored" })
    const running = customModel(added, { acquisitionState: { _tag: "Installed", installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/m.gguf", ownership: "Magnitude" }, residencyState: { _tag: "Ready" } } as never })
    expect(modelCardStatus(running, Option.none())).toEqual({ label: "Running", tone: "ready" })
  })
})

describe("model card download", () => {
  const progress = { stage: "downloading" as const, completedBytes: 240_000_000, totalBytes: 6_000_000_000, bytesPerSecond: Option.some(24_000_000) }
  it("estimates remaining time from bytes still to transfer", () => {
    expect(modelTransferEstimate(progress)).toBe("est. 4 min")
    expect(modelTransferEstimate({ ...progress, completedBytes: 5_000_000_000 })).toBe("est. <1 min")
  })
  it("waits for a usable rate and known total before claiming an ETA", () => {
    expect(modelTransferEstimate({ ...progress, bytesPerSecond: Option.none() })).toBe("Estimating…")
    expect(modelTransferEstimate({ ...progress, bytesPerSecond: Option.some(0) })).toBe("Estimating…")
    expect(modelTransferEstimate({ ...progress, completedBytes: 0, totalBytes: 0 })).toBe("Estimating…")
  })
  it("clears the estimate during queueing, preparation, verification and publishing", () => {
    for (const stage of ["queued", "resolving", "checking_space", "verifying", "publishing"] as const) expect(modelTransferEstimate({ ...progress, stage })).toBeNull()
  })
  // ACN projects an added model whose target hasn't landed as Assessing: it is assessed only from its files.
  const notDownloaded = (acquisitionState: CustomLocalModel["acquisitionState"]) =>
    customModel(added, { acquisitionState, servingState: { _tag: "Assessing", profile: { contextLength: 100_000 } }, storageBytes: 0, drafterPending: true })
  it("keeps Download for an added model whose download was cancelled or failed, though it has no assessment", () => {
    const cancelled = notDownloaded({ _tag: "NotInstalled" })
    expect(addedModelAwaitingFiles(cancelled)).toBe(true)
    expect(modelCanDownload(cancelled)).toBe(true)
    expect(modelCanDownload(notDownloaded({ _tag: "InstallFailed", failure: { _tag: "Interrupted" } } as never))).toBe(true)
  })
  it("downloads a catalog model only once it is assessed to fit", () => {
    const absent = { ...catalogModel(), acquisitionState: { _tag: "NotInstalled" as const } }
    expect(addedModelAwaitingFiles(absent)).toBe(false)
    expect(modelCanDownload(absent)).toBe(true)
    expect(modelCanDownload({ ...absent, servingState: { _tag: "Assessing", profile: { contextLength: 100_000 } } } as never)).toBe(false)
    expect(modelCanDownload({ ...absent, servingState: { ...absent.servingState, assessment: doesNotFit } } as never)).toBe(false)
  })
  it("never treats a downloaded or found model as awaiting files", () => {
    expect(addedModelAwaitingFiles(customModel(added))).toBe(false)
    expect(addedModelAwaitingFiles(customModel(foundIn({ _tag: "LmStudio" })))).toBe(false)
  })
})

describe("model update description", () => {
  const updating = (parts: readonly ["Model" | "ImageInput" | "Drafter", ...("Model" | "ImageInput" | "Drafter")[]]): CatalogLocalModel => {
    const model = catalogModel()
    return { ...model, acquisitionState: { ...(model.acquisitionState as Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>), _tag: "UpdateAvailable", parts } }
  }
  it("offers a drafter-only catalog update in the Drafter cell, not as the model's Update", () => {
    expect(modelCardDrafter(updating(["Drafter"]))).toEqual({ _tag: "Add" })
    expect(modelCardDrafter(updating(["Model", "Drafter"]))).toEqual({ _tag: "None", changeable: false })
  })
  it("says which part of a catalog model is new", () => {
    expect(modelUpdateDescription(updating(["Drafter"]))).toEqual(Option.some("A newer version of the drafter is available."))
    expect(modelUpdateDescription(updating(["Model"]))).toEqual(Option.some("A newer version of the model is available."))
    expect(modelUpdateDescription(updating(["Model", "ImageInput", "Drafter"]))).toEqual(Option.some("Newer versions of the model, image input and drafter are available."))
  })
  it("says an added model's repository has a newer version, and nothing without an update", () => {
    const added_ = customModel(added)
    const installed = added_.acquisitionState as Extract<CustomLocalModel["acquisitionState"], { _tag: "Installed" }>
    expect(modelUpdateDescription({ ...added_, acquisitionState: { ...installed, _tag: "UpdateAvailable", parts: ["Model"] } })).toEqual(Option.some("The Hugging Face repository has a newer version of this model."))
    expect(modelUpdateDescription(catalogModel())).toEqual(Option.none())
  })
})
