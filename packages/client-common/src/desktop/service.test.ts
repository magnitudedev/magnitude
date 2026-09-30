import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { modelTrayPresentation } from "./service"
import { makeSetupModel } from "./fixtures/model"
import { LocalInferenceMemoryDomainIdSchema, type CatalogLocalModel, type LocalModelsState } from "@magnitudedev/sdk"

const observed = (residencyState: Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>["residencyState"]): LocalModelsState => ({
  preparation: { discovery: { complete: true, modelsFound: 1 }, assessment: { complete: true, settledModels: 1, totalModels: 1, failure: Option.none() } },
  models: [{ ...makeSetupModel(true), acquisitionState: {
    _tag: "Installed", installation: { _tag: "Resolved", primaryPath: "/models/test.gguf", installedBytes: 1, ownership: "Magnitude" }, residencyState,
  } }],
})

describe("desktop model retirement controls", () => {
  it("keeps Stop available while native cleanup is incomplete", () => {
    const presentation = modelTrayPresentation(observed({ _tag: "Stopping", reason: "user_stop", allocation: { _tag: "Planned", allocation: Option.none() } }))
    expect(presentation.label).toContain("Stopping")
    expect(presentation.canStop).toBe(true)
  })
  it("removes Stop only after observed unloading", () => {
    expect(modelTrayPresentation(observed({ _tag: "Unloaded" }))).toEqual({ label: "No model loaded", status: Option.none(), canStop: false })
  })
})

describe("tray model line", () => {
  it("shows a measured stage's progress and an unmeasured stage's work", () => {
    const optimizing = modelTrayPresentation(observed({ _tag: "Loading", stage: "optimizing", fraction: 0.34, plannedAllocation: Option.none() }))
    expect(Option.map(optimizing.status, ({ phase, detail }) => ({ phase, detail }))).toEqual(
      Option.some({ phase: "Optimizing", detail: { _tag: "Progress", fraction: 0.34 } }))
    expect(optimizing.label).toMatch(/ · Optimizing 34%$/)
    const preparing = modelTrayPresentation(observed({ _tag: "Loading", stage: "preparing", fraction: 0, plannedAllocation: Option.none() }))
    expect(Option.map(preparing.status, ({ phase, detail }) => ({ phase, detail }))).toEqual(
      Option.some({ phase: "Preparing", detail: { _tag: "Working" } }))
  })
  it("shows the memory a loaded model holds", () => {
    const loaded = modelTrayPresentation(observed({ _tag: "Ready", allocation: { contextWindowTokens: 4096, memoryDomains: [
      { memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("unified"), modelBytes: 16 * 1024 ** 3, contextBytes: 2 * 1024 ** 3, computeBytes: 0, auxiliaryBytes: 0 },
    ] } }))
    expect(Option.map(loaded.status, ({ detail }) => detail)).toEqual(Option.some({ _tag: "Memory", text: "18 GB" }))
    expect(loaded.label).toMatch(/ · Loaded · 18 GB$/)
  })
})
