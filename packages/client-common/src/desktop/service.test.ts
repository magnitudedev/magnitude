import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { modelTrayPresentation } from "./service"
import { makeSetupModel } from "./fixtures/model"
import type { CatalogLocalModel, LocalModelsState } from "@magnitudedev/sdk"

const observed = (residencyState: Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>["residencyState"]): LocalModelsState => ({
  preparation: { discovery: { complete: true, modelsFound: 1 }, assessment: { complete: true, settledModels: 1, totalModels: 1 } },
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
    expect(modelTrayPresentation(observed({ _tag: "Unloaded" }))).toEqual({ label: "No model loaded", canStop: false })
  })
})
