import { type LocalModel, type ModelResidency } from "@magnitudedev/sdk"
import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeInstalledCatalogModel,
  makeModel,
} from "../features/local-inference/test-fixtures"
import {
  JsonLocalModelSchema,
  ModelsLoadJsonDataSchema,
  ModelsStatusJsonDataSchema,
  ModelsStopJsonDataSchema,
  localModelJson,
  modelsLoadJsonData,
  modelsStatusJsonData,
  modelsStopJsonData,
} from "./models-json"

const encodedModel = (model: LocalModel) => Schema.encodeSync(JsonLocalModelSchema)(localModelJson(model))

const transfer = {
  stage: "downloading" as const,
  completedBytes: 25,
  totalBytes: 100,
  bytesPerSecond: Option.some(50),
}

const installedFields = () => {
  const model = makeInstalledCatalogModel()
  if (model.acquisitionState._tag !== "Installed") throw new Error("Expected installed fixture")
  return {
    installation: model.acquisitionState.installation,
    residencyState: model.acquisitionState.residencyState,
  }
}

const withResidency = (residencyState: ModelResidency): LocalModel => {
  const model = makeInstalledCatalogModel()
  return {
    ...model,
    acquisitionState: {
      _tag: "Installed",
      installation: installedFields().installation,
      residencyState,
    },
  }
}

describe("model command JSON projection", () => {
  it("normalizes every catalog installation state to the small plugin vocabulary", () => {
    const installed = installedFields()
    const cases: readonly [LocalModel, string][] = [
      [makeCatalogModel(), "not_installed"],
      [makeAcquiringModel({ _tag: "Installing", progress: transfer }), "installing"],
      [makeAcquiringModel({ _tag: "InstallFailed", failure: { _tag: "Interrupted" } }), "unavailable"],
      [makeInstalledCatalogModel(), "installed"],
      [makeAcquiringModel({ _tag: "UpdateAvailable", ...installed }), "installed"],
      [makeAcquiringModel({ _tag: "Updating", ...installed, progress: transfer }), "installed"],
      [makeAcquiringModel({
        _tag: "UpdateFailed",
        ...installed,
        failure: { _tag: "NetworkUnavailable" },
      }), "installed"],
      [makeAcquiringModel({ _tag: "Removing", ...installed }), "removing"],
      [makeAcquiringModel({
        _tag: "RemoveFailed",
        ...installed,
        failure: { code: "remove_failed", message: "Removal failed", retryable: true },
      }), "unavailable"],
    ]

    for (const [model, installation] of cases) {
      const encoded = encodedModel(model)
      expect(encoded.installation).toBe(installation)
      expect(Object.keys(encoded).every((key) =>
        ["displayName", "installation", "modelId", "residency"].includes(key)
      )).toBe(true)
      expect(encoded).not.toHaveProperty("source")
      expect(encoded).not.toHaveProperty("memoryBytes")
      expect(encoded).not.toHaveProperty("contextLength")
      expect(JSON.stringify(encoded)).not.toContain('"_tag"')
    }
  })

  it("normalizes unavailable discovered models without exposing failure internals", () => {
    const ready = makeModel()
    const unavailable: LocalModel = {
      ...ready,
      state: {
        _tag: "Unavailable",
        installation: ready.state.installation,
        failure: { code: "unavailable", message: "Cannot inspect model", retryable: true },
      },
    }

    expect(encodedModel(unavailable)).toEqual({
      modelId: unavailable.modelId,
      displayName: `${unavailable.presentation.displayName} (${unavailable.presentation.variantLabel})`,
      installation: "unavailable",
    })
  })

  it("normalizes every residency state without stages, progress, allocation, or failure details", () => {
    const allocation = {
      contextWindowTokens: 32_768,
      parallelSequences: 1,
      physicalContextTokens: 32_768,
      memoryDomains: [],
    }
    const cases: readonly [ModelResidency, string][] = [
      [{ _tag: "Unloaded" }, "unloaded"],
      [{ _tag: "Requested" }, "loading"],
      [{ _tag: "Loading", stage: "loading", progress: Option.some(0.25), plannedAllocation: Option.none() }, "loading"],
      [{ _tag: "Ready", allocation }, "ready"],
      [{ _tag: "Stopping", reason: "user_stop", allocation: { _tag: "Resident", allocation } }, "stopping"],
      [{
        _tag: "Failed",
        failure: { code: "load_failed", message: "Could not load", retryable: true },
      }, "failed"],
    ]

    for (const [residency, expected] of cases) {
      const encoded = encodedModel(withResidency(residency))
      expect(encoded.residency).toBe(expected)
      expect(Object.keys(encoded).sort()).toEqual(["displayName", "installation", "modelId", "residency"].sort())
    }
  })

  it("keeps list visibility and ordering identical to human model status", () => {
    const hidden = makeCatalogModel()
    const discovered = makeModel()
    const installed = makeInstalledCatalogModel({
      presentation: { ...hidden.presentation, displayName: "Alpha" },
    })
    const encoded = Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "List",
      models: [discovered, hidden, installed],
    }))

    expect(encoded).toMatchObject({ state: "ready" })
    expect(encoded.models.map(({ modelId }) => modelId)).toEqual([installed.modelId, discovered.modelId])
  })

  it("uses one stable array shape for initialization, list, and addressed status", () => {
    expect(Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "Initializing",
    }))).toEqual({ state: "initializing", models: [] })

    expect(Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "List",
      models: [],
    }))).toEqual({ state: "ready", models: [] })

    const model = makeCatalogModel()
    expect(Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "Detail",
      model,
    }))).toEqual({
      state: "ready",
      models: [{
        modelId: model.modelId,
        displayName: `${model.presentation.displayName} (${model.presentation.variantLabel})`,
        installation: "not_installed",
      }],
    })
  })

  it("defines minimal mutation acknowledgements", () => {
    const model = makeModel()
    expect(Schema.encodeSync(ModelsLoadJsonDataSchema)(modelsLoadJsonData(model.modelId)))
      .toEqual({ modelId: model.modelId })
    expect(Schema.encodeSync(ModelsStopJsonDataSchema)(modelsStopJsonData())).toEqual({})
  })
})
