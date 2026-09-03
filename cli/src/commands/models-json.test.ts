import {
  type LocalModel,
  type ModelResidency,
} from "@magnitudedev/sdk"
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

const encodedModel = (model: LocalModel) =>
  Schema.encodeSync(JsonLocalModelSchema)(localModelJson(model))

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
  it("projects every catalog installation state without leaking internal tags", () => {
    const installed = installedFields()
    const cases: readonly [LocalModel, string][] = [
      [makeCatalogModel(), "not_installed"],
      [makeAcquiringModel({ _tag: "Installing", progress: transfer }), "installing"],
      [makeAcquiringModel({ _tag: "InstallFailed", failure: { _tag: "Interrupted" } }), "install_failed"],
      [makeInstalledCatalogModel(), "installed"],
      [makeAcquiringModel({ _tag: "UpdateAvailable", ...installed }), "update_available"],
      [makeAcquiringModel({ _tag: "Updating", ...installed, progress: transfer }), "updating"],
      [makeAcquiringModel({
        _tag: "UpdateFailed",
        ...installed,
        failure: { _tag: "NetworkUnavailable" },
      }), "update_failed"],
      [makeAcquiringModel({ _tag: "Removing", ...installed }), "removing"],
      [makeAcquiringModel({
        _tag: "RemoveFailed",
        ...installed,
        failure: { code: "remove_failed", message: "Removal failed", retryable: true },
      }), "remove_failed"],
    ]

    for (const [model, state] of cases) {
      const encoded = encodedModel(model)
      expect(encoded.installation.state).toBe(state)
      expect(JSON.stringify(encoded)).not.toContain('"_tag"')
    }
  })

  it("projects every acquisition failure category as a stable product message", () => {
    const failures = [
      { _tag: "Interrupted" as const },
      { _tag: "InsufficientDiskSpace" as const, requiredBytes: 100, availableBytes: 25 },
      { _tag: "SourceUnavailable" as const },
      { _tag: "NetworkUnavailable" as const },
      { _tag: "LocalStorageFailure" as const },
      { _tag: "CorruptDownload" as const },
      { _tag: "Internal" as const, message: "Internal transfer failure" },
    ]

    for (const failure of failures) {
      const encoded = encodedModel(makeAcquiringModel({ _tag: "InstallFailed", failure }))
      expect(encoded.installation.state).toBe("install_failed")
      if (encoded.installation.state !== "install_failed") continue
      expect(encoded.installation.error.message.length).toBeGreaterThan(0)
    }
  })

  it("preserves exact transfer quantities and omits an unavailable transfer rate", () => {
    const withRate = encodedModel(makeAcquiringModel({ _tag: "Installing", progress: transfer }))
    expect(withRate.installation).toEqual({
      state: "installing",
      progress: {
        stage: "downloading",
        completedBytes: 25,
        totalBytes: 100,
        bytesPerSecond: 50,
      },
    })

    const withoutRate = encodedModel(makeAcquiringModel({
      _tag: "Installing",
      progress: { ...transfer, bytesPerSecond: Option.none() },
    }))
    expect(withoutRate.installation).not.toHaveProperty("progress.bytesPerSecond")
    expect(JSON.stringify(withoutRate)).not.toContain("null")
  })

  it("projects unavailable discovered models as installation failures without residency", () => {
    const ready = makeModel()
    const unavailable: LocalModel = {
      ...ready,
      state: {
        _tag: "Unavailable",
        installation: ready.state.installation,
        failure: { code: "unavailable", message: "Cannot inspect model", retryable: true },
      },
    }
    const encoded = encodedModel(unavailable)
    expect(encoded.installation).toEqual({
      state: "unavailable",
      error: { message: "Cannot inspect model" },
    })
    expect(encoded).not.toHaveProperty("residency")
    expect(encoded).not.toHaveProperty("contextLength")
  })

  it("projects every residency state and only its state-specific fields", () => {
    const allocation = {
      contextWindowTokens: 32_768,
      parallelSequences: 1,
      physicalContextTokens: 32_768,
      memoryDomains: [],
    }
    const cases: readonly [ModelResidency, unknown][] = [
      [{ _tag: "Unloaded" }, { state: "unloaded" }],
      [{ _tag: "Requested" }, { state: "requested" }],
      [{
        _tag: "Loading",
        stage: "loading",
        progress: Option.some(0.25),
        plannedAllocation: Option.none(),
      }, { state: "loading", stage: "loading", progress: 0.25 }],
      [{
        _tag: "Loading",
        stage: "verifying",
        progress: Option.none(),
        plannedAllocation: Option.none(),
      }, { state: "loading", stage: "verifying" }],
      [{ _tag: "Ready", allocation }, { state: "ready" }],
      [{
        _tag: "Stopping",
        reason: "user_stop",
        allocation: { _tag: "Resident", allocation },
      }, { state: "stopping", reason: "user_stop" }],
      [{
        _tag: "Failed",
        failure: { code: "load_failed", message: "Could not load", retryable: true },
      }, { state: "failed", error: { message: "Could not load", retryable: true } }],
    ]

    for (const [residency, expected] of cases) {
      const encoded = encodedModel(withResidency(residency))
      expect(encoded).toHaveProperty("residency")
      if (!("residency" in encoded)) continue
      expect(encoded.residency).toEqual(expected)
    }
  })

  it("projects low-memory residency failures without leaking allocation diagnostics", () => {
    const encoded = encodedModel(withResidency({
      _tag: "Failed",
      failure: {
        _tag: "LowMemory",
        code: "low_memory",
        message: "More memory is required",
        retryable: false,
        requiredSystemMemoryBytes: 100,
        allocationHeadroomBytes: 20,
        systemReserveBytes: 10,
        loadBoundaryBytes: 90,
        minimumAdditionalAvailableBytes: 70,
        parallelSequences: 1,
      },
    }))
    expect(encoded).toHaveProperty("residency")
    if (!("residency" in encoded)) return
    expect(encoded.residency).toEqual({
      state: "failed",
      error: { message: "More memory is required", retryable: false },
    })
    expect(JSON.stringify(encoded)).not.toContain("requiredSystemMemoryBytes")
  })

  it("emits byte-accurate memory, exact context, canonical identity, and friendly display name", () => {
    const model = makeInstalledCatalogModel()
    const encoded = encodedModel(model)
    expect(encoded.modelId).toBe(model.modelId)
    expect(encoded.displayName).toContain(model.presentation.displayName)
    expect(encoded.source).toBe("catalog")
    expect(encoded).toHaveProperty("memoryBytes")
    expect(encoded).toHaveProperty("contextLength")
    if (!("memoryBytes" in encoded) || !("contextLength" in encoded)) return
    expect(encoded.memoryBytes).toBe(0)
    expect(encoded.contextLength).toBe(32_768)
  })

  it("keeps list visibility and ordering identical to human model status", () => {
    const hidden = makeCatalogModel()
    const discovered = makeModel()
    const installed = makeInstalledCatalogModel({
      presentation: { ...hidden.presentation, displayName: "Alpha" },
    })
    const data = modelsStatusJsonData({ _tag: "List", models: [discovered, hidden, installed] })
    const encoded = Schema.encodeSync(ModelsStatusJsonDataSchema)(data)
    expect(encoded).toMatchObject({ state: "ready", view: "list" })
    if (encoded.state !== "ready" || encoded.view !== "list") return
    expect(encoded.models.map(({ modelId }) => modelId)).toEqual([installed.modelId, discovered.modelId])
  })

  it("distinguishes initializing, empty list, and addressed detail documents", () => {
    expect(Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "Initializing",
      view: "list",
    }))).toEqual({ state: "initializing", view: "list" })
    expect(Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "Initializing",
      view: "detail",
    }))).toEqual({ state: "initializing", view: "detail" })

    expect(Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "List",
      models: [],
    }))).toEqual({ state: "ready", view: "list", models: [] })

    const model = makeCatalogModel()
    const detail = Schema.encodeSync(ModelsStatusJsonDataSchema)(modelsStatusJsonData({
      _tag: "Detail",
      model,
    }))
    expect(detail).toMatchObject({
      state: "ready",
      view: "detail",
      model: { modelId: model.modelId, installation: { state: "not_installed" } },
    })
  })

  it("defines exact mutation acknowledgements", () => {
    const model = makeModel()
    expect(Schema.encodeSync(ModelsLoadJsonDataSchema)(modelsLoadJsonData(model.modelId)))
      .toEqual({ modelId: model.modelId, outcome: "load_requested" })
    expect(Schema.encodeSync(ModelsStopJsonDataSchema)(modelsStopJsonData()))
      .toEqual({ outcome: "stopped" })
  })
})
