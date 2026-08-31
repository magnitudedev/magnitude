import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalModel } from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  installedLocalModels,
  modelDownloadFailureMessage,
  localModelMaximumContextLength,
  performanceRange,
  selectionModelId,
  selectionMetadata,
  selectionProviderModelId,
} from "./view-model"
import {
  GIB,
  makeAcquiringModel,
  makeCatalogModel,
  makeModel,
  makeView,
  withDoesNotFitAssessment,
} from "./test-fixtures"

describe("unified local inference projection", () => {
  it("formats structured insufficient-disk failures as passive user guidance", () => {
    expect(modelDownloadFailureMessage({
      _tag: "InsufficientDiskSpace",
      requiredBytes: 37_923_968_128,
      availableBytes: 33_440_665_600,
    })).toBe("Not enough disk space. Free at least 4.48 GB and try again.")
  })

  it("references the canonical installed model without copying it", () => {
    const model = makeModel()
    const view = makeView({ models: [model] })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)
    expect(selection?.model).toBe(model)
    expect(selectionModelId(selection!)).toBe(model.modelId)
    expect(Option.getOrThrow(selectionProviderModelId(selection!))).toBeDefined()
  })

  it("publishes one downloadable row for one scored catalog model", () => {
    const model = makeCatalogModel()
    const view = makeView({ models: [model], ready: false })
    const selections = buildLocalInferenceSelections(view.models, view.slots)
    expect(selections).toHaveLength(1)
    expect(selections[0]?.model).toBe(model)
    expect(selections[0]?.kind).toBe("downloadable")
  })

  it("keeps active acquisition visible without manufacturing a download record", () => {
    const model = makeAcquiringModel({
      _tag: "Installing",
      progress: {
        stage: "downloading",
        completedBytes: GIB,
        totalBytes: 2 * GIB,
        bytesPerSecond: Option.none(),
      },
    })
    const view = makeView({ models: [model], ready: false })
    expect(buildLocalInferenceSelections(view.models, view.slots)[0]?.model).toBe(model)
  })

  it("filters installed choices from acquisition and assessment on the same model", () => {
    const installed = makeModel()
    const catalog = makeCatalogModel()
    expect(installedLocalModels(makeView({ models: [installed, catalog] }).models))
      .toEqual([installed])
  })

  it("retains an unavailable Hugging Face discovery as installed material", () => {
    const base = makeModel()
    const failure = { code: "invalid_artifact", message: "Invalid artifact", retryable: false }
    const unavailable: LocalModel = {
      ...base,
      modelId: base.modelId,
      state: {
        _tag: "Unavailable",
        installation: base.state.installation,
        failure,
      },
    }
    expect(installedLocalModels(makeView({ models: [unavailable] }).models))
      .toEqual([unavailable])
  })

  it("excludes non-fitting models from onboarding selections", () => {
    const doesNotFit = withDoesNotFitAssessment(makeCatalogModel())
    const view = makeView({ models: [doesNotFit], ready: false })

    expect(buildLocalInferenceSelections(view.models, view.slots)).toEqual([])
  })

  it("reads calibrated performance from the model Fits assessment", () => {
    expect(performanceRange(makeCatalogModel())).toMatchObject({
      lowerContext: 25_000,
      upperContext: 32_768,
    })
  })

  it("uses canonical model identity and serving metadata without exposing bundles", () => {
    const model = makeModel()
    expect(model.modelId).toBe("hf:test/model/model-q4.gguf")
    expect(localModelMaximumContextLength(model)).toEqual(Option.some(32_768))
  })

  it("shows the configured speculative method in selection metadata", () => {
    const base = makeModel()
    if (base.state.servingState._tag !== "Assessed") throw new Error("fixture must be assessed")
    const model = makeModel({ state: {
      ...base.state,
      servingState: {
        ...base.state.servingState,
        speculativeMethod: Option.some({ _tag: "DFlash" }),
      },
    } })
    const view = makeView({ models: [model] })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)

    expect(selectionMetadata(selection!)).toMatch(/ · DFlash$/)

    const embedded = makeModel({ state: {
      ...base.state,
      servingState: {
        ...base.state.servingState,
        speculativeMethod: Option.some({ _tag: "Mtp" }),
      },
    } })
    const [embeddedSelection] = buildLocalInferenceSelections(
      makeView({ models: [embedded] }).models,
      makeView().slots,
    )
    expect(selectionMetadata(embeddedSelection!)).toMatch(/ · MTP$/)

    const standalone = makeModel()
    const [standaloneSelection] = buildLocalInferenceSelections(
      makeView({ models: [standalone] }).models,
      makeView().slots,
    )
    expect(selectionMetadata(standaloneSelection!)).not.toMatch(/ · (MTP|DFlash|DSpark)$/)
  })
})
