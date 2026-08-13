import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { ModelDownloadIdSchema } from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  formatDownloadBytes,
  installedLocalModels,
  localModelBundleKey,
  localModelMaximumContextLength,
  performanceRange,
  selectionConfigurationId,
  selectionProviderModelId,
} from "./view-model"
import {
  GIB,
  makeAcquiringModel,
  makeCatalogModel,
  makeModel,
  makeRecommendation,
  makeStandaloneBundle,
  makeView,
  withDoesNotFitAssessment,
} from "./test-fixtures"

describe("unified local inference projection", () => {
  it("formats model artifacts in decimal gigabytes", () => {
    expect(formatDownloadBytes(30 * GIB)).toBe("32.2 GB")
  })

  it("references the canonical installed model without copying it", () => {
    const model = makeModel()
    const view = makeView({ models: [model] })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)
    expect(selection?.model).toBe(model)
    expect(Option.getOrThrow(selectionConfigurationId(selection!)))
      .toBe(model.servingState._tag === "Assessed" ? model.servingState.configuration.id : undefined)
    expect(Option.getOrThrow(selectionProviderModelId(selection!))).toBeDefined()
  })

  it("publishes one row when one model carries several recommendation annotations", () => {
    const base = makeCatalogModel()
    if (base.servingState._tag !== "Assessed") throw new Error("fixture must be assessed")
    const model = {
      ...base,
      servingState: {
        ...base.servingState,
        recommendations: [
          makeRecommendation({ intent: "balanced" }),
          makeRecommendation({ intent: "fastest" }),
        ],
      },
    }
    const view = makeView({ models: [model], ready: false })
    const selections = buildLocalInferenceSelections(view.models, view.slots)
    expect(selections).toHaveLength(1)
    expect(selections[0]?.model).toBe(model)
    expect(selections[0]?.recommendation).toMatchObject({
      _tag: "Some",
      value: { intent: "balanced" },
    })
  })

  it("keeps active acquisition visible without manufacturing a download record", () => {
    const model = makeAcquiringModel({
      _tag: "Downloading",
      downloadId: ModelDownloadIdSchema.make("download"),
      stage: "downloading",
      completedBytes: GIB,
      totalBytes: 2 * GIB,
      bytesPerSecond: Option.none(),
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

  it("handles embedded and separate speculative bundle packages", () => {
    const target = makeStandaloneBundle("package_target")
    const draft = makeStandaloneBundle("package_draft")
    if (target._tag !== "Standalone" || draft._tag !== "Standalone") {
      throw new Error("test bundles must contain standalone packages")
    }
    const embedded = makeModel({
      bundle: {
        _tag: "SpeculativeDecoding",
        target: target.package,
        draftSource: { _tag: "Embedded" },
        method: { _tag: "Mtp" },
      },
    })
    const separate = makeModel({
      bundle: {
        _tag: "SpeculativeDecoding",
        target: target.package,
        draftSource: {
          _tag: "Separate",
          draft: {
            ...draft.package,
            properties: { ...draft.package.properties, maximumContextLength: 16_384 },
          },
        },
        method: { _tag: "DSpark" },
      },
    })

    expect(localModelBundleKey(embedded)).toContain("speculative:Mtp:Embedded:package_target")
    expect(localModelMaximumContextLength(embedded)).toBe(32_768)
    expect(localModelBundleKey(separate)).toContain(
      "speculative:DSpark:Separate:package_target:package_draft",
    )
    expect(localModelMaximumContextLength(separate)).toBe(16_384)
  })
})
