import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { ModelDownloadIdSchema } from "@magnitudedev/sdk"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeModel,
  makeRecommendation,
  makeView,
} from "../local-inference/test-fixtures"
import {
  ONBOARDING_MODEL_DETAIL_ROWS,
  onboardingLocalModelViewportRows,
  onboardingModelDetailRows,
  onboardingSelectionEnterAction,
  onboardingModelRowName,
  scrollOnboardingModelIntoView,
  type OnboardingModelChooserOperation,
} from "./chooser"
import { buildLocalInferenceSelections } from "../local-inference/view-model"

describe("onboarding model chooser identity", () => {
  it("describes the Enter action from the selected model state", () => {
    expect(onboardingSelectionEnterAction("recommendation")).toBe("download")
    expect(onboardingSelectionEnterAction("stored")).toBe("load")
    expect(onboardingSelectionEnterAction("running")).toBe("select")
    expect(onboardingSelectionEnterAction(undefined)).toBeNull()
  })

  it("derives detail height from explicit row regions", () => {
    expect(onboardingModelDetailRows({
      recommendation: false,
      memoryWarning: false,
      statusRows: 2,
    })).toBe(22)
    expect(onboardingModelDetailRows({
      recommendation: false,
      memoryWarning: true,
      statusRows: 5,
    })).toBe(26)
    expect(ONBOARDING_MODEL_DETAIL_ROWS).toBe(26)
  })

  it("lets local models fill the remaining wide-layout rows", () => {
    expect(onboardingLocalModelViewportRows({
      wide: true,
      localCount: 12,
      detailPanelRows: ONBOARDING_MODEL_DETAIL_ROWS,
      downloadRows: 5,
      sectionGap: 1,
    })).toBe(19)
    expect(onboardingLocalModelViewportRows({
      wide: false,
      localCount: 12,
      detailPanelRows: ONBOARDING_MODEL_DETAIL_ROWS,
      downloadRows: 5,
      sectionGap: 1,
    })).toBe(4)
  })

  it("keeps variants in downloadable model names", () => {
    const base = makeCatalogModel()
    if (base.servingState._tag !== "Assessed") throw new Error("fixture must be assessed")
    const model = {
      ...base,
      servingState: {
        ...base.servingState,
        recommendations: [makeRecommendation()],
      },
    }
    const view = makeView({ models: [model], ready: false })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)

    expect(selection).toBeDefined()
    if (!selection) return
    expect(onboardingModelRowName(selection)).toBe("Qwen Test (Q4)")
  })

  it("keeps variants in installed model names", () => {
    const model = makeModel()
    const view = makeView({ models: [model] })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)

    expect(selection).toBeDefined()
    if (!selection) return
    expect(onboardingModelRowName(selection)).toBe("Qwen Test (Q4)")
  })

  it("scrolls by presentation identity without copying model fields", () => {
    const calls: string[] = []
    scrollOnboardingModelIntoView({
      scrollChildIntoView: (id: string) => { calls.push(id) },
    } as never, "model-id")
    expect(calls).toEqual(["onboarding-model:model-id"])
  })

  it("does nothing before the model viewport is mounted", () => {
    expect(() => scrollOnboardingModelIntoView(null, "model-id")).not.toThrow()
  })

  it("carries the canonical model through acquisition operations", () => {
    const model = makeAcquiringModel({
      _tag: "Downloading",
      downloadId: ModelDownloadIdSchema.make("download"),
      stage: "downloading",
      completedBytes: 1,
      totalBytes: 2,
      bytesPerSecond: Option.none(),
    })
    const operation: OnboardingModelChooserOperation = {
      _tag: "Downloading",
      model,
      starting: false,
      cancelling: false,
      cancelError: null,
      onCancel: () => undefined,
    }
    expect(operation._tag === "Downloading" && operation.model).toBe(model)
  })
})
