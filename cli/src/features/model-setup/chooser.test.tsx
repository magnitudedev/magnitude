import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { DownloadAttemptIdSchema } from "@magnitudedev/sdk"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeModel,
  makeRecommendation,
  makeView,
} from "../local-inference/test-fixtures"
import {
  onboardingModelRowName,
  scrollOnboardingModelIntoView,
  type OnboardingModelChooserOperation,
} from "./chooser"
import { buildLocalInferenceSelections } from "../local-inference/view-model"

describe("onboarding model chooser identity", () => {
  it("keeps variants out of downloadable model names", () => {
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
    expect(onboardingModelRowName(selection)).toBe(model.presentation.displayName)
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
      attemptIds: [DownloadAttemptIdSchema.make("attempt")],
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
      onRetry: () => undefined,
    }
    expect(operation.model).toBe(model)
  })
})
