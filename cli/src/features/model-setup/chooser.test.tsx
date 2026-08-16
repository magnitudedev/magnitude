import { act } from "react"
import { testRender } from "@opentui/react/test-utils"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { ModelDownloadIdSchema } from "@magnitudedev/sdk"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeHardware,
  makeModel,
  makeRecommendation,
  makeView,
} from "../local-inference/test-fixtures"
import {
  ONBOARDING_MODEL_DETAIL_ROWS,
  onboardingLocalModelViewportRows,
  onboardingModelActionLabel,
  OnboardingModelChooser,
  onboardingModelDetailRows,
  onboardingSelectionEnterAction,
  onboardingModelRowName,
  scrollOnboardingModelIntoView,
  type OnboardingModelChooserOperation,
} from "./chooser"
import { buildLocalInferenceSelections } from "../local-inference/view-model"

describe("onboarding model chooser identity", () => {
  it("renders the empty choice state without dereferencing a selection", async () => {
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[]}
        width={120}
        error={null}
        operation={null}
        onSelect={() => undefined}
        onExit={() => undefined}
        exitKind="Skip"
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("No compatible models found.")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps exact active-operation detail visible when discovery options are empty", async () => {
    const model = makeModel()
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[]}
        width={120}
        error={null}
        operation={{ _tag: "Configuring", model }}
        onSelect={() => undefined}
        onExit={() => undefined}
        exitKind="Skip"
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Qwen Test (Q4)")
      expect(frame).toContain("Configuring model…")
      expect(frame).not.toContain("No compatible models found.")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

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
      downloadRows: 0,
      modelSummaryRadarGap: true,
    })).toBe(18)
    expect(onboardingModelDetailRows({
      recommendation: false,
      memoryWarning: false,
      downloadRows: 4,
      modelSummaryRadarGap: true,
    })).toBe(22)
    expect(ONBOARDING_MODEL_DETAIL_ROWS).toBe(22)
  })

  it("lets local models fill the remaining wide-layout rows", () => {
    expect(onboardingLocalModelViewportRows({
      wide: true,
      localCount: 12,
      detailPanelRows: ONBOARDING_MODEL_DETAIL_ROWS,
      downloadRows: 5,
      sectionGap: 1,
    })).toBe(15)
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
        recommendations: [
          makeRecommendation(),
          makeRecommendation({ intent: "fastest" }),
        ],
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
    const view = makeView({ models: [model], ready: false })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)

    expect(selection).toBeDefined()
    if (!selection) return
    expect(onboardingModelRowName(selection)).toBe("Qwen Test (Q4)")
  })

  it("shows an installed recommendation instead of the generic load label", () => {
    const installed = makeModel()
    const catalog = makeCatalogModel()
    if (catalog.servingState._tag !== "Assessed") throw new Error("fixture must be assessed")
    const recommendedCatalogModel = {
      ...catalog,
      servingState: {
        ...catalog.servingState,
        recommendations: [
          makeRecommendation(),
          makeRecommendation({ intent: "fastest" }),
        ],
      },
    }
    const view = makeView({ models: [installed, recommendedCatalogModel], ready: false })
    const selections = buildLocalInferenceSelections(view.models, view.slots)
    const installedSelection = selections.find(({ kind }) => kind === "stored")

    expect(selections).toHaveLength(1)
    expect(installedSelection && onboardingModelActionLabel(installedSelection))
      .toBe("Balanced / Fastest")
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
      onCancel: () => undefined,
    }
    expect(operation._tag === "Downloading" && operation.model).toBe(model)
  })
})
