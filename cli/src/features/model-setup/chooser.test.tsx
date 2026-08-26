import { act, useState } from "react"
import { testRender } from "@opentui/react/test-utils"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeHardware,
  makeModel,
  makeView,
} from "../local-inference/test-fixtures"
import {
  ONBOARDING_MODEL_DETAIL_ROWS,
  onboardingLocalModelLayout,
  onboardingModelActionLabel,
  OnboardingModelChooser,
  onboardingModelDetailRows,
  onboardingSelectionEnterAction,
  onboardingModelRowName,
  scrollOnboardingModelIntoView,
  type OnboardingModelChooserOperation,
} from "./chooser"
import { buildLocalInferenceSelections } from "../local-inference/view-model"

const rankingControls = { fastToSmart: 0.5 }
const onRankingControlsChange = () => undefined

const makeRankedCatalogModel = (
  id: string,
  displayName: string,
  rankingScores: { intelligence: number; speed: number; quality: number },
) => {
  const model = makeCatalogModel()
  if (model.servingState._tag !== "Assessed") throw new Error("catalog fixture must be assessed")
  return {
    ...model,
    modelId: ProviderModelIdSchema.make(id),
    presentation: { ...model.presentation, displayName },
    servingState: { ...model.servingState, rankingScores: Option.some(rankingScores) },
  }
}

describe("onboarding model chooser identity", () => {
  it("renders a non-focusable scale and adjusts it from any model selection", async () => {
    const updates: typeof rankingControls[] = []
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:test", kind: "downloadable", model: makeCatalogModel() }]}
        rankingControls={rankingControls}
        onRankingControlsChange={(controls) => { updates.push(controls) }}
        width={120}
        error={null}
        operation={null}
        onSelect={() => undefined}
        onExit={() => undefined}
        exitKind="Skip"
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Fastest   Faster   Balanced   Smarter  Smartest")
      expect(frame).toContain("┤    ←/→ change preference")
      const fastTrack = frame.split("\n")
        .find((line) => line.includes("┼"))
        ?.match(/[├┤┼─]+/)?.[0]
      expect(fastTrack).not.toContain("◆")
      expect(fastTrack?.match(/[├┤┼]/g)).toHaveLength(5)
      expect([...fastTrack ?? ""].flatMap((character, index) =>
        /[├┤┼]/.test(character) ? [index] : []))
        .toEqual([0, 10, 20, 30, 40])
      await act(async () => view.mockInput.pressArrow("left"))
      expect(updates.at(-1)?.fastToSmart).toBeCloseTo(0.25)
      await act(async () => view.mockInput.pressArrow("right"))
      expect(updates.at(-1)?.fastToSmart).toBeCloseTo(0.75)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps the cursor on the same rank when preference reorders models", async () => {
    const smart = makeRankedCatalogModel(
      "a-smart",
      "Smart Model",
      { intelligence: 1, speed: 0.4, quality: 1 },
    )
    const fast = makeRankedCatalogModel(
      "b-fast",
      "Fast Model",
      { intelligence: 0.4, speed: 1, quality: 1 },
    )
    const StatefulChooser = () => {
      const [controls, setControls] = useState(rankingControls)
      return (
        <OnboardingModelChooser
          hardware={Result.success(makeHardware())}
          options={[
            { id: "downloadable:smart", kind: "downloadable", model: smart },
            { id: "downloadable:fast", kind: "downloadable", model: fast },
          ]}
          rankingControls={controls}
          onRankingControlsChange={setControls}
          width={120}
          error={null}
          operation={null}
          onSelect={() => undefined}
          onExit={() => undefined}
          exitKind="Skip"
        />
      )
    }
    const view = await testRender(<StatefulChooser />, { width: 120, height: 44 })

    try {
      await act(view.renderOnce)
      await act(async () => view.mockInput.pressArrow("down"))
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toMatch(/› 2\. Fast Model/)
      await act(async () => view.mockInput.pressArrow("left"))
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toMatch(/› 2\. Smart Model/)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("renders the empty choice state without dereferencing a selection", async () => {
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
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
      const frame = view.captureCharFrame()
      expect(frame).toContain("No compatible models found.")
      expect(frame).not.toContain("Choose a local model")
      const lines = frame.split("\n")
      const setupColumn = lines.find((line) => line.includes("MAGNITUDE SETUP"))?.indexOf("MAGNITUDE SETUP")
      const contentColumn = lines.find((line) => line.includes("Test CPU"))?.indexOf("Test CPU")
      expect(setupColumn).toBeGreaterThan(0)
      expect(contentColumn).toBe(setupColumn)
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
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
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

  it("replaces the preference scale with model loading progress", async () => {
    let cancellations = 0
    const state = makeView({ models: [makeModel()], ready: false })
    const options = buildLocalInferenceSelections(state.models, state.slots)
    const model = state.models.models[0]!
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(state.hardware)}
        options={options}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={{
          _tag: "Activating",
          providerModelId: model.modelId,
          model,
          status: { _tag: "Loading", stage: "loading", progress: Option.none() },
          onCancel: () => { cancellations += 1 },
          onRetry: () => undefined,
          onChooseAnother: () => undefined,
        }}
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
      expect(frame).toContain("INTELLIGENCE")
      expect(frame).toContain("Loading model into memory…")
      expect(frame).toContain("0%")
      expect(frame).toContain("Cancel (Esc)")
      expect(frame).not.toContain("←/→ change preference")
      expect(frame.indexOf("Loading model into memory…")).toBeLessThan(frame.indexOf("ON THIS COMPUTER"))
      expect(frame.indexOf("Loading model into memory…")).toBeLessThan(frame.indexOf("INTELLIGENCE"))
      const lines = frame.split("\n")
      const setupColumn = lines.find((line) => line.includes("MAGNITUDE SETUP"))
        ?.indexOf("MAGNITUDE SETUP")
      const progressLine = lines.find((line) => line.includes("0%"))
      expect(setupColumn).toBeDefined()
      expect(progressLine).toBeDefined()
      expect(progressLine?.trimEnd().length).toBe((setupColumn ?? 0) + 97)
      await act(async () => {
        view.mockInput.pressEscape()
        await new Promise((resolve) => setTimeout(resolve, 25))
      })
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("Cancel loading?  › Yes")
      await act(async () => view.mockInput.pressEnter())
      expect(cancellations).toBe(1)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps download measurements on the status line in the three-row operation region", async () => {
    const model = makeAcquiringModel({
      _tag: "Installing",
      progress: {
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      },
    })
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:test", kind: "downloadable", model }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={{
          _tag: "Downloading",
          model,
          starting: false,
          cancelling: false,
          onCancel: () => undefined,
        }}
        onSelect={() => undefined}
        onExit={() => undefined}
        exitKind="Skip"
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Downloading · 1 B / 2 B · Estimating…")
      expect(frame).toContain("Cancel (Esc)")
      expect(frame).not.toContain("Estimating time remaining")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("shows catalog release recency before compact model facts", async () => {
    const model = makeCatalogModel()
    const state = makeView({ models: [model], ready: false })
    const options = buildLocalInferenceSelections(state.models, state.slots)
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(state.hardware)}
        options={options}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
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
      const frame = view.captureCharFrame()
      expect(frame).toContain("1. Qwen Test (Q4)")
      expect(frame).toMatch(/\d+ days ago · Dense \(8B\) · Text only/)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("shows a ranked installed model with Load and keeps it under On This Computer", async () => {
    const model = makeCatalogModel()
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "installed:test", kind: "stored", model }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={null}
        onSelect={() => undefined}
        onExit={() => undefined}
        exitKind="Skip"
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toMatch(/1\. Qwen Test \(Q4\).*Load/)
      const installedSection = frame.slice(frame.indexOf("ON THIS COMPUTER"))
      expect(installedSection).toMatch(/Qwen Test \(Q4\).*Load/)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("shows exact hidden-model counts above and below an overflowing local list", async () => {
    const options = Array.from({ length: 20 }, (_, index) => {
      const base = makeModel({ modelId: ProviderModelIdSchema.make(`local-${index + 1}`) })
      return {
        id: `installed:${index + 1}`,
        kind: "stored" as const,
        model: {
          ...base,
          presentation: { ...base.presentation, displayName: `Local Model ${index + 1}` },
        },
      }
    })
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={options}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={null}
        onSelect={() => undefined}
        onExit={() => undefined}
        exitKind="Skip"
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const initialFrame = view.captureCharFrame()
      expect(initialFrame).toContain("… and 4 more")
      const initialLines = initialFrame.split("\n")
      const firstModelRow = initialLines.find((line) =>
        line.includes("Local Model 1 (Q4)") && line.includes("Load"))
      expect(initialLines.find((line) => line.includes("… and 4 more"))?.indexOf("…"))
        .toBe(firstModelRow?.indexOf("Local Model 1"))
      expect(initialLines.findIndex((line) =>
        line.includes("Local Model 1 (Q4)") && line.includes("Load")))
        .toBe(initialLines.findIndex((line) => line.includes("ON THIS COMPUTER")) + 1)
      for (let index = 0; index < 16; index += 1) {
        await act(async () => view.mockInput.pressArrow("down"))
      }
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("… and 2 more")
      expect(frame).toContain("… and 3 more")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("describes the Enter action from the selected model state", () => {
    expect(onboardingSelectionEnterAction("downloadable")).toBe("download")
    expect(onboardingSelectionEnterAction("stored")).toBe("load")
    expect(onboardingSelectionEnterAction("running")).toBe("select")
    expect(onboardingSelectionEnterAction(undefined)).toBeNull()
  })

  it("derives detail height from explicit row regions", () => {
    expect(onboardingModelDetailRows({
      memoryWarning: false,
      modelSummaryRadarGap: true,
    })).toBe(18)
    expect(onboardingModelDetailRows({
      memoryWarning: true,
      modelSummaryRadarGap: true,
    })).toBe(19)
    expect(ONBOARDING_MODEL_DETAIL_ROWS).toBe(18)
  })

  it("uses overflow indicator rows inside the local-model row budget", () => {
    expect(onboardingLocalModelLayout({
      wide: true,
      localCount: 12,
      detailPanelRows: ONBOARDING_MODEL_DETAIL_ROWS,
      rankedRows: 5,
      sectionGap: 1,
    })).toEqual({ viewportRows: 11, showOverflow: true })
    expect(onboardingLocalModelLayout({
      wide: false,
      localCount: 12,
      detailPanelRows: ONBOARDING_MODEL_DETAIL_ROWS,
      rankedRows: 5,
      sectionGap: 1,
    })).toEqual({ viewportRows: 4, showOverflow: true })
  })

  it("keeps variants in downloadable model names", () => {
    const model = makeCatalogModel()
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

  it("shows the generic load label for an installed model", () => {
    const installed = makeModel()
    const view = makeView({ models: [installed], ready: false })
    const selections = buildLocalInferenceSelections(view.models, view.slots)
    const installedSelection = selections.find(({ kind }) => kind === "stored")

    expect(selections).toHaveLength(1)
    expect(installedSelection && onboardingModelActionLabel(installedSelection))
      .toBe("Load")
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
      _tag: "Installing",
      progress: {
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      },
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
