import { act, useState } from "react"
import { testRender } from "@opentui/react/test-utils"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/sdk"
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
  onboardingModelOperationHint,
  onboardingModelRowEmphasis,
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
  rankingScores: { intelligence: number; speed: number; fidelity: number },
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
  it("distinguishes keyboard selection from the locked operation subject", () => {
    expect(onboardingModelRowEmphasis({
      selected: true,
      operationSubject: false,
      disabled: false,
    })).toBe("selected")
    expect(onboardingModelRowEmphasis({
      selected: false,
      operationSubject: true,
      disabled: true,
    })).toBe("subject")
    expect(onboardingModelRowEmphasis({
      selected: false,
      operationSubject: false,
      disabled: true,
    })).toBe("muted")
  })

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
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Fastest   Faster   Balanced   Smarter  Smartest")
      expect(frame).not.toContain("┤    ←/→ change preference")
      expect(frame).toContain("←/→ change preferences · ↑/↓ choose models · Enter to download · Ctrl+C to exit")
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

  it("selects a model when its row is clicked", async () => {
    const selected: ProviderModelId[] = []
    const model = makeCatalogModel()
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:test", kind: "downloadable", model }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={null}
        onSelect={(modelId) => { selected.push(modelId) }}
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const lines = view.captureCharFrame().split("\n")
      const y = lines.findIndex((line) => line.includes("1. Qwen Test (Q4)"))
      const x = lines[y]?.indexOf("Qwen Test (Q4)") ?? -1
      expect(x).toBeGreaterThanOrEqual(0)
      expect(y).toBeGreaterThanOrEqual(0)
      await act(async () => view.mockMouse.click(x, y))
      expect(selected).toEqual([model.modelId])
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("uses softened weights at the Fastest and Smartest endpoints", async () => {
    const updates: typeof rankingControls[] = []
    const StatefulChooser = () => {
      const [controls, setControls] = useState(rankingControls)
      return (
        <OnboardingModelChooser
          hardware={Result.success(makeHardware())}
          options={[{ id: "downloadable:test", kind: "downloadable", model: makeCatalogModel() }]}
          rankingControls={controls}
          onRankingControlsChange={(next) => {
            updates.push(next)
            setControls(next)
          }}
          width={120}
          error={null}
          operation={null}
          onSelect={() => undefined}
        />
      )
    }
    const view = await testRender(<StatefulChooser />, { width: 120, height: 44 })

    try {
      await act(view.renderOnce)
      await act(async () => view.mockInput.pressArrow("left"))
      await act(view.renderOnce)
      await act(async () => view.mockInput.pressArrow("left"))
      expect(updates.at(-1)?.fastToSmart).toBeCloseTo(0.05)

      for (let index = 0; index < 4; index += 1) {
        await act(async () => view.mockInput.pressArrow("right"))
        await act(view.renderOnce)
      }
      expect(updates.at(-1)?.fastToSmart).toBeCloseTo(0.95)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps narrow setup regions distinct with full ranked and installed sections", async () => {
    const baseHardware = makeHardware()
    const hardware = makeHardware({
      platform: "MacOS",
      architecture: "Arm64",
      processor: Option.some("Apple M4 Max"),
      accelerators: baseHardware.accelerators.map((accelerator) => ({
        ...accelerator,
        name: "Apple M4 Max",
        backend: "Metal",
      })),
      memoryDomains: baseHardware.memoryDomains.map((domain) => ({
        ...domain,
        kind: "UnifiedMemory",
        sharesSystemMemory: true,
      })),
    })
    const options = Array.from({ length: 14 }, (_, index) => ({
      id: `${index < 10 ? "downloadable" : "installed"}:${index}`,
      kind: index < 10 ? "downloadable" as const : "stored" as const,
      model: makeRankedCatalogModel(
        `narrow-${index}`,
        `Narrow Model ${index}`,
        { intelligence: 0.7, speed: 0.7, fidelity: 0.7 },
      ),
    }))
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(hardware)}
        options={options}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={60}
        error={null}
        operation={null}
        onSelect={() => undefined}
      />,
      { width: 60, height: 70 },
    )

    try {
      await act(view.renderOnce)
      const lines = view.captureCharFrame().split("\n").map((line) => line.trim())
      expect(lines).toContain("MAGNITUDE SETUP")
      expect(lines).toContain("● Choose model")
      expect(lines).toContain("○ Install model")
      expect(lines).toContain("○ Select harness")
      expect(lines).not.toContain("←/→ change preference")
      expect(lines.some((line) => line.startsWith("You can switch models"))).toBe(false)
      expect(lines).toContain("←→ prefs · ↑↓ models · Enter download · Ctrl+C to exit")
      expect(lines.find((line) => line.includes("Fastest"))).not.toContain("─")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps setup steps horizontal when the chooser stacks but the steps still fit", async () => {
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:test", kind: "downloadable", model: makeCatalogModel() }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={80}
        error={null}
        operation={null}
        onSelect={() => undefined}
      />,
      { width: 80, height: 50 },
    )

    try {
      await act(view.renderOnce)
      const stepperLine = view.captureCharFrame().split("\n")
        .find((line) => line.includes("Choose model"))
      expect(stepperLine).toContain("Install model")
      expect(stepperLine).toContain("Select harness")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("scrolls stacked model details into view in a short terminal", async () => {
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:test", kind: "downloadable", model: makeCatalogModel() }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={60}
        error={null}
        operation={null}
        onSelect={() => undefined}
      />,
      { width: 60, height: 20 },
    )

    try {
      await act(view.renderOnce)
      expect(view.captureCharFrame()).not.toContain("INTELLIGENCE")
      await act(async () => {
        for (let step = 0; step < 12; step += 1) {
          await view.mockMouse.scroll(55, 9, "down", { delayMs: 0 })
        }
      })
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("INTELLIGENCE")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps the cursor on the same rank when preference reorders models", async () => {
    const smart = makeRankedCatalogModel(
      "a-smart",
      "Smart Model",
      { intelligence: 1, speed: 0.4, fidelity: 1 },
    )
    const fast = makeRankedCatalogModel(
      "b-fast",
      "Fast Model",
      { intelligence: 0.4, speed: 1, fidelity: 1 },
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
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Qwen Test (Q4)")
      expect(frame).toContain("INTELLIGENCE")
      expect(frame).toContain("Loading Qwen Test (Q4) into memory…")
      expect(frame).toContain("0%")
      expect(frame).not.toContain("Cancel (Esc)")
      expect(frame).toContain("Esc cancel · Ctrl+C to exit")
      expect(frame).not.toContain("› ")
      expect(frame).not.toContain("←/→ change preference")
      expect(frame.indexOf("Loading Qwen Test (Q4) into memory…")).toBeLessThan(frame.indexOf("ON THIS COMPUTER"))
      expect(frame.indexOf("Loading Qwen Test (Q4) into memory…")).toBeLessThan(frame.indexOf("INTELLIGENCE"))
      const lines = frame.split("\n")
      const setupColumn = lines.find((line) => line.includes("MAGNITUDE SETUP"))
        ?.indexOf("MAGNITUDE SETUP")
      const progressLine = lines.find((line) => line.includes("0%"))
      const progressLineIndex = lines.findIndex((line) => line.includes("0%"))
      expect(setupColumn).toBeDefined()
      expect(progressLine).toBeDefined()
      expect(progressLine?.trimEnd().length).toBe((setupColumn ?? 0) + 97)
      expect(lines[progressLineIndex + 1]?.trim()).toBe("")
      await act(async () => {
        view.mockInput.pressEscape()
        await new Promise((resolve) => setTimeout(resolve, 25))
      })
      await act(view.renderOnce)
      const confirmationFrame = view.captureCharFrame()
      expect(confirmationFrame).toContain("Cancel loading?  › Yes")
      expect(confirmationFrame.indexOf("Cancel loading?"))
        .toBeGreaterThan(confirmationFrame.indexOf("ON THIS COMPUTER"))
      await act(async () => view.mockInput.pressArrow("right"))
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("Cancel loading?    Yes  › No")
      await act(async () => view.mockInput.pressArrow("left"))
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("Cancel loading?  › Yes")
      await act(async () => view.mockInput.pressEnter())
      expect(cancellations).toBe(1)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("renders low memory as a failed load with a red-bar region and recovery controls", async () => {
    const model = makeModel()
    let retries = 0
    let chooseAnother = 0
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "installed:test", kind: "stored", model }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={{
          _tag: "Activating",
          providerModelId: model.modelId,
          model,
          status: {
            _tag: "Failed",
            failure: {
              _tag: "LowMemory",
              code: "low_memory",
              message: "not enough memory available",
              retryable: true,
              requiredSystemMemoryBytes: 8_000_000_000,
              allocationHeadroomBytes: 6_000_000_000,
              systemReserveBytes: 1_000_000_000,
              loadBoundaryBytes: 9_000_000_000,
              minimumAdditionalAvailableBytes: 3_000_000_000,
              parallelSequences: 1,
            },
          },
          onCancel: () => undefined,
          onRetry: () => { retries += 1 },
          onChooseAnother: () => { chooseAnother += 1 },
        }}
        onSelect={() => undefined}
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Not enough memory for Qwen Test (Q4) · Free at least 2.8 GB")
      expect(frame).toContain("█".repeat(8))
      expect(frame).not.toContain("0%")
      expect(frame).toContain("› Retry loading")
      expect(frame).toContain("Choose another model")
      expect(frame.indexOf("› Retry loading")).toBeGreaterThan(frame.indexOf("ON THIS COMPUTER"))
      expect(frame).not.toContain("←/→ choose action · Enter confirm · Ctrl+C to exit")
      expect(frame).not.toContain("Unable to load model")
      await act(async () => {
        view.mockInput.pressEscape()
        await new Promise((resolve) => setTimeout(resolve, 25))
      })
      expect(chooseAnother).toBe(0)
      await act(async () => view.mockInput.pressArrow("right"))
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("› Choose another model")
      await act(async () => view.mockInput.pressEnter())
      expect(chooseAnother).toBe(1)
      await act(async () => view.mockInput.pressArrow("left"))
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("› Retry loading")
      await act(async () => view.mockInput.pressEnter())
      expect(retries).toBe(1)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("places unexpected setup errors below the combined footer hint", async () => {
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:test", kind: "downloadable", model: makeCatalogModel() }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error="Unexpected error loading Qwen Test (Q4) · Worker exited before initialization."
        operation={null}
        onSelect={() => undefined}
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame.indexOf("Unexpected error loading Qwen Test (Q4)"))
        .toBeGreaterThan(frame.indexOf("←/→ change preferences · ↑/↓ choose models · Enter to download · Ctrl+C to exit"))
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("keeps download measurements on the status line in the three-row operation region", async () => {
    const model = makeAcquiringModel({
      _tag: "Installing",
      progress: {
        stage: "downloading",
        completedBytes: 1_800_000_000,
        totalBytes: 20_000_000_000,
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
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Downloading · 1.80 GB / 20.00 GB · Estimating…")
      expect(frame).not.toContain("Cancel (Esc)")
      expect(frame).toContain("Esc cancel · Ctrl+C to exit")
      expect(frame.indexOf("Esc cancel · Ctrl+C to exit"))
        .toBeGreaterThan(frame.indexOf("INTELLIGENCE"))
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

  it("omits Download from downloadable rows and gives the space to the model name", async () => {
    const model = makeRankedCatalogModel(
      "long-downloadable",
      "A Downloadable Model Name That Uses The Reclaimed Space",
      { intelligence: 0.7, speed: 0.7, fidelity: 0.7 },
    )
    const view = await testRender(
      <OnboardingModelChooser
        hardware={Result.success(makeHardware())}
        options={[{ id: "downloadable:long", kind: "downloadable", model }]}
        rankingControls={rankingControls}
        onRankingControlsChange={onRankingControlsChange}
        width={120}
        error={null}
        operation={null}
        onSelect={() => undefined}
      />,
      { width: 120, height: 44 },
    )

    try {
      await act(view.renderOnce)
      const modelLine = view.captureCharFrame().split("\n")
        .find((line) => line.includes("A Downloadable Model Name"))
      expect(modelLine).toContain("A Downloadable Model Name That")
      expect(modelLine).not.toMatch(/\sDownload\s*$/)
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

  it("omits an action label for a downloadable model", () => {
    const model = makeCatalogModel()
    const selection = { id: "downloadable:test", kind: "downloadable", model } as const

    expect(onboardingModelActionLabel(selection)).toBeNull()
  })

  it("shows update as the model-level action when an installed model is stale", () => {
    const installed = makeModel()
    if (installed.acquisitionState._tag !== "Installed") {
      throw new Error("installed fixture must be installed")
    }
    const update = {
      ...installed,
      acquisitionState: { ...installed.acquisitionState, _tag: "UpdateAvailable" as const },
    }
    const view = makeView({ models: [update], ready: false })
    const [selection] = buildLocalInferenceSelections(view.models, view.slots)

    expect(selection && onboardingModelActionLabel(selection)).toBe("Update")
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
    expect(onboardingModelOperationHint(operation)).toBe("Esc cancel · Ctrl+C to exit")
  })
})
