import { beforeEach, expect, test, vi } from "vitest"
import { Atom, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { testRender } from "@opentui/react/test-utils"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import type { LocalInferenceState, LocalModelRecommendation } from "@magnitudedev/client-common"
import { act } from "react"

const localInferenceActions = vi.hoisted(() => ({
  downloadModel: vi.fn(),
}))

const emptyLocalInferenceState = {
  activeBinding: null,
  host: {
    platform: "test",
    architecture: "test",
    topologyFingerprint: "test",
    systemMemoryBytes: 0,
    cpuModel: null,
    logicalCores: 1,
    memoryDomains: [],
    residentMemory: null,
  },
  choices: [],
  operations: [],
  recommendationState: { _tag: "Loading" },
  warnings: [],
} as const satisfies LocalInferenceState

const recommendedModel = {
  configurationId: "recommended-model",
  catalogModelId: "recommended-model-catalog",
  artifactFingerprint: "example/recommended-model:revision:content",
  modelId: Option.none(),
  badge: "recommended",
  displayName: "Recommended Model",
  family: "test",
  architecture: "dense",
  totalParametersBillions: Option.none(),
  activeParametersBillions: Option.none(),
  effectiveParametersBillions: Option.none(),
  quantization: {
    format: "UD-Q5_K_XL",
    quantAwareCheckpoint: false,
    fidelityLabel: "High fidelity",
    fidelityEvidence: "Catalog evidence.",
    fidelitySourceUrl: "https://example.invalid/model",
  },
  quantTag: "UD-Q5_K_XL",
  repo: "example/recommended-model",
  revision: "revision",
  files: [{
    path: "recommended-model.gguf",
    role: "weights",
    sizeBytes: 10_000,
    sha256: "sha256",
  }],
  totalDownloadBytes: 10_000,
  sourcePageUrl: "https://example.invalid/recommended-model",
  license: { id: "test", url: "https://example.invalid/license", acknowledgementRequired: false },
  contextTokens: 100_000,
  modelMaximumContextTokens: 200_000,
  estimatedRuntimeBytes: 12_000,
  stableCapacityBudgetBytes: 20_000,
  fitMarginBytes: 8_000,
  fitClass: "cpu_or_unified",
  constrainedContext: false,
  explanation: "Fits this machine.",
} as const satisfies LocalModelRecommendation

let localInferenceState: LocalInferenceState = emptyLocalInferenceState

const textPosition = (frame: string, label: string): { x: number; y: number } => {
  const lines = frame.split("\n")
  const y = lines.findIndex((line) => line.includes(label))
  if (y < 0) throw new Error(`Could not find ${label}`)
  return { x: lines[y]!.indexOf(label) + 1, y }
}

beforeEach(() => {
  localInferenceState = emptyLocalInferenceState
  localInferenceActions.downloadModel.mockClear()
})

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useSettingsState: () => ({
    keyAlreadySet: false,
    saveApiKey: () => {},
    saving: false,
    saveError: null,
  }),
  useAgentClient: () => ({
    query: () => Atom.make(Result.success(null)),
  }),
  useLocalInferenceState: () => ({
    state: Result.success(localInferenceState),
    mutationResults: [Result.initial()],
    pending: {
      download: false,
      activate: false,
      delete: false,
      restart: false,
      disable: false,
    },
    mutationFailure: Option.none(),
    downloadModel: localInferenceActions.downloadModel,
    activateModel: () => {},
    deleteModel: () => {},
    restart: () => {},
    disable: () => {},
  }),
}))
vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    success: "green",
  }),
}))

const { ModelSetupScreen } = await import("./screen")

test("local management renders local inference even when it is entered independently", async () => {
  const view = await testRender(
    <ModelSetupScreen
      mode="management"
      onExit={() => {}}
    />,
    { width: 80, height: 24 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("LOCAL MODEL SETUP")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("opens directly on hardware and model recommendations", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    recommendationState: { _tag: "Ready", recommendations: [] },
  }
  const view = await testRender(
    <ModelSetupScreen mode="management" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Choose what this machine should run")
    expect(frame).toContain("HARDWARE DETECTED")
    expect(frame).toContain("No curated model currently fits")
    expect(frame).not.toContain("How many local coding sessions")
    expect(frame).not.toContain("See recommendations")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("recommendation controls follow the model list instead of filling the terminal footer", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    recommendationState: { _tag: "Ready", recommendations: [] },
  }
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame.indexOf("No curated model")).toBeLessThan(frame.indexOf("↑/↓ choose"))
    expect(frame.indexOf("↑/↓ choose")).toBeLessThan(frame.indexOf("Skip for now"))
    expect(frame).not.toContain("Back (←)")
    expect(frame.split("\n")[textPosition(frame, "↑/↓ choose").y + 1]?.replaceAll("█", "").trim()).toBe("")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("clicking an already running model continues setup with that model", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    activeBinding: {
      selectionId: "running-model",
      providerModelId: ProviderModelIdSchema.make("running-provider-model"),
      contextTokens: 200_000,
    },
    choices: [{
      _tag: "Running",
      choiceId: "running-model",
      displayName: "Qwen3.6 35B-A3B",
      providerModelId: ProviderModelIdSchema.make("running-provider-model"),
      contextTokens: Option.some(200_000),
      fitClass: "cpu_or_unified",
      availability: { _tag: "Available" },
      fitAssessment: { _tag: "NotAssessed" },
      explanation: "Already running.",
      residency: "loaded",
      quantization: Option.some({
        format: "UD-Q6_K_XL",
        quantAwareCheckpoint: false,
        fidelityLabel: "Very high fidelity",
        fidelityEvidence: "Catalog evidence.",
        fidelitySourceUrl: "https://example.invalid/model",
      }),
      sizeBytes: Option.some(32_600_719_872),
    }],
  }
  const onComplete = vi.fn()
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} onComplete={onComplete} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const runningModel = textPosition(view.captureCharFrame(), "Qwen3.6 35B-A3B")
    await act(async () => view.mockMouse.moveTo(runningModel.x, runningModel.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(runningModel.x, runningModel.y))
    expect(onComplete).toHaveBeenCalledTimes(1)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("shows recommendation loading state without claiming that no model fits", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    recommendationState: { _tag: "Loading" },
  }
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Calculating recommendations for this machine…")
    expect(frame).not.toContain("No curated model currently fits")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("clicking a possible download starts that model download", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    recommendationState: { _tag: "Ready", recommendations: [recommendedModel] },
  }
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).not.toContain("Install runtime")
    const model = textPosition(view.captureCharFrame(), recommendedModel.displayName)
    await act(async () => view.mockMouse.moveTo(model.x, model.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(model.x, model.y))

    expect(localInferenceActions.downloadModel).toHaveBeenCalledWith(recommendedModel.configurationId)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("shows mirrored download progress without blocking navigation", async () => {
  const onComplete = vi.fn()
  localInferenceState = {
    ...emptyLocalInferenceState,
    recommendationState: { _tag: "Ready", recommendations: [recommendedModel] },
    operations: [{
      operationId: "operation-1",
      kind: "download",
      selectionId: recommendedModel.configurationId,
      providerModelId: ProviderModelIdSchema.make("recommended-model-catalog"),
      status: "running",
      stage: "downloading",
      progress: Option.some({ completedBytes: 2_500, totalBytes: 10_000 }),
      failure: Option.none(),
      startedAt: "2026-07-20T00:00:00.000Z",
      updatedAt: "2026-07-20T00:00:01.000Z",
    }],
  }
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} onComplete={onComplete} />,
    { width: 120, height: 34 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Downloading · downloading · 25%")

    const skip = textPosition(view.captureCharFrame(), "Skip for now (Esc)")
    await act(async () => view.mockMouse.click(skip.x, skip.y))
    await act(view.renderOnce)
    expect(onComplete).toHaveBeenCalledOnce()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("recommendations show a human-readable detected hardware panel before models", async () => {
  const gib = 1024 ** 3
  localInferenceState = {
    ...emptyLocalInferenceState,
    host: {
      platform: "macos",
      architecture: "aarch64",
      topologyFingerprint: "test",
      systemMemoryBytes: 64 * gib,
      cpuModel: "Apple M4 Max",
      logicalCores: 16,
      memoryDomains: [{
        id: "system",
        kind: "unified_memory",
        totalCapacityBytes: 64 * gib,
        stableCapacityBytes: 51.2 * gib,
        currentFreeBytes: null,
        sharesSystemMemory: true,
        backendNames: ["Metal"],
        deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
      residentMemory: null,
    },
    recommendationState: { _tag: "Ready", recommendations: [recommendedModel] },
  }
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 35 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("HARDWARE DETECTED")
    expect(frame).toContain("Apple M4 Max")
    expect(frame).toContain("macOS · Apple Silicon · 16 logical CPU cores")
    expect(frame).toContain("64.0 GiB unified memory · Metal GPU acceleration")
    expect(frame.indexOf("HARDWARE DETECTED")).toBeLessThan(frame.indexOf("RECOMMENDED DOWNLOADS"))
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("does not render the obsolete session-concurrency question", async () => {
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Choose what this machine should run")
    expect(frame).not.toContain("session")
    expect(frame).not.toContain("64K")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("skipping local onboarding completes without opening cloud setup", async () => {
  const onComplete = vi.fn()
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} onComplete={onComplete} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const localSkip = textPosition(view.captureCharFrame(), "Skip for now (Esc)")
    await act(async () => view.mockMouse.moveTo(localSkip.x, localSkip.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(localSkip.x, localSkip.y))
    expect(onComplete).toHaveBeenCalledTimes(1)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("Ctrl+C exits from local model setup", async () => {
  const onExit = vi.fn()
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={onExit} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    await act(async () => view.mockInput.pressCtrlC())
    expect(onExit).toHaveBeenCalledTimes(1)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
