import { beforeEach, expect, test, vi } from "vitest"
import { Atom, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { testRender } from "@opentui/react/test-utils"
import { ProviderModelIdSchema, type LocalInferenceState, type LocalModelRecommendation } from "@magnitudedev/sdk"
import { act } from "react"

const localInferenceActions = vi.hoisted(() => ({
  downloadModel: vi.fn(),
}))

const emptyLocalInferenceState = {
  usage: null,
  activeBinding: null,
  llamaCpp: {
    minimumBuild: 8868,
    recommendedBuild: 10011,
    installations: [],
    selectedInstallationId: Option.none(),
    activeManagedInstallationId: Option.none(),
    managedInstall: { availability: { _tag: "Available", build: 10011 }, operation: { _tag: "Idle" } },
    diagnostics: [],
  },
  host: { _tag: "Unavailable", message: "not needed" },
  choices: [],
  operations: [],
  recommendations: [],
  warnings: [],
} as const satisfies LocalInferenceState

const recommendedModel = {
  configurationId: "recommended-model",
  catalogModelId: "recommended-model-catalog",
  badge: "recommended",
  displayName: "Recommended Model",
  family: "test",
  architecture: "dense",
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
    sizeBytes: 10_000,
    sha256: "sha256",
    downloadUrl: "https://example.invalid/recommended-model.gguf",
  }],
  totalDownloadBytes: 10_000,
  sourcePageUrl: "https://example.invalid/recommended-model",
  license: { id: "test", url: "https://example.invalid/license", acknowledgementRequired: false },
  contextTokens: 64_000,
  servingProfile: {
    sessionConcurrency: "one",
    parallelSlots: 1,
    contextTokensPerSlot: 64_000,
    totalContextCapacityTokens: 64_000,
    slotAllocation: "uniform",
    runtimeProfileId: "test",
  },
  modelMaximumContextTokens: 64_000,
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
    mutationBusy: false,
    mutationFailure: Option.none(),
    configureUsage: () => {},
    installLlamaCpp: () => {},
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

const { ModelSetupScreen, resolveModelSetupSurface } = await import("./screen")

test("local management renders local inference even when it is entered independently", async () => {
  const view = await testRender(
    <ModelSetupScreen
      initialStep="local"
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

test("management preserves an explicitly selected cloud surface", () => {
  expect(resolveModelSetupSurface("management", "cloud")).toBe("cloud-provider")
})

test("saved usage answers remain preselected without skipping the questions", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    usage: { sessionConcurrency: "one" },
  }
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="management" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("How many local coding sessions will you run at once?")
    expect(frame).toContain("● One session")
    expect(frame).not.toContain("┌")
    expect(frame.indexOf("How many Magnitude sessions")).toBeLessThan(frame.indexOf("↑/↓ move"))
    expect(frame.indexOf("↑/↓ move")).toBeLessThan(frame.indexOf("Skip for now"))
    expect(frame.indexOf("Skip for now")).toBeLessThan(frame.indexOf("See recommendations"))
    expect(frame.split("\n")[textPosition(frame, "↑/↓ move").y + 1]?.trim()).toBe("")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("recommendation controls follow the model list instead of filling the terminal footer", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    usage: { sessionConcurrency: "one" },
  }
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const recommendations = textPosition(view.captureCharFrame(), "See recommendations")
    await act(async () => view.mockMouse.moveTo(recommendations.x, recommendations.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(recommendations.x, recommendations.y))
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame.indexOf("No curated model")).toBeLessThan(frame.indexOf("↑/↓ choose"))
    expect(frame.indexOf("↑/↓ choose")).toBeLessThan(frame.indexOf("Back (←)"))
    expect(frame.indexOf("Back (←)")).toBeLessThan(frame.indexOf("Skip for now"))
    expect(frame.split("\n")[textPosition(frame, "↑/↓ choose").y + 1]?.replaceAll("█", "").trim()).toBe("")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("clicking an already running model continues setup with that model", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    usage: { sessionConcurrency: "one" },
    llamaCpp: {
      ...emptyLocalInferenceState.llamaCpp,
      installations: [{ id: "managed", executables: { serverPath: "/managed/llama-server", fitParamsPath: "/managed/llama-fit-params" }, build: 10011, ownership: "magnitude", discoveries: [] }],
      selectedInstallationId: Option.some("managed"),
    },
    activeBinding: {
      _tag: "External",
      selectionId: "running-model",
      providerModelId: ProviderModelIdSchema.make("running-provider-model"),
      contextTokens: 200_000,
    },
    choices: [{
      _tag: "RunningExternal",
      choiceId: "running-model",
      displayName: "Qwen3.6 35B-A3B",
      providerModelId: ProviderModelIdSchema.make("running-provider-model"),
      contextTokens: 200_000,
      fitClass: "cpu_or_unified",
      availability: { _tag: "Available" },
      fitAssessment: { _tag: "NotAssessed" },
      explanation: "Already running.",
      residency: "loaded",
      quantization: {
        format: "UD-Q6_K_XL",
        quantAwareCheckpoint: false,
        fidelityLabel: "Very high fidelity",
        fidelityEvidence: "Catalog evidence.",
        fidelitySourceUrl: "https://example.invalid/model",
      },
      sizeBytes: 32_600_719_872,
    }],
  }
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const recommendations = textPosition(view.captureCharFrame(), "See recommendations")
    await act(async () => view.mockMouse.click(recommendations.x, recommendations.y))
    await act(view.renderOnce)

    const runningModel = textPosition(view.captureCharFrame(), "Qwen3.6 35B-A3B")
    await act(async () => view.mockMouse.moveTo(runningModel.x, runningModel.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(runningModel.x, runningModel.y))
    await act(view.renderOnce)

    expect(view.captureCharFrame()).toContain("CLOUD MODELS (OPTIONAL)")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("clicking a possible download starts that model download", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    usage: { sessionConcurrency: "one" },
    llamaCpp: {
      ...emptyLocalInferenceState.llamaCpp,
      installations: [{ id: "managed", executables: { serverPath: "/managed/llama-server", fitParamsPath: "/managed/llama-fit-params" }, build: 10011, ownership: "magnitude", discoveries: [] }],
      selectedInstallationId: Option.some("managed"),
    },
    recommendations: [recommendedModel],
  }
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const recommendations = textPosition(view.captureCharFrame(), "See recommendations")
    await act(async () => view.mockMouse.click(recommendations.x, recommendations.y))
    await act(view.renderOnce)

    const model = textPosition(view.captureCharFrame(), recommendedModel.displayName)
    await act(async () => view.mockMouse.moveTo(model.x, model.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(model.x, model.y))

    expect(localInferenceActions.downloadModel).toHaveBeenCalledWith(recommendedModel.configurationId)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("Right Arrow opens recommendations from the usage questions", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    usage: { sessionConcurrency: "one" },
  }
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="onboarding" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("See recommendations (→)")
    await act(async () => view.mockInput.pressArrow("right"))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Choose what this machine should run")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("local and cloud onboarding actions can be clicked with the mouse", async () => {
  const onComplete = vi.fn()
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="onboarding" onExit={() => {}} onComplete={onComplete} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const localSkip = textPosition(view.captureCharFrame(), "Skip for now (Esc)")
    await act(async () => view.mockMouse.moveTo(localSkip.x, localSkip.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(localSkip.x, localSkip.y))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("CLOUD MODELS (OPTIONAL)")
    expect(view.captureCharFrame()).toContain("Connect cloud models too large to run on this machine")
    const cloudFrame = view.captureCharFrame()
    expect(cloudFrame).toContain("Use Exa web search for external research")
    expect(cloudFrame).not.toContain("Primary")
    expect(cloudFrame).not.toContain("Secondary")
    expect(cloudFrame).not.toContain("Ctrl+C close")
    expect(cloudFrame.indexOf("Press ← to return")).toBeLessThan(cloudFrame.indexOf("Back to local models (←)"))
    expect(cloudFrame.indexOf("Back to local models (←)")).toBeLessThan(cloudFrame.indexOf("Skip for now (Esc)"))

    const cloudBack = textPosition(view.captureCharFrame(), "Back to local models (←)")
    await act(async () => view.mockMouse.moveTo(cloudBack.x, cloudBack.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(cloudBack.x, cloudBack.y))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("How many local coding sessions will you run at once?")

    const localSkipAgain = textPosition(view.captureCharFrame(), "Skip for now (Esc)")
    await act(async () => view.mockMouse.click(localSkipAgain.x, localSkipAgain.y))
    await act(view.renderOnce)
    const cloudSkip = textPosition(view.captureCharFrame(), "Skip for now (Esc)")
    await act(async () => view.mockMouse.click(cloudSkip.x, cloudSkip.y))
    expect(onComplete).toHaveBeenCalledTimes(1)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("Ctrl+C exits from local model setup", async () => {
  const onExit = vi.fn()
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="onboarding" onExit={onExit} />,
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
