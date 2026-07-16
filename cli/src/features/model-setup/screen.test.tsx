import { beforeEach, expect, test, vi } from "vitest"
import { Atom, Result } from "@effect-atom/atom-react"
import { testRender } from "@opentui/react/test-utils"
import type { LocalInferenceState } from "@magnitudedev/sdk"
import { act } from "react"

const emptyLocalInferenceState = {
  schemaVersion: 3,
  usage: null,
  activeBinding: null,
  distribution: { _tag: "Missing" },
  host: { _tag: "Unavailable", message: "not needed" },
  choices: [],
  operations: [],
  recommendations: [],
  warnings: [],
} as const satisfies LocalInferenceState

let localInferenceState: LocalInferenceState = emptyLocalInferenceState

const textPosition = (frame: string, label: string): { x: number; y: number } => {
  const lines = frame.split("\n")
  const y = lines.findIndex((line) => line.includes(label))
  if (y < 0) throw new Error(`Could not find ${label}`)
  return { x: lines[y]!.indexOf(label) + 1, y }
}

beforeEach(() => {
  localInferenceState = emptyLocalInferenceState
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
    configureUsage: () => {},
    installDistribution: () => {},
    downloadModel: () => {},
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
    usage: { localModelRole: "main", sessionConcurrency: "one" },
  }
  const view = await testRender(
    <ModelSetupScreen initialStep="local" mode="management" onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("How do you plan to use local models?")
    expect(frame).toContain("● As my main agent")
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
    usage: { localModelRole: "main", sessionConcurrency: "one" },
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

test("Right Arrow opens recommendations from the usage questions", async () => {
  localInferenceState = {
    ...emptyLocalInferenceState,
    usage: { localModelRole: "main", sessionConcurrency: "one" },
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
    expect(view.captureCharFrame()).toContain("MAGNITUDE CLOUD FALLBACK")

    const cloudBack = textPosition(view.captureCharFrame(), "Back to local models (←)")
    await act(async () => view.mockMouse.moveTo(cloudBack.x, cloudBack.y))
    await act(view.renderOnce)
    await act(async () => view.mockMouse.click(cloudBack.x, cloudBack.y))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("How do you plan to use local models?")

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
