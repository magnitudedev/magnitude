import { act } from "react"
import { testRender } from "@opentui/react/test-utils"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { beforeEach, expect, test, vi } from "vitest"
import { GIB, makeModel, makeRecommendation, makeView } from "../local-inference/test-fixtures"

const actions = vi.hoisted(() => ({
  downloadRecommendedModel: vi.fn(),
  retryModelDownload: vi.fn(),
  cancelModelDownload: vi.fn(),
  dismissModelDownloadFailure: vi.fn(),
  deleteLocalModel: vi.fn(),
  assignSlot: vi.fn(),
  clearSlot: vi.fn(),
  loadModel: vi.fn(),
  unloadModel: vi.fn(),
}))
let state = makeView({ ready: false })

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useLocalInferenceState: () => ({
    state: Result.success(state),
    mutationFailure: Option.none(),
    ...actions,
  }),
}))
vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue", foreground: "white", muted: "gray", success: "green",
    error: "red", warning: "yellow", border: "gray",
  }),
}))

const { ModelSetupScreen } = await import("./screen")

const textPosition = (frame: string, label: string): { x: number; y: number } => {
  const lines = frame.split("\n")
  const y = lines.findIndex((line) => line.includes(label))
  if (y < 0) throw new Error(`Could not find ${label}`)
  return { x: lines[y]!.indexOf(label) + 1, y }
}

beforeEach(() => {
  state = makeView({ ready: false })
  for (const action of Object.values(actions)) action.mockClear()
})

test("preserves the local model setup screen instead of introducing a model-selection screen", async () => {
  const view = await testRender(
    <ModelSetupScreen mode="management" onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("LOCAL MODEL SETUP")
    expect(frame).toContain("HARDWARE DETECTED")
    expect(frame).not.toContain("SELECT A MODEL")
    expect(frame).not.toContain("primary slot")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("clicking an available inventory entry requests its download", async () => {
  const model = makeModel({
    download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
    preparation: { _tag: "NotDownloaded" },
  })
  const recommendation = makeRecommendation()
  state = makeView({ models: [model], recommendations: [recommendation], ready: false })
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const position = textPosition(view.captureCharFrame(), model.displayName)
    await act(async () => view.mockMouse.click(position.x, position.y))
    expect(actions.downloadRecommendedModel).toHaveBeenCalledWith(recommendation.id)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders download progress as one label followed by percentage", async () => {
  const model = makeModel()
  state = makeView({
    models: [{
      ...model,
      download: {
        _tag: "Downloading",
        completedBytes: model.downloadBytes / 4,
        totalBytes: model.downloadBytes,
      },
      preparation: { _tag: "NotDownloaded" },
    }],
    ready: false,
  })
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Downloading 25%")
    expect(frame).not.toContain("Downloading Downloading")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders consumer recommendation intent and its trade-off explanation", async () => {
  const model = makeModel({
    download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
    preparation: { _tag: "NotDownloaded" },
  })
  const recommendation = makeRecommendation({
    intent: "fastest",
    explanation: "Prioritizes responsive generation at about 42.0 tokens/sec.",
    fit: {
      requiredBytes: 12 * GIB,
      availableBytes: 32 * GIB,
      estimatedTokensPerSecond: Option.some(42),
    },
  })
  state = makeView({ models: [model], recommendations: [recommendation], ready: false })
  const view = await testRender(
    <ModelSetupScreen mode="onboarding" onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Fastest")
    expect(frame).toContain("Prioritizes responsive generation")
    expect(frame).not.toContain("Alternative Option")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
