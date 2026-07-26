import { act, useState } from "react"
import { testRender } from "@opentui/react/test-utils"
import { Result } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { beforeEach, expect, test, vi } from "vitest"
import {
  ModelSlotBlocked,
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLoading,
} from "@magnitudedev/sdk"
import {
  GIB,
  LOCAL_PROVIDER_ID,
  TEST_MEMORY_DOMAIN_ID,
  TEST_MODEL_ID,
  TEST_REASONING_EFFORT,
  makeCatalogCandidate,
  makeModel,
  makeRecommendation,
  makeView,
} from "../local-inference/test-fixtures"

const actions = vi.hoisted(() => ({
  downloadCatalogModel: vi.fn(),
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
let mutationFailure: Option.Option<Result.Result<unknown, unknown>> = Option.none()
let slotAssignment: Result.Result<unknown, unknown> = Result.initial()

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useLocalInferenceState: () => ({
    state: Result.success(state),
    slotAssignment,
    mutationFailure,
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
  mutationFailure = Option.none()
  slotAssignment = Result.initial()
  for (const action of Object.values(actions)) action.mockClear()
  actions.assignSlot.mockResolvedValue(undefined)
})

test("withholds provisional models and errors until curated setup is coherent", async () => {
  const provisionalName = "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
  const provisionalFailure = "The selected model is temporarily unavailable"
  const provisional = makeModel({
    displayName: provisionalName,
    preparation: {
      _tag: "Unavailable",
      providerModelIds: [],
      failure: {
        code: "model_unavailable",
        message: provisionalFailure,
        retryable: true,
      },
    },
  })
  const partial = makeView({ models: [provisional], ready: false })
  state = {
    ...partial,
    models: {
      models: [provisional],
      recommendations: {
        _tag: "Loading",
        progress: [
          {
            id: "hardware",
            status: {
              _tag: "Completed",
              startedAtMs: 1_000,
              durationMs: 100,
              cached: false,
            },
            completedItems: Option.some(1),
            totalItems: Option.some(1),
          },
          {
            id: "inventory",
            status: { _tag: "Running", startedAtMs: 1_100 },
            completedItems: Option.none(),
            totalItems: Option.none(),
          },
        ],
      },
    },
    catalog: new ProviderModelCatalogLoading(),
    slots: {
      ...partial.slots,
      slots: {
        ...partial.slots.slots,
        primary: new ModelSlotBlocked({
          slotId: PRIMARY_SLOT_ID,
          selection: {
            providerId: LOCAL_PROVIDER_ID,
            providerModelId: TEST_MODEL_ID,
            reasoningEffort: TEST_REASONING_EFFORT,
          },
          reason: { _tag: "ModelUnavailable", message: provisionalFailure },
        }),
      },
    },
  }
  let refresh: (() => void) | undefined
  const Harness = () => {
    const [, setRevision] = useState(0)
    refresh = () => setRevision((revision) => revision + 1)
    return <ModelSetupScreen onExit={() => {}} />
  }
  const view = await testRender(<Harness />, { width: 100, height: 30 })
  try {
    await act(view.renderOnce)
    const partialFrame = view.captureCharFrame()
    expect(partialFrame).toContain("SETUP PROGRESS")
    expect(partialFrame).toContain("Checking for downloaded models")
    expect(partialFrame).not.toContain(provisionalName)
    expect(partialFrame).not.toContain(provisionalFailure)
    expect(partialFrame).not.toContain("Unable to use this model")

    const curatedName = "Qwen3.6 35B-A3B"
    const curated = makeModel({
      displayName: curatedName,
      download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
      preparation: { _tag: "NotDownloaded" },
    })
    state = makeView({
      models: [curated],
      recommendations: [makeRecommendation()],
      ready: false,
    })
    await act(async () => refresh?.())
    await act(view.renderOnce)
    const readyFrame = view.captureCharFrame()
    expect(readyFrame).toContain(curatedName)
    expect(readyFrame).not.toContain(provisionalName)
    expect(readyFrame).not.toContain(provisionalFailure)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders a terminal discovery failure exactly once and no partial model cards", async () => {
  const failureMessage = "Curated model lookup failed"
  const partial = makeView({
    models: [makeModel({ displayName: "provisional-model.gguf" })],
    ready: false,
  })
  state = {
    ...partial,
    models: {
      ...partial.models,
      recommendations: {
        _tag: "Failed",
        failure: {
          code: "recommendations_unavailable",
          message: failureMessage,
          retryable: true,
        },
        progress: [{
          id: "hardware",
          status: {
            _tag: "Failed",
            startedAtMs: 1_000,
            durationMs: 250,
            failure: {
              code: "recommendations_unavailable",
              message: failureMessage,
              retryable: true,
            },
          },
          completedItems: Option.none(),
          totalItems: Option.none(),
        }],
      },
    },
  }
  const view = await testRender(
    <ModelSetupScreen onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame.split(failureMessage)).toHaveLength(2)
    expect(frame).not.toContain("provisional-model.gguf")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("shows a neutral finalizing state while the initial provider catalog settles", async () => {
  const partial = makeView({
    models: [makeModel({ displayName: "provisional-model.gguf" })],
    recommendations: [makeRecommendation()],
    ready: false,
  })
  state = {
    ...partial,
    catalog: new ProviderModelCatalogLoading(),
  }
  const view = await testRender(
    <ModelSetupScreen onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Finalizing model availability")
    expect(frame).not.toContain("provisional-model.gguf")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("preserves the local model setup screen instead of introducing a model-selection screen", async () => {
  const view = await testRender(
    <ModelSetupScreen onExit={() => {}} />,
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
    <ModelSetupScreen onExit={() => {}} />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const position = textPosition(view.captureCharFrame(), model.displayName)
    await act(async () => view.mockMouse.click(position.x, position.y))
    expect(actions.downloadCatalogModel).toHaveBeenCalledWith(recommendation.candidate.id)
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
    <ModelSetupScreen onExit={() => {}} />,
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
    candidate: makeCatalogCandidate({
      memory: [{
        memoryDomainId: TEST_MEMORY_DOMAIN_ID,
        capacityBytes: 32 * GIB,
        requiredBytes: 12 * GIB,
        compatibilityReserveBytes: 2 * GIB,
        warningReserveBytes: 4 * GIB,
        remainingBytes: 18 * GIB,
      }],
      estimatedTokensPerSecond: Option.some(42),
    }),
  })
  state = makeView({ models: [model], recommendations: [recommendation], ready: false })
  const view = await testRender(
    <ModelSetupScreen onExit={() => {}} />,
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

test("keeps setup open until the slot-assignment mutation succeeds", async () => {
  state = makeView({ ready: false })
  let refresh: (() => void) | undefined
  let resolveAssignment: (() => void) | undefined
  actions.assignSlot.mockReturnValue(new Promise<void>((resolve) => {
    resolveAssignment = resolve
  }))
  const onComplete = vi.fn()
  const Harness = () => {
    const [, setRevision] = useState(0)
    refresh = () => setRevision((revision) => revision + 1)
    return <ModelSetupScreen onExit={() => {}} onComplete={onComplete} />
  }
  const view = await testRender(<Harness />, { width: 100, height: 30 })
  try {
    await act(view.renderOnce)
    const position = textPosition(view.captureCharFrame(), "Qwen Test")
    await act(async () => view.mockMouse.click(position.x, position.y))
    expect(actions.assignSlot).toHaveBeenCalledTimes(1)
    expect(onComplete).not.toHaveBeenCalled()

    slotAssignment = Result.initial(true)
    await act(async () => refresh?.())
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Selecting this model")

    await act(async () => view.mockMouse.click(position.x, position.y))
    expect(actions.assignSlot).toHaveBeenCalledTimes(1)

    resolveAssignment?.()
    await act(async () => Promise.resolve())
    expect(onComplete).toHaveBeenCalledTimes(1)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("keeps setup open and shows an assignment failure", async () => {
  state = makeView({ ready: false })
  actions.assignSlot.mockRejectedValue(new Error("assignment failed"))
  let refresh: (() => void) | undefined
  const onComplete = vi.fn()
  const Harness = () => {
    const [, setRevision] = useState(0)
    refresh = () => setRevision((revision) => revision + 1)
    return <ModelSetupScreen onExit={() => {}} onComplete={onComplete} />
  }
  const view = await testRender(<Harness />, { width: 100, height: 30 })
  try {
    await act(view.renderOnce)
    const position = textPosition(view.captureCharFrame(), "Qwen Test")
    await act(async () => view.mockMouse.click(position.x, position.y))

    const failure = Result.failure(Cause.fail(new Error("assignment failed")))
    mutationFailure = Option.some(failure)
    await act(async () => refresh?.())
    await act(view.renderOnce)

    expect(onComplete).not.toHaveBeenCalled()
    expect(view.captureCharFrame()).toContain("assignment failed")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
