import { act, useState } from "react"
import { KeyEvent } from "@opentui/core"
import { testRender } from "@opentui/react/test-utils"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  CatalogCandidateIdSchema,
  ModelOfferingTargetIdSchema,
  ProviderModelIdSchema,
  RecommendationIdSchema,
} from "@magnitudedev/sdk"
import { beforeEach, expect, test, vi } from "vitest"
import {
  makeCatalogCandidate,
  makeModel,
  makeRecommendation,
  makeView,
  GIB,
} from "../local-inference/test-fixtures"

const keyboard = vi.hoisted(() => ({
  handler: undefined as ((key: KeyEvent) => void) | undefined,
}))

vi.mock("@opentui/react", async (importOriginal) => ({
  ...await importOriginal<typeof import("@opentui/react")>(),
  useKeyboard: (handler: (key: KeyEvent) => void) => { keyboard.handler = handler },
}))

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    error: "red",
    warning: "yellow",
    success: "green",
    border: "gray",
  }),
}))

const { OnboardingModelChooser, OnboardingModelPreparation } = await import("./chooser")

const onLoad = vi.fn()
const onDownload = vi.fn()
const onContinue = vi.fn()
const onSkip = vi.fn()

const press = (name: string) => keyboard.handler?.(new KeyEvent({
  name,
  ctrl: false,
  meta: false,
  shift: false,
  option: false,
  sequence: "",
  number: false,
  raw: "",
  eventType: "press",
  source: "raw",
}))

beforeEach(() => {
  onLoad.mockClear()
  onDownload.mockClear()
  onContinue.mockClear()
  onSkip.mockClear()
})

const chooserView = () => {
  const remoteCandidateId = CatalogCandidateIdSchema.make("candidate_remote")
  const remoteTargetId = ModelOfferingTargetIdSchema.make("target_remote")
  const remoteProviderModelId = ProviderModelIdSchema.make("configuration_remote")
  const remoteModel = makeModel({
    targetId: remoteTargetId,
    catalogCandidateIds: [remoteCandidateId],
    displayName: "Remote Model",
    download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * 1024 ** 3 },
    preparation: { _tag: "NotDownloaded" },
  })
  return makeView({
    ready: false,
    models: [makeModel({ displayName: "Installed Model" }), remoteModel],
    recommendations: [makeRecommendation({
      candidate: makeCatalogCandidate({
        id: remoteCandidateId,
        targetId: remoteTargetId,
        providerModelId: remoteProviderModelId,
        displayName: "Remote Model",
        profile: { contextLength: 200_000 },
        estimatedTokensPerSecond: 36.2,
      }),
    })],
  })
}

const chooserProps = (state: ReturnType<typeof chooserView>) => ({
  hardware: Result.success(state.hardware),
  models: state.models,
  catalog: state.catalog,
  slots: state.slots,
  onLoad,
  onDownload,
})

const chooserViewWithInventory = (installedCount: number, downloadCount: number) => {
  const installed = Array.from({ length: installedCount }, (_, index) => makeModel({
    targetId: ModelOfferingTargetIdSchema.make(`target_installed_${index + 1}`),
    displayName: `Installed ${index + 1}`,
  }))
  const downloads = Array.from({ length: downloadCount }, (_, index) => {
    const number = index + 1
    const intents = ["balanced", "best_quality", "fastest", "lightweight"] as const
    const candidateId = CatalogCandidateIdSchema.make(`candidate_remote_${number}`)
    const targetId = ModelOfferingTargetIdSchema.make(`target_remote_${number}`)
    const providerModelId = ProviderModelIdSchema.make(`configuration_remote_${number}`)
    const model = makeModel({
      targetId,
      catalogCandidateIds: [candidateId],
      displayName: `Remote ${number}`,
      download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * 1024 ** 3 },
      preparation: { _tag: "NotDownloaded" },
    })
    const recommendation = makeRecommendation({
      id: RecommendationIdSchema.make(`recommendation_remote_${number}`),
      intent: intents[index] ?? "balanced",
      explanation: number === downloadCount
        ? "A deliberately long recommendation description that wraps across several lines without changing the chooser height when this model becomes selected."
        : "Balanced local inference.",
      candidate: makeCatalogCandidate({
        id: candidateId,
        targetId,
        providerModelId,
        displayName: `Remote ${number}`,
      }),
    })
    return { model, recommendation }
  })
  return makeView({
    ready: false,
    models: [...installed, ...downloads.map(({ model }) => model)],
    recommendations: downloads.map(({ recommendation }) => recommendation),
  })
}

test("renders compact installed and downloadable rows with an informational detail pane", async () => {
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={100}
      error={null}
      operation={null}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("ON THIS COMPUTER")
    expect(frame).toContain("Installed Model")
    expect(frame).toContain("Load")
    expect(frame).toContain("AVAILABLE TO DOWNLOAD")
    expect(frame).toContain("Remote Model")
    expect(frame).toMatch(/Remote Model\s+Balanced/)
    expect(frame).not.toMatch(/Remote Model\s+Download/)
    expect(frame).not.toContain("Download & load")
    expect(frame).not.toContain("[ Load ]")
    expect(frame).toContain("Enter select")
    await act(async () => press("down"))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("~36 tok/s at 200K ctx")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("loads a stored model without invoking download", async () => {
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={100}
      error={null}
      operation={null}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    await act(async () => press("enter"))
    expect(onLoad).toHaveBeenCalledWith(expect.objectContaining({
      displayName: "Installed Model",
      reasoningEffort: "none",
    }))
    expect(onDownload).not.toHaveBeenCalled()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("does not accept another selection while an operation is active", async () => {
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={100}
      error={null}
      operation={{
        _tag: "Activating",
        providerModelId: ProviderModelIdSchema.make("configuration_installed"),
        displayName: "Installed Model",
        phase: "Loading",
        failure: null,
        onRetry: vi.fn(),
        onChooseAnother: vi.fn(),
      }}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    await act(async () => press("down"))
    await act(async () => press("enter"))
    expect(onLoad).not.toHaveBeenCalled()
    expect(onDownload).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("shows hardware detection in the persistent metadata position before state arrives", async () => {
  const view = await testRender(
    <OnboardingModelPreparation hardware={Result.initial()} progress={[]} error={null} width={100} onSkip={onSkip} />,
    { width: 100, height: 16 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Preparing local models")
    expect(frame).toContain("Detecting hardware…")
    expect(frame).not.toContain("ON THIS COMPUTER")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("replaces hardware progress with persistent left-aligned machine metadata", async () => {
  const base = chooserView()
  const state = {
    ...base,
    models: {
      ...base.models,
      recommendations: {
        _tag: "Loading" as const,
        progress: [
          {
            id: "hardware" as const,
            status: {
              _tag: "Completed" as const,
              startedAtMs: 1_000,
              durationMs: 10,
              cached: false,
            },
            completedItems: Option.some(1),
            totalItems: Option.some(1),
            estimatedRemainingMs: Option.none(),
          },
          {
            id: "inventory" as const,
            status: {
              _tag: "Completed" as const,
              startedAtMs: 1_010,
              durationMs: 20,
              cached: false,
            },
            completedItems: Option.some(1),
            totalItems: Option.some(1),
            estimatedRemainingMs: Option.none(),
          },
          {
            id: "analysis" as const,
            status: { _tag: "Running" as const, startedAtMs: 1_030 },
            completedItems: Option.some(12),
            totalItems: Option.some(20),
            estimatedRemainingMs: Option.none(),
          },
          {
            id: "recommendations" as const,
            status: { _tag: "Pending" as const },
            completedItems: Option.none(),
            totalItems: Option.none(),
            estimatedRemainingMs: Option.none(),
          },
        ],
      },
    },
  }
  const view = await testRender(
    <OnboardingModelPreparation hardware={Result.success(state.hardware)} progress={state.models.recommendations.progress} error={null} width={100} onSkip={onSkip} />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    const rows = frame.split("\n")
    const hardwareRow = rows.find((line) => line.includes("Test CPU"))
    const progressRow = rows.find((line) => line.includes("Found 1 downloaded model"))
    expect(hardwareRow).toContain("Linux x86-64 · 16 cores · 64 GiB RAM")
    expect(frame).toContain("Test GPU · 24 GiB VRAM · CUDA")
    expect(frame).not.toContain("Detected hardware")
    expect(frame).toContain("Evaluating models for this machine · 12/20")
    expect(frame).toContain("Preparing recommendations")
    expect(hardwareRow?.indexOf("Test CPU")).toBe(progressRow?.indexOf("✓"))
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("keeps the chooser unchanged until authoritative download state arrives", async () => {
  const candidate = makeCatalogCandidate({
    id: CatalogCandidateIdSchema.make("candidate_remote"),
    targetId: ModelOfferingTargetIdSchema.make("target_remote"),
    providerModelId: ProviderModelIdSchema.make("configuration_remote"),
    displayName: "Remote Model",
    download: {
      _tag: "Downloading",
      stage: "downloading",
      completedBytes: 8 * GIB,
      totalBytes: 16 * GIB,
      bytesPerSecond: Option.some(32 * 1024 ** 2),
    },
  })
  let publishAuthoritativeDownload: () => void = () => undefined
  const TransitionHarness = () => {
    const [downloading, setDownloading] = useState(false)
    publishAuthoritativeDownload = () => setDownloading(true)
    return (
      <OnboardingModelChooser
        {...chooserProps(chooserView())}
        width={100}
        error={null}
        operation={downloading ? {
          _tag: "Downloading",
          candidate,
          cancelling: false,
          cancelError: null,
          onCancel: vi.fn(),
          onRetry: vi.fn(),
        } : null}
        onContinue={onContinue}
        onSkip={onSkip}
      />
    )
  }
  const view = await testRender(
    <TransitionHarness />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    await act(async () => press("down"))
    await act(async () => press("enter"))
    expect(onDownload).toHaveBeenCalledWith({
      targetId: candidate.targetId,
      providerModelId: candidate.providerModelId,
      displayName: candidate.displayName,
      reasoningEffort: "none",
    })
    await act(view.renderOnce)
    const preMirrorFrame = view.captureCharFrame()
    expect(preMirrorFrame).toContain("Balanced local inference")
    expect(preMirrorFrame).not.toContain("Downloading Remote Model")

    await act(async () => publishAuthoritativeDownload())
    await act(view.renderOnce)
    const downloadingFrame = view.captureCharFrame()
    expect(downloadingFrame).toContain("50%")
    expect(downloadingFrame).toContain("8.59 GB / 17.2 GB")
    expect(downloadingFrame).not.toContain("Balanced local inference")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("keeps the chosen row highlighted and locks navigation while download details are active", async () => {
  const candidate = makeCatalogCandidate({
    id: CatalogCandidateIdSchema.make("candidate_remote"),
    displayName: "Remote Model",
    downloadBytes: 16 * GIB,
    download: {
      _tag: "Downloading",
      stage: "downloading",
      completedBytes: 8 * GIB,
      totalBytes: 16 * GIB,
      bytesPerSecond: Option.some(32 * 1024 ** 2),
    },
  })
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={70}
      error={null}
      operation={{
        _tag: "Downloading",
        candidate,
        cancelling: false,
        cancelError: null,
        onCancel: vi.fn(),
        onRetry: vi.fn(),
      }}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 70, height: 40 },
  )
  try {
    await act(view.renderOnce)
    const initialFrame = view.captureCharFrame()
    expect(initialFrame).toMatch(/› Remote Model\s+Balanced/)
    expect(initialFrame).toContain("Downloading Remote Model")
    expect(initialFrame).toContain("Test CPU · Linux x86-64 · 16 cores · 64 GiB RAM")
    expect(initialFrame).toContain("Test GPU · 24 GiB VRAM · CUDA")
    expect(initialFrame).toContain("50%")
    expect(initialFrame).toContain("8.59 GB / 17.2 GB")
    expect(initialFrame).toContain("32 MB/s")
    expect(initialFrame).toContain("Cancel (Esc)")
    expect(initialFrame).toContain("Download in progress · Esc cancel")
    expect(initialFrame).not.toContain("Enter select")

    await act(async () => press("up"))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toMatch(/› Remote Model\s+Balanced/)
    expect(onLoad).not.toHaveBeenCalled()
    expect(onDownload).not.toHaveBeenCalled()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders the authoritative loading lifecycle", async () => {
  vi.useFakeTimers()
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={100}
      error={null}
      operation={{
        _tag: "Activating",
        providerModelId: ProviderModelIdSchema.make("configuration_remote"),
        displayName: "Remote Model",
        phase: "Loading",
        failure: null,
        onRetry: vi.fn(),
        onChooseAnother: vi.fn(),
      }}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 34 },
  )
  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Loading model weights…")

    await act(async () => {
      vi.advanceTimersByTime(8_000)
    })
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Loading model weights…")
  } finally {
    await act(async () => view.renderer.destroy())
    vi.useRealTimers()
  }
})

test("renders the authoritative stopping lifecycle", async () => {
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={100}
      error={null}
      operation={{
        _tag: "Activating",
        providerModelId: ProviderModelIdSchema.make("configuration_remote"),
        displayName: "Remote Model",
        phase: "Stopping",
        failure: null,
        onRetry: vi.fn(),
        onChooseAnother: vi.fn(),
      }}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 34 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Stopping Remote Model")
    expect(frame).toContain("Stopping model…")
    expect(frame).not.toContain("Loading model weights…")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("keeps four rows per section and scrolls only the installed-model window", async () => {
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserViewWithInventory(6, 4))}
      width={100}
      error={null}
      operation={null}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 34 },
  )
  try {
    await act(view.renderOnce)
    const initialFrame = view.captureCharFrame()
    expect(initialFrame).toContain("Installed 1")
    expect(initialFrame).toContain("Installed 4")
    expect(initialFrame).not.toContain("Installed 5")
    expect(initialFrame).not.toContain("Installed 6")
    expect(initialFrame).toContain("Remote 1")
    expect(initialFrame).toContain("Remote 4")
    expect(initialFrame).toMatch(/Remote 1\s+Balanced/)
    expect(initialFrame).toMatch(/Remote 2\s+Best Quality/)
    expect(initialFrame).toMatch(/Remote 3\s+Fastest/)
    expect(initialFrame).toMatch(/Remote 4\s+Lightweight/)

    for (let index = 0; index < 5; index += 1) {
      await act(async () => press("down"))
    }
    await act(view.renderOnce)
    const installedScrolledFrame = view.captureCharFrame()
    expect(installedScrolledFrame).not.toContain("Installed 1")
    expect(installedScrolledFrame).not.toContain("Installed 2")
    expect(installedScrolledFrame).toContain("Installed 3")
    expect(installedScrolledFrame).toContain("Installed 6")
    expect(installedScrolledFrame).toContain("Remote 1")
    expect(installedScrolledFrame).toContain("Remote 4")

    for (let index = 0; index < 4; index += 1) {
      await act(async () => press("down"))
    }
    await act(view.renderOnce)
    const recommendationFrame = view.captureCharFrame()
    expect(recommendationFrame).toMatch(/Remote 4\s+Lightweight/)
    expect(recommendationFrame).toContain("that wraps across several lines")
    expect(recommendationFrame).toContain("selected.")
    expect(recommendationFrame.split("\n").findIndex((line) => line.includes("╰")))
      .toBe(initialFrame.split("\n").findIndex((line) => line.includes("╰")))
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("excludes unrelated failed downloads and unusable installed models", async () => {
  const base = chooserViewWithInventory(0, 4)
  const state = {
    ...base,
    models: {
      ...base.models,
      models: [
        ...base.models.models,
        makeModel({
          targetId: ModelOfferingTargetIdSchema.make("target_failed"),
          displayName: "Unrelated Failed Download",
          download: {
            _tag: "Failed" as const,
            completedBytes: 4 * GIB,
            totalBytes: 16 * GIB,
            failure: {
              code: "download_failed",
              message: "Download failed",
              retryable: true,
            },
          },
          preparation: { _tag: "NotDownloaded" as const },
        }),
        makeModel({
          targetId: ModelOfferingTargetIdSchema.make("target_unusable"),
          displayName: "Unusable Installed Model",
          preparation: {
            _tag: "Unavailable" as const,
            providerModelIds: [],
            failure: {
              code: "does_not_fit",
              message: "Requires more memory",
              retryable: false,
            },
          },
        }),
      ],
    },
  }
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(state)}
      width={100}
      error={null}
      operation={null}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 100, height: 30 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).not.toContain("Unrelated Failed Download")
    expect(frame).not.toContain("Unusable Installed Model")

    for (let index = 0; index < 4; index += 1) {
      await act(async () => press("down"))
    }
    await act(async () => press("enter"))
    expect(onDownload).toHaveBeenCalledWith({
      targetId: ModelOfferingTargetIdSchema.make("target_remote_4"),
      providerModelId: ProviderModelIdSchema.make("configuration_remote_4"),
      displayName: "Remote 4",
      reasoningEffort: "none",
    })
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("stacks the informational pane without clipping actions on narrower terminals", async () => {
  const view = await testRender(
    <OnboardingModelChooser
      {...chooserProps(chooserView())}
      width={70}
      error={null}
      operation={null}
      onContinue={onContinue}
      onSkip={onSkip}
    />,
    { width: 70, height: 34 },
  )
  try {
    await act(view.renderOnce)
    await act(async () => press("down"))
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Installed Model")
    expect(frame).toContain("Load")
    expect(frame).toContain("Remote Model")
    expect(frame).toMatch(/Remote Model\s+Balanced/)
    expect(frame).toContain("Q4_K_M · 17.2 GB · 200K ctx")
    expect(frame).toContain("Balanced local inference.")
    expect(frame).toContain("Test CPU")
    expect(frame).toContain("Test GPU · 24 GiB VRAM · CUDA")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
