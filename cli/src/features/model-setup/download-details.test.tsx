import { act } from "react"
import { KeyEvent } from "@opentui/core"
import { testRender } from "@opentui/react/test-utils"
import { Option } from "effect"
import { beforeEach, expect, test, vi } from "vitest"
import { DownloadAttemptIdSchema, ModelVariantLabelSchema } from "@magnitudedev/sdk"
import { makeAcquiringModel, GIB } from "../local-inference/test-fixtures"

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
    border: "gray",
  }),
}))

const { OnboardingModelDownloadDetails } = await import("./download-details")

const onCancel = vi.fn()
const onRetry = vi.fn()
const model = makeAcquiringModel(
  {
    _tag: "Downloading",
    attemptIds: [DownloadAttemptIdSchema.make("download_qwen")],
    stage: "downloading",
    completedBytes: 19 * GIB,
    totalBytes: 30 * GIB,
    bytesPerSecond: Option.some(48 * 1024 ** 2),
  },
  {
    presentation: {
      displayName: "Qwen Test",
      variantLabel: ModelVariantLabelSchema.make("Q6"),
      description: "Test model",
      license: Option.none(),
      quantization: "Q6_K",
      precisionLabel: "6-bit",
    },
  },
)

beforeEach(() => {
  onCancel.mockClear()
  onRetry.mockClear()
})

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

test("shows compact download details with progress, rate, and ETA", async () => {
  const view = await testRender(
    <OnboardingModelDownloadDetails
      model={model}
      width={56}
      height={11}
      operation={{
        _tag: "Failed",
        onChooseAnother: onCancel,
        onRetry,
      }}
    />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Downloading Qwen Test (Q6) · Q6_K")
    expect(frame).toContain("63%")
    expect(frame).toContain("20.4 GB / 32.2 GB")
    expect(frame).toContain("48 MB/s · about 4 minutes remaining")
    expect(frame).toContain("Cancel (Esc)")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("shows zero-percent download details while admission is starting", async () => {
  const startingModel = makeAcquiringModel({
    _tag: "NotInstalled",
    completedBytes: 0,
    totalBytes: 30 * GIB,
  })
  const view = await testRender(
    <OnboardingModelDownloadDetails
      model={startingModel}
      width={56}
      height={11}
      operation={{
        _tag: "Active",
        starting: true,
        cancelling: false,
        cancelError: null,
        onCancel,
        onRetry,
      }}
    />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Downloading Qwen Test")
    expect(frame).toContain("0%")
    expect(frame).not.toContain("0 B /")
    expect(frame).not.toContain("Cancel (Esc)")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("requires confirmation before cancelling and supports keyboard choice", async () => {
  const view = await testRender(
    <OnboardingModelDownloadDetails
      model={model}
      width={56}
      height={11}
      operation={{
        _tag: "Active",
        starting: false,
        cancelling: false,
        cancelError: null,
        onCancel,
        onRetry,
      }}
    />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    await act(async () => press("escape"))
    await act(view.renderOnce)
    const confirmationFrame = view.captureCharFrame()
    expect(confirmationFrame).toContain("Are you sure you want to cancel?")
    expect(confirmationFrame).toMatch(/Yes\s+No/)
    expect(onCancel).not.toHaveBeenCalled()

    await act(async () => press("right"))
    await act(async () => press("return"))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Cancel (Esc)")
    expect(onCancel).not.toHaveBeenCalled()

    await act(async () => press("escape"))
    await act(async () => press("return"))
    expect(onCancel).toHaveBeenCalledTimes(1)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("shows failed-download actions in the details pane", async () => {
  const failedModel = makeAcquiringModel({
      _tag: "Failed",
      attemptIds: [DownloadAttemptIdSchema.make("download_failed")],
      completedBytes: 19 * GIB,
      totalBytes: 30 * GIB,
      failure: {
        code: "transport_failed",
        message: "Download failed",
        retryable: true,
      },
  })
  const view = await testRender(
    <OnboardingModelDownloadDetails
      model={failedModel}
      width={56}
      height={11}
      operation={{
        _tag: "Active",
        starting: false,
        cancelling: false,
        cancelError: null,
        onCancel,
        onRetry,
      }}
    />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Couldn’t download Qwen Test")
    expect(frame).toContain("Download failed")
    expect(frame).not.toContain("63%")
    expect(frame).toMatch(/Retry\s+Choose another model/)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
