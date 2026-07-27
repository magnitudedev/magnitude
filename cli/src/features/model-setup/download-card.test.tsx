import { act } from "react"
import { KeyEvent } from "@opentui/core"
import { testRender } from "@opentui/react/test-utils"
import { Option } from "effect"
import { beforeEach, expect, test, vi } from "vitest"
import { makeCatalogCandidate, GIB } from "../local-inference/test-fixtures"

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

const { OnboardingModelDownloadCard } = await import("./download-card")

const onCancel = vi.fn()
const onRetry = vi.fn()
const candidate = makeCatalogCandidate({
  displayName: "Qwen Test",
  quantization: "Q6_K",
  downloadBytes: 30 * GIB,
  download: {
    _tag: "Downloading",
    stage: "downloading",
    completedBytes: 19 * GIB,
    totalBytes: 30 * GIB,
    bytesPerSecond: Option.some(48 * 1024 ** 2),
  },
})

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

test("shows one centered download summary with progress, rate, and ETA", async () => {
  const view = await testRender(
    <OnboardingModelDownloadCard
      candidate={candidate}
      width={100}
      cancelling={false}
      cancelError={null}
      onCancel={onCancel}
      onRetry={onRetry}
    />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Downloading Qwen Test · Q6_K")
    expect(frame).toContain("63%")
    expect(frame).toContain("19 GB / 30 GB")
    expect(frame).toContain("48 MB/s · about 4 minutes remaining")
    expect(frame).toContain("Cancel (Esc)")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("requires confirmation before cancelling and supports keyboard choice", async () => {
  const view = await testRender(
    <OnboardingModelDownloadCard
      candidate={candidate}
      width={100}
      cancelling={false}
      cancelError={null}
      onCancel={onCancel}
      onRetry={onRetry}
    />,
    { width: 100, height: 24 },
  )
  try {
    await act(view.renderOnce)
    await act(async () => press("escape"))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Are you sure you want to cancel?")
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
