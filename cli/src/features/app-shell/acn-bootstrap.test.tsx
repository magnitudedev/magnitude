import { act } from "react"
import { KeyEvent } from "@opentui/core"
import { testRender } from "@opentui/react/test-utils"
import { Option } from "effect"
import { expect, test, vi } from "vitest"

const keyboard = vi.hoisted(
  (): { handler: ((key: KeyEvent) => void) | undefined } => ({
    handler: undefined,
  }),
)

vi.mock("@opentui/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@opentui/react")>()),
  useKeyboard: (handler: (key: KeyEvent) => void) => {
    keyboard.handler = handler
  },
}))

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    error: "red",
    background: "black",
  }),
}))

const { AcnBootstrapScreen } = await import("./acn-bootstrap")

test.each([
  ["Discovering", "Looking for Magnitude"],
  ["WaitingForOwner", "Waiting for previous Magnitude process"],
  ["LaunchingAcn", "Starting Magnitude"],
  ["ResolvingLocalInference", "Preparing local inference"],
  ["LaunchingLocalInference", "Starting local inference"],
] as const)("renders the %s startup phase", async (phase, label) => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{ _tag: "Starting", phase }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain(label)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders CUDA backend preparation with its hardware label", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{
        _tag: "Starting",
        phase: {
          _tag: "PreparingBackend",
          backend: { _tag: "Cuda", hardwareLabel: "NVIDIA GeForce RTX 3060" },
        },
      }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain(
      "Preparing CUDA backend for NVIDIA GeForce RTX 3060",
    )
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders Metal backend preparation on a Mac", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{
        _tag: "Starting",
        phase: {
          _tag: "PreparingBackend",
          backend: { _tag: "Metal", hardwareLabel: "Apple M4 Max" },
        },
      }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain(
      "Preparing Metal backend for Apple M4 Max",
    )
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

const keyEvent = (name: string, ctrl = false) =>
  new KeyEvent({
    name,
    ctrl,
    meta: false,
    shift: false,
    option: false,
    sequence: "",
    number: false,
    raw: "",
    eventType: "press",
    source: "raw",
  })

test("renders exactly one empty row between the installation title and bar", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{
        _tag: "Installing",
        phase: "DownloadingInferenceEngine",
        overallProgress: 0.88,
        detailIsExact: true,
        detail: Option.some({
          completed: 19 * 1024 * 1024,
          totalBytes: 19.7 * 1024 * 1024,
          unit: "Bytes",
          attempt: Option.some(1),
        }),
      }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    const lines = view.captureCharFrame().split("\n")
    const titleRow = lines.findIndex((line) =>
      line.includes("Installing Magnitude"),
    )
    const barRow = lines.findIndex((line) => line.includes("88%"))

    expect(barRow - titleRow).toBe(2)
    expect(lines[titleRow + 1]?.trim()).toBe("")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("quits on Ctrl-C during startup without consuming unrelated keys", async () => {
  const onQuit = vi.fn()
  const view = await testRender(
    <AcnBootstrapScreen
      state={{ _tag: "Starting", phase: "LaunchingAcn" }}
      onRetry={() => undefined}
      onQuit={onQuit}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    const unrelated = keyEvent("x")
    keyboard.handler?.(unrelated)
    expect(unrelated.defaultPrevented).toBe(false)

    const ctrlC = keyEvent("c", true)
    keyboard.handler?.(ctrlC)
    expect(ctrlC.defaultPrevented).toBe(true)
    expect(onQuit).toHaveBeenCalledOnce()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
